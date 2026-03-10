import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from db.supabase import get_client
from services.embeddings import embed
from services.conversations import check_cache, save_conversation
from services.agents.graph import chat_graph
from services.agents.state import ChatState

logger = logging.getLogger(__name__)
router = APIRouter()

QUESTION_MAX_LEN = 1000


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    username: str
    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_LEN)
    history: list[HistoryMessage] = []
    config: dict
    sessionId: str | None = None


@router.post("/api/chat")
async def chat(req: ChatRequest):
    supabase = get_client()

    result = supabase.table("users").select("*").eq("username", req.username).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")
    user = result.data

    # ── 1. 임베딩 (캐시 체크에도 필요) ────────────────────────────────────
    try:
        query_embedding = embed(req.question)
    except Exception as e:
        logger.error("embed() failed: %s", e)
        raise HTTPException(status_code=503, detail="임베딩 서비스에 일시적인 문제가 발생했습니다.")

    # ── 2. 캐시 체크 ───────────────────────────────────────────────────────
    cached = None
    try:
        cached = check_cache(user["id"], query_embedding)
    except Exception as e:
        logger.warning("cache check failed: %s", e)

    # ── 3. 캐시 히트 → 즉시 반환 ──────────────────────────────────────────
    if cached:
        async def stream_cached():
            yield f"data: {json.dumps({'type': 'cache_hit'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': cached.answer}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_cached(),
            media_type="text/event-stream; charset=utf-8",
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
        )

    # ── 4. LangGraph initial state ─────────────────────────────────────────
    initial_state: ChatState = {
        "username": req.username,
        "question": req.question,
        "config": req.config,
        "history": [{"role": m.role, "content": m.content} for m in req.history],
        "session_id": req.sessionId,
        "user_data": user,
        "query_embedding": query_embedding,  # 재계산 방지
        "search_results": [],
        "context": "",
        "graph_fallback": False,
        "citations": [],
        "pinned_qa_context": "",
        "draft_answer": "",
        "fact_check_passed": True,
        "fact_check_notes": "",
        "final_answer": "",
    }

    # ── 5. SSE 스트리밍 (astream_events) ──────────────────────────────────
    async def generate():
        final_state: dict = {}
        draft_answer = ""

        try:
            async for event in chat_graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event.get("name", "")
                meta = event.get("metadata", {})

                # 각 노드 시작 → 상태 전송
                if kind == "on_chain_start" and name in ("retrieval", "persona", "factcheck"):
                    phase_map = {"retrieval": "retrieval", "persona": "generating", "factcheck": "factcheck"}
                    yield f"data: {json.dumps({'type': 'status', 'phase': phase_map[name]}, ensure_ascii=False)}\n\n"

                # retrieval 노드 완료 → citations + graph_fallback 전송
                elif kind == "on_chain_end" and name == "retrieval":
                    output = event["data"].get("output", {})
                    final_state.update(output)
                    if output.get("graph_fallback"):
                        yield f"data: {json.dumps({'type': 'graph_fallback', 'used': True}, ensure_ascii=False)}\n\n"
                    citations = output.get("citations", [])
                    yield f"data: {json.dumps({'type': 'citations', 'sources': citations}, ensure_ascii=False)}\n\n"

                # persona 노드 LLM 토큰 실시간 스트리밍
                elif kind == "on_chat_model_stream" and meta.get("langgraph_node") == "persona":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        draft_answer += chunk.content
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk.content}, ensure_ascii=False)}\n\n"

                # factcheck 노드 완료 → fact_check_warn 전송
                elif kind == "on_chain_end" and name == "factcheck":
                    output = event["data"].get("output", {})
                    final_state.update(output)
                    if not output.get("fact_check_passed", True):
                        yield f"data: {json.dumps({'type': 'fact_check_warn', 'notes': output.get('fact_check_notes', '')}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("astream_events failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': '스트리밍 오류가 발생했습니다.'}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        # ── 6. 백그라운드 저장 ─────────────────────────────────────────────
        final_answer = final_state.get("final_answer") or draft_answer
        if final_answer:
            try:
                save_conversation(
                    user_id=user["id"],
                    session_id=req.sessionId,
                    question=req.question,
                    answer=final_answer,
                    interviewer_config=req.config,
                    question_embedding=query_embedding,
                )
            except Exception as e:
                logger.warning("save_conversation failed: %s", e)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )
