import json
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
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
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
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

    # ── 4. LangGraph 실행 ──────────────────────────────────────────────────
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
        "draft_answer": "",
        "fact_check_passed": True,
        "fact_check_notes": "",
        "final_answer": "",
    }

    try:
        final_state = await chat_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("LangGraph execution failed: %s", e)
        raise HTTPException(status_code=503, detail="서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.")

    # ── 5. 백그라운드 저장 ─────────────────────────────────────────────────
    def _save() -> None:
        try:
            save_conversation(
                user_id=user["id"],
                session_id=req.sessionId,
                question=req.question,
                answer=final_state["final_answer"],
                interviewer_config=req.config,
                question_embedding=query_embedding,
            )
        except Exception as e:
            logger.warning("save_conversation failed: %s", e)

    background_tasks.add_task(_save)

    # ── 6. SSE 스트리밍 ────────────────────────────────────────────────────
    async def generate():
        # 그래프 폴백 알림
        if final_state["graph_fallback"]:
            yield f"data: {json.dumps({'type': 'graph_fallback', 'used': True}, ensure_ascii=False)}\n\n"

        # 팩트체크 경고
        if not final_state["fact_check_passed"]:
            yield f"data: {json.dumps({'type': 'fact_check_warn', 'notes': final_state['fact_check_notes']}, ensure_ascii=False)}\n\n"

        # 출처
        yield f"data: {json.dumps({'type': 'citations', 'sources': final_state['citations']}, ensure_ascii=False)}\n\n"

        # 답변 청크 스트리밍 (단어 단위)
        answer = final_state["final_answer"]
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )
