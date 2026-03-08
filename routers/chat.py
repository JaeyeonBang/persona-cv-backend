import json
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from db.supabase import get_client
from services.embeddings import embed
from services.search import search_documents, build_context
from services.llm import build_system_prompt, stream_chat
from services.conversations import check_cache, save_conversation

router = APIRouter()


class ChatRequest(BaseModel):
    username: str
    question: str
    config: dict
    sessionId: str | None = None


@router.post("/api/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    supabase = get_client()

    # 유저 조회 (username 기준)
    result = supabase.table("users").select("*").eq("username", req.username).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")
    user = result.data

    # 질문 임베딩 (캐시 + RAG 공용)
    query_embedding: list[float] | None = None
    search_results = []
    context = ""

    try:
        query_embedding = embed(req.question)
    except Exception:
        pass

    # 캐시 확인
    cached = None
    if query_embedding:
        try:
            cached = check_cache(user["id"], query_embedding)
        except Exception:
            pass

    # 캐시 미스 → RAG
    if not cached and query_embedding:
        try:
            search_results = search_documents(user["id"], query_embedding)
            context = build_context(search_results)
        except Exception:
            import traceback
            with open("traceback.txt", "w") as f:
                f.write(traceback.format_exc())

    system_prompt = build_system_prompt(user, req.config, context, search_results)

    # answer_parts는 generate()가 완료된 후 background_tasks에서 참조됨
    answer_parts: list[str] = []

    def _save_sync() -> None:
        """StreamingResponse 완료 후 BackgroundTasks로 호출되는 동기 저장."""
        try:
            save_conversation(
                user_id=user["id"],
                session_id=req.sessionId,
                question=req.question,
                answer="".join(answer_parts),
                interviewer_config=req.config,
                question_embedding=query_embedding or [],
            )
        except Exception:
            pass

    async def generate():
        if cached:
            # 캐시 히트: 저장된 답변을 즉시 스트림
            yield f"data: {json.dumps({'type': 'cache_hit'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'citations', 'sources': []}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': cached.answer}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 1. citations 메타데이터를 SSE 이벤트로 먼저 전송
        citations = [
            {
                "index": i + 1,
                "title": r.title,
                "excerpt": r.content[:300],
                "url": r.source_url,
            }
            for i, r in enumerate(search_results)
            if r.similarity >= 0.25
        ]
        yield f"data: {json.dumps({'type': 'citations', 'sources': citations}, ensure_ascii=False)}\n\n"

        # 2. LLM 스트리밍 — 답변 누적
        async for chunk in stream_chat(system_prompt, req.question):
            answer_parts.append(chunk)
            yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    if query_embedding:
        background_tasks.add_task(_save_sync)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )
