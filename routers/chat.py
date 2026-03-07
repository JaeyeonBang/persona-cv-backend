from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from db.supabase import get_client
from services.embeddings import embed
from services.search import search_documents, build_context
from services.llm import build_system_prompt, stream_chat

router = APIRouter()


class ChatRequest(BaseModel):
    userId: str
    question: str
    config: dict
    sessionId: str | None = None


@router.post("/api/chat")
async def chat(req: ChatRequest):
    supabase = get_client()

    # 유저 조회
    result = supabase.table("users").select("*").eq("id", req.userId).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")
    user = result.data

    # RAG: 질문 임베딩 → 벡터 검색 → 컨텍스트 구성
    try:
        query_embedding = embed(req.question)
        search_results = search_documents(req.userId, query_embedding)
        context = build_context(search_results)
    except Exception:
        context = ""

    system_prompt = build_system_prompt(user, req.config, context)

    async def generate():
        async for chunk in stream_chat(system_prompt, req.question):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff"},
    )
