import json
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from db.supabase import get_client
from services.embeddings import embed
from services.search import hybrid_search, build_context
from services.llm import build_system_prompt, stream_chat
from services.conversations import check_cache, save_conversation

logger = logging.getLogger(__name__)
router = APIRouter()


def _extract_relevant_excerpt(content: str, query: str, max_len: int = 200) -> str:
    """청크 내에서 query 키워드와 가장 관련 높은 문장을 추출한다."""
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?。])\s+|(?<=\n)', content) if s.strip()]
    if not sentences:
        return content[:max_len]

    query_words = set(re.sub(r'[^\w]', ' ', query.lower()).split())
    if not query_words:
        return sentences[0][:max_len]

    def score(sentence: str) -> int:
        words = set(re.sub(r'[^\w]', ' ', sentence.lower()).split())
        return len(query_words & words)

    best = max(sentences, key=score)
    # 가장 관련 높은 문장 + 앞뒤 문맥 1문장씩 포함
    idx = sentences.index(best)
    parts = sentences[max(0, idx - 1): idx + 2]
    excerpt = ' '.join(parts)
    return excerpt[:max_len] if len(excerpt) > max_len else excerpt

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

    query_embedding: list[float] | None = None
    search_results = []
    context = ""
    graph_context = ""

    try:
        query_embedding = embed(req.question)
    except Exception as e:
        logger.error("embed() failed for question=%r: %s", req.question[:50], e)
        raise HTTPException(status_code=503, detail="임베딩 서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.")

    cached = None
    try:
        cached = check_cache(user["id"], query_embedding)
    except Exception as e:
        logger.warning("cache check failed: %s", e)

    try:
        # 캐시 히트 여부 관계없이 citations 용도로 검색 실행
        search_results, graph_context = await hybrid_search(
            user_id=user["id"],
            query=req.question,
            query_embedding=query_embedding,
        )
        if not cached:
            context = build_context(search_results)
            if graph_context:
                context = f"{context}\n\n{graph_context}".strip() if context else graph_context
    except Exception as e:
        logger.error("RAG search failed for user=%s: %s", user["id"], e)

    system_prompt = build_system_prompt(user, req.config, context, search_results)

    answer_parts: list[str] = []

    def _save_sync() -> None:
        try:
            save_conversation(
                user_id=user["id"],
                session_id=req.sessionId,
                question=req.question,
                answer="".join(answer_parts),
                interviewer_config=req.config,
                question_embedding=query_embedding or [],
            )
        except Exception as e:
            logger.warning("save_conversation failed: %s", e)

    # citations 공통 생성 (캐시 히트 / 신규 공통)
    citations = [
        {
            "index": i + 1,
            "title": r.title,
            "excerpt": _extract_relevant_excerpt(r.content, req.question),
            "url": r.source_url,
            "source": r.source,
        }
        for i, r in enumerate(search_results)
        if r.similarity >= 0.25
    ]

    async def generate():
        if cached:
            yield f"data: {json.dumps({'type': 'cache_hit'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'citations', 'sources': citations}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': cached.answer}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        # Graphiti 폴백 사용 여부도 알림
        if graph_context:
            yield f"data: {json.dumps({'type': 'graph_fallback', 'used': True}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'citations', 'sources': citations}, ensure_ascii=False)}\n\n"

        async for chunk in stream_chat(system_prompt, req.question, req.history):
            answer_parts.append(chunk)
            yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    background_tasks.add_task(_save_sync)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )
