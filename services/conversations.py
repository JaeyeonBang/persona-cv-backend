import uuid
from dataclasses import dataclass
from db.supabase import get_client


@dataclass
class CachedAnswer:
    id: str
    question: str
    answer: str
    similarity: float


def check_cache(user_id: str, query_embedding: list[float], threshold: float = 0.95) -> CachedAnswer | None:
    """유사도 임계값 이상의 캐시된 답변을 반환합니다. 없으면 None."""
    supabase = get_client()
    response = supabase.rpc(
        "match_conversation_cache",
        {
            "p_user_id": user_id,
            "p_query_embedding": query_embedding,
            "p_threshold": threshold,
            "p_limit": 1,
        },
    ).execute()

    rows = response.data or []
    if not rows:
        return None

    row = rows[0]
    return CachedAnswer(
        id=row["id"],
        question=row["question"],
        answer=row["answer"],
        similarity=row["similarity"],
    )


def save_conversation(
    user_id: str,
    session_id: str | None,
    question: str,
    answer: str,
    interviewer_config: dict,
    question_embedding: list[float],
    conversation_id: str | None = None,
) -> str:
    """Q&A 쌍과 질문 임베딩을 conversations 테이블에 저장합니다. conversation_id를 반환합니다."""
    supabase = get_client()
    cid = conversation_id or str(uuid.uuid4())
    supabase.table("conversations").insert(
        {
            "id": cid,
            "user_id": user_id,
            "session_id": session_id or "",
            "question": question,
            "answer": answer,
            "interviewer_config": interviewer_config,
            "question_embedding": question_embedding,
            "is_cached": False,
        }
    ).execute()
    return cid
