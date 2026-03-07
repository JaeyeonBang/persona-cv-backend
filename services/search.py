from dataclasses import dataclass
from db.supabase import get_client


@dataclass
class SearchResult:
    id: str
    title: str
    content: str
    similarity: float


def search_documents(user_id: str, query_embedding: list[float], limit: int = 3) -> list[SearchResult]:
    """pgvector 코사인 유사도 검색."""
    supabase = get_client()
    response = supabase.rpc(
        "match_documents",
        {
            "p_user_id": user_id,
            "p_query_embedding": query_embedding,
            "p_match_count": limit,
        },
    ).execute()

    return [
        SearchResult(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            similarity=row["similarity"],
        )
        for row in (response.data or [])
    ]


def build_context(results: list[SearchResult], threshold: float = 0.3) -> str:
    """검색 결과를 LLM 컨텍스트 문자열로 조합합니다."""
    relevant = [r for r in results if r.similarity >= threshold]
    if not relevant:
        return ""
    return "\n\n---\n\n".join(
        f"[{i + 1}] {r.title}\n{r.content[:1500]}" for i, r in enumerate(relevant)
    )
