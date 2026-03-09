"""
하이브리드 검색 서비스.

1단계: pgvector 코사인 유사도 검색
2단계: 최고 유사도 < GRAPHITI_FALLBACK_THRESHOLD 이면 Graphiti 그래프 검색 폴백
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from db.supabase import get_client

logger = logging.getLogger(__name__)

# Vector 신뢰도 임계값 — 이 값 미만이면 Graphiti 폴백 실행
GRAPHITI_FALLBACK_THRESHOLD = 0.7


@dataclass
class SearchResult:
    id: str
    title: str
    content: str
    similarity: float
    source_url: str | None = field(default=None)
    source: str = field(default="vector")  # "vector" | "graph"


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

    results = [
        SearchResult(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            similarity=row["similarity"],
            source="vector",
        )
        for row in (response.data or [])
    ]

    if results:
        chunk_ids = [r.id for r in results]
        # chunk_id → document_id 조회
        chunks_resp = supabase.table("document_chunks").select("id, document_id").in_("id", chunk_ids).execute()
        chunk_to_doc = {row["id"]: row["document_id"] for row in (chunks_resp.data or [])}

        # document_id → source_url 조회
        doc_ids = list(set(chunk_to_doc.values()))
        if doc_ids:
            url_resp = supabase.table("documents").select("id, source_url").in_("id", doc_ids).execute()
            doc_url_map = {row["id"]: row["source_url"] for row in (url_resp.data or [])}
        else:
            doc_url_map = {}

        for r in results:
            doc_id = chunk_to_doc.get(r.id)
            r.source_url = doc_url_map.get(doc_id) if doc_id else None

    return results


async def hybrid_search(
    user_id: str,
    query: str,
    query_embedding: list[float],
    limit: int = 3,
) -> tuple[list[SearchResult], str]:
    """하이브리드 검색.

    Returns:
        (search_results, graph_context)
        - search_results: vector 검색 결과 (citations용)
        - graph_context: Graphiti 결과를 포맷한 컨텍스트 문자열 (빈 문자열이면 폴백 없음)
    """
    from services.graphiti_client import search as graphiti_search, graph_results_to_context

    vector_results = search_documents(user_id, query_embedding, limit)

    max_similarity = max((r.similarity for r in vector_results), default=0.0)
    graph_context = ""

    if max_similarity < GRAPHITI_FALLBACK_THRESHOLD:
        logger.info(
            "Vector max similarity %.3f < %.1f — running Graphiti fallback",
            max_similarity,
            GRAPHITI_FALLBACK_THRESHOLD,
        )
        try:
            graph_results = await graphiti_search(group_id=user_id, query=query, limit=5)
            graph_context = graph_results_to_context(graph_results)
            if graph_context:
                logger.info("Graphiti returned %d results", len(graph_results))
        except Exception as e:
            logger.warning("Graphiti fallback failed: %s", e)

    return vector_results, graph_context


def build_context(results: list[SearchResult], threshold: float = 0.25) -> str:
    """Vector 검색 결과를 LLM 컨텍스트 문자열로 조합한다."""
    relevant = [r for r in results if r.similarity >= threshold]
    if not relevant:
        return ""
    return "\n\n---\n\n".join(
        f"[{i + 1}] {r.title}\n{r.content[:1500]}" for i, r in enumerate(relevant)
    )
