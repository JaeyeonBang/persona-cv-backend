"""
Graphiti Knowledge Graph 클라이언트.

Neo4j + graphiti-core 를 사용해 이력 데이터를 Knowledge Graph로 구조화하고,
벡터 검색 신뢰도가 낮을 때 그래프 검색으로 폴백한다.

Neo4j 미연결 시 모든 메서드는 빈 결과를 반환하여 서비스가 정상 동작하도록 설계.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# graphiti-core 미설치 환경에서도 import 에러 없이 동작
try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    _GRAPHITI_AVAILABLE = True
except ImportError:
    _GRAPHITI_AVAILABLE = False
    logger.info("graphiti-core not installed — graph search disabled")


@dataclass
class GraphSearchResult:
    fact: str
    score: float
    source: str = ""


def _build_client() -> Any | None:
    """Graphiti 클라이언트를 초기화한다. 설정 미비 시 None 반환."""
    if not _GRAPHITI_AVAILABLE:
        return None

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("OPENROUTER_MODEL", "z-ai/glm-4.7-flash")
    embedding_model = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    if not neo4j_password or not openrouter_key:
        logger.info("Graphiti skipped: NEO4J_PASSWORD or OPENROUTER_API_KEY not set")
        return None

    try:
        llm_client = OpenAIClient(
            config=LLMConfig(
                api_key=openrouter_key,
                model=model,
                base_url=openrouter_base,
            )
        )
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=openrouter_key,
                model=embedding_model,
                base_url=openrouter_base,
            )
        )
        return Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client=llm_client, embedder=embedder)
    except Exception as e:
        logger.warning("Graphiti init failed: %s", e)
        return None


# 모듈 로드 시 한 번만 초기화
_graphiti: Any | None = _build_client()


def is_available() -> bool:
    return _graphiti is not None


async def add_episode(group_id: str, name: str, content: str, source: str = "document") -> bool:
    """문서 청크를 Graphiti 에피소드로 추가한다.

    Args:
        group_id: 사용자 ID (네임스페이스)
        name: 에피소드 제목 (문서 제목)
        content: 에피소드 내용 (청크 텍스트)
        source: 소스 타입
    Returns:
        성공 여부
    """
    if not is_available():
        return False

    try:
        from datetime import datetime
        await _graphiti.add_episode(  # type: ignore
            name=name,
            episode_body=content,
            source=EpisodeType.text,
            source_description=source,
            reference_time=datetime.utcnow(),
            group_id=group_id,
        )
        return True
    except Exception as e:
        logger.warning("Graphiti add_episode failed: %s", e)
        return False


async def search(group_id: str, query: str, limit: int = 5) -> list[GraphSearchResult]:
    """그래프 검색 수행.

    Args:
        group_id: 사용자 ID (네임스페이스)
        query: 검색 쿼리
        limit: 최대 결과 수
    Returns:
        GraphSearchResult 리스트
    """
    if not is_available():
        return []

    try:
        results = await _graphiti.search(  # type: ignore
            query=query,
            group_ids=[group_id],
            num_results=limit,
        )
        return [
            GraphSearchResult(
                fact=edge.fact,
                score=getattr(edge, "score", 0.5),
                source=getattr(edge, "source_node_uuid", ""),
            )
            for edge in (results or [])
        ]
    except Exception as e:
        logger.warning("Graphiti search failed: %s", e)
        return []


def graph_results_to_context(results: list[GraphSearchResult]) -> str:
    """그래프 검색 결과를 LLM 컨텍스트 문자열로 변환."""
    if not results:
        return ""
    lines = "\n".join(f"- {r.fact}" for r in results)
    return f"[Knowledge Graph 관계 정보]\n{lines}"
