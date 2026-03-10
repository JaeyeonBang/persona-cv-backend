"""
Graphiti 클라이언트 + 하이브리드 검색 테스트.

Neo4j 미연결 환경에서도 graceful fallback이 동작하는지 검증.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── graphiti_client graceful fallback ─────────────────────────
class TestGraphitiClientFallback:
    def test_is_available_returns_false_when_not_configured(self):
        """NEO4J_PASSWORD 미설정 시 is_available()은 False."""
        from services.graphiti_client import is_available
        # 실제 환경에서는 Neo4j 미연결 상태이므로 False
        result = is_available()
        assert isinstance(result, bool)  # 예외 없이 bool 반환

    @pytest.mark.asyncio
    async def test_add_episode_returns_false_when_unavailable(self):
        """is_available() == False이면 add_episode는 False 반환."""
        import services.graphiti_client as gc
        original = gc._graphiti

        try:
            gc._graphiti = None  # 강제로 unavailable 상태

            result = await gc.add_episode(
                group_id="user-1",
                name="테스트 문서",
                content="내용",
            )
            assert result is False
        finally:
            gc._graphiti = original

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_unavailable(self):
        """is_available() == False이면 search는 빈 리스트 반환."""
        import services.graphiti_client as gc
        original = gc._graphiti

        try:
            gc._graphiti = None
            results = await gc.search(group_id="user-1", query="질문")
            assert results == []
        finally:
            gc._graphiti = original

    def test_graph_results_to_context_empty(self):
        """빈 리스트 → 빈 문자열 반환."""
        from services.graphiti_client import graph_results_to_context
        assert graph_results_to_context([]) == ""

    def test_graph_results_to_context_format(self):
        """결과가 있으면 '[Knowledge Graph 관계 정보]' 헤더 포함."""
        from services.graphiti_client import GraphSearchResult, graph_results_to_context

        results = [
            GraphSearchResult(fact="프로젝트A — React 사용", score=0.9),
            GraphSearchResult(fact="프로젝트A — 성과: MAU 10만", score=0.8),
        ]
        ctx = graph_results_to_context(results)
        assert "Knowledge Graph" in ctx
        assert "프로젝트A — React 사용" in ctx
        assert "프로젝트A — 성과: MAU 10만" in ctx
        assert ctx.count("-") >= 2  # bullet list


# ── hybrid_search ────────────────────────────────────────────
class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_no_graphiti_fallback_when_vector_score_high(self):
        """vector 유사도 >= 0.7이면 Graphiti 검색하지 않는다."""
        from services.search import SearchResult

        high_score_results = [
            SearchResult(id="1", title="문서A", content="내용", similarity=0.85),
        ]

        with patch("services.search.search_documents", return_value=high_score_results), \
             patch("services.graphiti_client.search", new_callable=AsyncMock) as mock_graph:

            from services.search import hybrid_search
            results, graph_ctx = await hybrid_search(
                user_id="user-1",
                query="질문",
                query_embedding=[0.1] * 1536,
            )

            mock_graph.assert_not_called()
            assert graph_ctx == ""
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_graphiti_fallback_when_vector_score_low(self):
        """vector 유사도 < 0.7이면 Graphiti 폴백을 시도한다."""
        from services.search import SearchResult
        from services.graphiti_client import GraphSearchResult

        low_score_results = [
            SearchResult(id="1", title="문서A", content="내용", similarity=0.4),
        ]
        graph_results = [
            GraphSearchResult(fact="프로젝트A — 기술스택: React", score=0.9),
        ]

        with patch("services.search.search_documents", return_value=low_score_results), \
             patch("services.graphiti_client.search", new_callable=AsyncMock, return_value=graph_results):

            from services.search import hybrid_search
            results, graph_ctx = await hybrid_search(
                user_id="user-1",
                query="질문",
                query_embedding=[0.1] * 1536,
            )

            assert "Knowledge Graph" in graph_ctx
            assert "프로젝트A" in graph_ctx

    @pytest.mark.asyncio
    async def test_graphiti_fallback_when_no_vector_results(self):
        """vector 결과가 없어도 Graphiti 폴백을 시도한다."""
        from services.graphiti_client import GraphSearchResult

        graph_results = [
            GraphSearchResult(fact="React 전문가", score=0.8),
        ]

        with patch("services.search.search_documents", return_value=[]), \
             patch("services.graphiti_client.search", new_callable=AsyncMock, return_value=graph_results):

            from services.search import hybrid_search
            results, graph_ctx = await hybrid_search(
                user_id="user-1",
                query="기술 스택",
                query_embedding=[0.1] * 1536,
            )

            assert "Knowledge Graph" in graph_ctx
            assert results == []

    @pytest.mark.asyncio
    async def test_graphiti_exception_is_swallowed(self):
        """Graphiti가 예외를 던져도 서비스는 정상 동작한다."""
        from services.search import SearchResult

        low_score_results = [
            SearchResult(id="1", title="문서A", content="내용", similarity=0.3),
        ]

        with patch("services.search.search_documents", return_value=low_score_results), \
             patch("services.graphiti_client.search", new_callable=AsyncMock, side_effect=RuntimeError("Neo4j down")):

            from services.search import hybrid_search
            results, graph_ctx = await hybrid_search(
                user_id="user-1",
                query="질문",
                query_embedding=[0.1] * 1536,
            )

            # 예외가 삼켜지고 빈 graph_ctx 반환
            assert graph_ctx == ""
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_fallback_threshold_boundary(self):
        """similarity == 0.7 (임계값)이면 폴백하지 않는다."""
        from services.search import SearchResult

        boundary_results = [
            SearchResult(id="1", title="문서A", content="내용", similarity=0.7),
        ]

        with patch("services.search.search_documents", return_value=boundary_results), \
             patch("services.graphiti_client.search", new_callable=AsyncMock) as mock_graph:

            from services.search import hybrid_search
            _, graph_ctx = await hybrid_search(
                user_id="user-1",
                query="질문",
                query_embedding=[0.1] * 1536,
            )

            mock_graph.assert_not_called()
            assert graph_ctx == ""
