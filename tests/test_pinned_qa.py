"""
pinned_qa 라우터 테스트.

실제 Supabase 호출을 mock으로 대체합니다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


def _mock_user():
    return {
        "id": "user-uuid-1",
        "username": "testuser",
        "name": "테스트 유저",
        "title": "개발자",
        "bio": "안녕하세요",
        "persona_config": {},
    }


def _mock_item():
    return {
        "id": "item-uuid-1",
        "user_id": "user-uuid-1",
        "question": "기술 스택이 뭔가요?",
        "answer": "저는 Python과 TypeScript를 주로 사용합니다.",
        "display_order": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


class TestListPinnedQA:
    def test_returns_empty_list(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = _mock_user()
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = []

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.get("/api/pinned-qa?username=testuser")

        assert res.status_code == 200
        assert res.json() == []

    def test_returns_items(self, client):
        mock_supabase = MagicMock()
        # user lookup
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = _mock_user()
        # pinned_qa lookup
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [_mock_item()]

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.get("/api/pinned-qa?username=testuser")

        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["question"] == "기술 스택이 뭔가요?"

    def test_user_not_found_returns_404(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.get("/api/pinned-qa?username=unknown")

        assert res.status_code == 404


class TestCreatePinnedQA:
    def test_create_success(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = _mock_user()
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [_mock_item()]

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.post("/api/pinned-qa", json={
                "username": "testuser",
                "question": "기술 스택이 뭔가요?",
                "answer": "Python과 TypeScript입니다.",
                "display_order": 0,
            })

        assert res.status_code == 201

    def test_user_not_found_returns_404(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.post("/api/pinned-qa", json={
                "username": "unknown",
                "question": "Q",
                "answer": "A",
            })

        assert res.status_code == 404


class TestUpdatePinnedQA:
    def test_update_success(self, client):
        updated = {**_mock_item(), "answer": "수정된 답변"}
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [updated]

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.put("/api/pinned-qa/item-uuid-1", json={"answer": "수정된 답변"})

        assert res.status_code == 200
        assert res.json()["answer"] == "수정된 답변"

    def test_no_fields_returns_400(self, client):
        with patch("routers.pinned_qa.get_client", return_value=MagicMock()):
            res = client.put("/api/pinned-qa/item-uuid-1", json={})

        assert res.status_code == 400

    def test_not_found_returns_404(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value.data = []

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.put("/api/pinned-qa/nonexistent", json={"answer": "A"})

        assert res.status_code == 404


class TestDeletePinnedQA:
    def test_delete_success(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = None

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.delete("/api/pinned-qa/item-uuid-1")

        assert res.status_code == 204


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = _mock_user()

        mock_response = MagicMock()
        mock_response.content = "저는 Python과 TypeScript를 주로 사용합니다."

        with (
            patch("routers.pinned_qa.get_client", return_value=mock_supabase),
            patch("routers.pinned_qa._get_llm") as mock_llm_factory,
        ):
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_factory.return_value = mock_llm

            res = client.post("/api/pinned-qa/generate", json={
                "username": "testuser",
                "question": "기술 스택이 뭔가요?",
            })

        assert res.status_code == 200
        assert "answer" in res.json()

    def test_user_not_found_returns_404(self, client):
        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None

        with patch("routers.pinned_qa.get_client", return_value=mock_supabase):
            res = client.post("/api/pinned-qa/generate", json={
                "username": "unknown",
                "question": "Q?",
            })

        assert res.status_code == 404
