"""
conversations 라우터 단위 테스트

테스트 대상:
  POST /api/conversations/{id}/feedback
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _mock_supabase_update():
    mock = MagicMock()
    mock.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    return mock


class TestFeedbackEndpoint:
    """POST /api/conversations/{id}/feedback"""

    def test_positive_feedback_returns_204(self):
        mock = _mock_supabase_update()
        with patch("routers.conversations.get_client", return_value=mock):
            res = client.post(
                "/api/conversations/conv-abc-123/feedback",
                json={"feedback": 1},
            )
        assert res.status_code == 204

    def test_negative_feedback_returns_204(self):
        mock = _mock_supabase_update()
        with patch("routers.conversations.get_client", return_value=mock):
            res = client.post(
                "/api/conversations/conv-abc-123/feedback",
                json={"feedback": -1},
            )
        assert res.status_code == 204

    def test_zero_feedback_returns_400(self):
        res = client.post(
            "/api/conversations/conv-abc-123/feedback",
            json={"feedback": 0},
        )
        assert res.status_code == 400

    def test_invalid_feedback_2_returns_400(self):
        res = client.post(
            "/api/conversations/conv-abc-123/feedback",
            json={"feedback": 2},
        )
        assert res.status_code == 400

    def test_missing_feedback_field_returns_422(self):
        res = client.post(
            "/api/conversations/conv-abc-123/feedback",
            json={},
        )
        assert res.status_code == 422

    def test_calls_update_with_feedback_value(self):
        mock = _mock_supabase_update()
        with patch("routers.conversations.get_client", return_value=mock):
            client.post(
                "/api/conversations/conv-abc-123/feedback",
                json={"feedback": 1},
            )
        mock.table.assert_called_once_with("conversations")
        mock.table.return_value.update.assert_called_once_with({"feedback": 1})

    def test_calls_eq_with_conversation_id(self):
        mock = _mock_supabase_update()
        with patch("routers.conversations.get_client", return_value=mock):
            client.post(
                "/api/conversations/my-conv-id-999/feedback",
                json={"feedback": -1},
            )
        mock.table.return_value.update.return_value.eq.assert_called_once_with(
            "id", "my-conv-id-999"
        )

    def test_error_detail_on_invalid_feedback(self):
        res = client.post(
            "/api/conversations/conv-123/feedback",
            json={"feedback": 99},
        )
        assert res.status_code == 400
        assert "feedback" in res.json()["detail"].lower()


def _mock_supabase_delete():
    mock = MagicMock()
    mock.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock()
    return mock


class TestDeleteEndpoint:
    """DELETE /api/conversations/{id}"""

    def test_returns_204(self):
        mock = _mock_supabase_delete()
        with patch("routers.conversations.get_client", return_value=mock):
            res = client.delete("/api/conversations/conv-xyz-789")
        assert res.status_code == 204

    def test_calls_delete_on_conversations_table(self):
        mock = _mock_supabase_delete()
        with patch("routers.conversations.get_client", return_value=mock):
            client.delete("/api/conversations/conv-xyz-789")
        mock.table.assert_called_once_with("conversations")
        mock.table.return_value.delete.assert_called_once()

    def test_calls_eq_with_conversation_id(self):
        mock = _mock_supabase_delete()
        with patch("routers.conversations.get_client", return_value=mock):
            client.delete("/api/conversations/target-conv-id")
        mock.table.return_value.delete.return_value.eq.assert_called_once_with(
            "id", "target-conv-id"
        )

    def test_response_body_is_empty(self):
        mock = _mock_supabase_delete()
        with patch("routers.conversations.get_client", return_value=mock):
            res = client.delete("/api/conversations/conv-abc")
        assert res.content == b""


def _mock_supabase_clear():
    mock = MagicMock()
    mock.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock()
    return mock


class TestClearEndpoint:
    """POST /api/conversations/clear"""

    def test_returns_204(self):
        mock = _mock_supabase_clear()
        with patch("routers.conversations.get_client", return_value=mock):
            res = client.post("/api/conversations/clear", json={"user_id": "user-123"})
        assert res.status_code == 204

    def test_missing_user_id_returns_400(self):
        res = client.post("/api/conversations/clear", json={"user_id": ""})
        assert res.status_code == 400

    def test_missing_user_id_field_returns_422(self):
        res = client.post("/api/conversations/clear", json={})
        assert res.status_code == 422

    def test_calls_delete_with_user_id(self):
        mock = _mock_supabase_clear()
        with patch("routers.conversations.get_client", return_value=mock):
            client.post("/api/conversations/clear", json={"user_id": "user-abc"})
        mock.table.return_value.delete.return_value.eq.assert_called_once_with(
            "user_id", "user-abc"
        )

    def test_detail_mentions_user_id_on_empty(self):
        res = client.post("/api/conversations/clear", json={"user_id": ""})
        assert res.status_code == 400
        assert "user_id" in res.json()["detail"].lower()


def _mock_supabase_patch():
    mock = MagicMock()
    mock.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    return mock


class TestPatchAnswerEndpoint:
    """PATCH /api/conversations/{id}"""

    def test_returns_200(self):
        mock = _mock_supabase_patch()
        with patch("routers.conversations.get_client", return_value=mock):
            res = client.patch(
                "/api/conversations/conv-patch-id",
                json={"answer": "새 답변입니다"},
            )
        assert res.status_code == 200

    def test_response_contains_id_and_answer(self):
        mock = _mock_supabase_patch()
        with patch("routers.conversations.get_client", return_value=mock):
            res = client.patch(
                "/api/conversations/conv-patch-id",
                json={"answer": "수정된 답변"},
            )
        body = res.json()
        assert body["id"] == "conv-patch-id"
        assert body["answer"] == "수정된 답변"

    def test_empty_answer_returns_400(self):
        res = client.patch(
            "/api/conversations/conv-patch-id",
            json={"answer": "   "},
        )
        assert res.status_code == 400

    def test_missing_answer_field_returns_422(self):
        res = client.patch("/api/conversations/conv-patch-id", json={})
        assert res.status_code == 422

    def test_calls_update_with_answer(self):
        mock = _mock_supabase_patch()
        with patch("routers.conversations.get_client", return_value=mock):
            client.patch(
                "/api/conversations/my-conv",
                json={"answer": "업데이트된 답변"},
            )
        mock.table.return_value.update.assert_called_once_with(
            {"answer": "업데이트된 답변"}
        )

    def test_calls_eq_with_conversation_id(self):
        mock = _mock_supabase_patch()
        with patch("routers.conversations.get_client", return_value=mock):
            client.patch(
                "/api/conversations/target-id-999",
                json={"answer": "답변"},
            )
        mock.table.return_value.update.return_value.eq.assert_called_once_with(
            "id", "target-id-999"
        )
