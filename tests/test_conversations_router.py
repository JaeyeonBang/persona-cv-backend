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
