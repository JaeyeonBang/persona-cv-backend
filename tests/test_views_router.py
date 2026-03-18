"""
views 라우터 단위 테스트
POST /api/views/{username}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _mock_supabase_rpc():
    mock = MagicMock()
    mock.rpc.return_value.execute.return_value = MagicMock()
    return mock


class TestViewsEndpoint:
    def test_post_returns_204(self):
        mock = _mock_supabase_rpc()
        with patch("routers.views.get_client", return_value=mock):
            res = client.post("/api/views/demo-user")
        assert res.status_code == 204

    def test_calls_rpc_with_correct_username(self):
        mock = _mock_supabase_rpc()
        with patch("routers.views.get_client", return_value=mock):
            client.post("/api/views/test-username")
        mock.rpc.assert_called_once_with(
            "increment_view_count", {"p_username": "test-username"}
        )

    def test_different_usernames_call_rpc_with_different_args(self):
        mock = _mock_supabase_rpc()
        with patch("routers.views.get_client", return_value=mock):
            client.post("/api/views/alice")
        call_kwargs = mock.rpc.call_args[0][1]
        assert call_kwargs["p_username"] == "alice"

    def test_rpc_exception_returns_500(self):
        mock = MagicMock()
        mock.rpc.return_value.execute.side_effect = Exception("DB error")
        with patch("routers.views.get_client", return_value=mock):
            res = client.post("/api/views/demo-user")
        assert res.status_code == 500
