"""
conversations.py 단위 테스트

테스트 대상:
  - check_cache: 캐시 조회 (RPC 호출 및 임계값 적용)
  - save_conversation: Q&A 저장 (insert 호출 검증)
"""

from unittest.mock import MagicMock, patch
from services.conversations import check_cache, save_conversation, CachedAnswer


# ── check_cache ───────────────────────────────────────────────────────────────

class TestCheckCache:
    def _mock_supabase(self, rpc_data: list) -> MagicMock:
        client = MagicMock()
        client.rpc.return_value.execute.return_value.data = rpc_data
        return client

    def test_returns_none_when_no_match(self):
        with patch("services.conversations.get_client", return_value=self._mock_supabase([])):
            result = check_cache("user-1", [0.1] * 1536)
        assert result is None

    def test_returns_cached_answer_on_hit(self):
        row = {"id": "c1", "question": "자기소개", "answer": "저는 ...", "similarity": 0.97}
        with patch("services.conversations.get_client", return_value=self._mock_supabase([row])):
            result = check_cache("user-1", [0.1] * 1536)
        assert isinstance(result, CachedAnswer)
        assert result.id == "c1"
        assert result.question == "자기소개"
        assert result.answer == "저는 ..."
        assert result.similarity == 0.97

    def test_passes_threshold_to_rpc(self):
        client = self._mock_supabase([])
        with patch("services.conversations.get_client", return_value=client):
            check_cache("user-1", [0.1] * 1536, threshold=0.9)
        call_kwargs = client.rpc.call_args[0][1]
        assert call_kwargs["p_threshold"] == 0.9

    def test_default_threshold_is_095(self):
        client = self._mock_supabase([])
        with patch("services.conversations.get_client", return_value=client):
            check_cache("user-1", [0.1] * 1536)
        call_kwargs = client.rpc.call_args[0][1]
        assert call_kwargs["p_threshold"] == 0.95

    def test_passes_user_id_to_rpc(self):
        client = self._mock_supabase([])
        with patch("services.conversations.get_client", return_value=client):
            check_cache("user-42", [0.1] * 1536)
        call_kwargs = client.rpc.call_args[0][1]
        assert call_kwargs["p_user_id"] == "user-42"

    def test_passes_embedding_to_rpc(self):
        client = self._mock_supabase([])
        embedding = [0.5] * 1536
        with patch("services.conversations.get_client", return_value=client):
            check_cache("user-1", embedding)
        call_kwargs = client.rpc.call_args[0][1]
        assert call_kwargs["p_query_embedding"] == embedding

    def test_limit_is_1(self):
        client = self._mock_supabase([])
        with patch("services.conversations.get_client", return_value=client):
            check_cache("user-1", [0.1] * 1536)
        call_kwargs = client.rpc.call_args[0][1]
        assert call_kwargs["p_limit"] == 1

    def test_returns_first_row_only(self):
        rows = [
            {"id": "c1", "question": "q1", "answer": "a1", "similarity": 0.99},
            {"id": "c2", "question": "q2", "answer": "a2", "similarity": 0.96},
        ]
        with patch("services.conversations.get_client", return_value=self._mock_supabase(rows)):
            result = check_cache("user-1", [0.1] * 1536)
        assert result.id == "c1"


# ── save_conversation ─────────────────────────────────────────────────────────

class TestSaveConversation:
    def _mock_supabase(self) -> MagicMock:
        client = MagicMock()
        client.table.return_value.insert.return_value.execute.return_value = MagicMock()
        return client

    def test_calls_insert_with_correct_fields(self):
        client = self._mock_supabase()
        embedding = [0.1] * 1536
        with patch("services.conversations.get_client", return_value=client):
            save_conversation(
                user_id="user-1",
                session_id="sess-abc",
                question="자기소개 해줘",
                answer="저는 김민준입니다.",
                interviewer_config={"answerLength": "medium"},
                question_embedding=embedding,
            )
        client.table.assert_called_once_with("conversations")
        insert_payload = client.table.return_value.insert.call_args[0][0]
        assert insert_payload["user_id"] == "user-1"
        assert insert_payload["session_id"] == "sess-abc"
        assert insert_payload["question"] == "자기소개 해줘"
        assert insert_payload["answer"] == "저는 김민준입니다."
        assert insert_payload["interviewer_config"] == {"answerLength": "medium"}
        assert insert_payload["question_embedding"] == embedding

    def test_none_session_id_becomes_empty_string(self):
        """session_id NOT NULL 컬럼이므로 None은 빈 문자열로 저장."""
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            save_conversation(
                user_id="user-1",
                session_id=None,
                question="q",
                answer="a",
                interviewer_config={},
                question_embedding=[],
            )
        insert_payload = client.table.return_value.insert.call_args[0][0]
        assert insert_payload["session_id"] == ""

    def test_empty_embedding_is_stored(self):
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            save_conversation("user-1", None, "q", "a", {}, [])
        insert_payload = client.table.return_value.insert.call_args[0][0]
        assert insert_payload["question_embedding"] == []

    def test_is_cached_is_false(self):
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            save_conversation("user-1", None, "q", "a", {}, [])
        insert_payload = client.table.return_value.insert.call_args[0][0]
        assert insert_payload["is_cached"] is False

    def test_calls_execute(self):
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            save_conversation("user-1", None, "q", "a", {}, [])
        client.table.return_value.insert.return_value.execute.assert_called_once()
