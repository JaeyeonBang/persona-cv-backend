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

    def test_returns_conversation_id_string(self):
        """save_conversation이 conversation_id 문자열을 반환한다."""
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            result = save_conversation("user-1", None, "q", "a", {}, [])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_uses_provided_conversation_id(self):
        """conversation_id 파라미터를 제공하면 그 값을 insert payload에 사용한다."""
        client = self._mock_supabase()
        fixed_id = "pre-generated-uuid-abc"
        with patch("services.conversations.get_client", return_value=client):
            result = save_conversation("user-1", None, "q", "a", {}, [], conversation_id=fixed_id)
        assert result == fixed_id
        insert_payload = client.table.return_value.insert.call_args[0][0]
        assert insert_payload["id"] == fixed_id

    def test_generates_uuid_when_conversation_id_is_none(self):
        """conversation_id가 None이면 새 UUID를 생성해 반환한다."""
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            result = save_conversation("user-1", None, "q", "a", {}, [], conversation_id=None)
        import uuid
        # 유효한 UUID v4 형식인지 검증
        parsed = uuid.UUID(result)
        assert str(parsed) == result

    def test_different_calls_generate_different_ids(self):
        """conversation_id 없이 두 번 호출하면 서로 다른 ID를 반환한다."""
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            id1 = save_conversation("user-1", None, "q1", "a1", {}, [])
            id2 = save_conversation("user-1", None, "q2", "a2", {}, [])
        assert id1 != id2

    def test_insert_payload_contains_id_field(self):
        """insert payload에 'id' 필드가 반드시 포함된다."""
        client = self._mock_supabase()
        with patch("services.conversations.get_client", return_value=client):
            save_conversation("user-1", None, "q", "a", {}, [])
        insert_payload = client.table.return_value.insert.call_args[0][0]
        assert "id" in insert_payload
