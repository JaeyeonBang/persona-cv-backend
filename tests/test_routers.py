import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_documents_empty_ids_returns_400():
    response = client.post("/api/documents/process", json={"documentIds": []})
    assert response.status_code == 400


def test_process_documents_queues_tasks():
    with patch("routers.documents.process_document") as mock_proc:
        response = client.post(
            "/api/documents/process",
            json={"documentIds": ["doc-1", "doc-2"]},
        )
    assert response.status_code == 200
    assert response.json()["queued"] == 2


def test_chat_user_not_found_returns_404():
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None

    with patch("routers.chat.get_client", return_value=mock_supabase):
        response = client.post(
            "/api/chat",
            json={
                "username": "nonexistent",
                "question": "안녕",
                "history": [],
                "config": {"language": "ko", "speechStyle": "formal", "answerLength": "medium", "questionStyle": "free"},
            },
        )
    assert response.status_code == 404
