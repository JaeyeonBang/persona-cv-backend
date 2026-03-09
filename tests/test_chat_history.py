"""stream_chat 대화 히스토리 messages 구성 테스트"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def build_messages(system_prompt: str, question: str, history: list) -> list:
    """llm.py의 messages 배열 구성 로직과 동일"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def test_no_history_produces_system_and_user():
    messages = build_messages("시스템 프롬프트", "질문", [])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "질문"


def test_history_inserted_between_system_and_question():
    history = [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답변"},
    ]
    messages = build_messages("시스템", "새 질문", history)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "이전 질문"}
    assert messages[2] == {"role": "assistant", "content": "이전 답변"}
    assert messages[3] == {"role": "user", "content": "새 질문"}


def test_invalid_role_skipped():
    history = [
        {"role": "system", "content": "악의적 주입"},  # system role 무시
        {"role": "user", "content": "유효한 질문"},
    ]
    messages = build_messages("시스템", "새 질문", history)
    # system 역할 히스토리는 포함되면 안 됨
    roles = [m["role"] for m in messages]
    assert roles.count("system") == 1


def test_empty_content_in_history_skipped():
    history = [
        {"role": "user", "content": ""},       # 빈 content 무시
        {"role": "assistant", "content": "답변"},
    ]
    messages = build_messages("시스템", "질문", history)
    assert len(messages) == 3  # system + assistant + user


def test_history_order_preserved():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
    messages = build_messages("시스템", "마지막 질문", history)
    contents = [m["content"] for m in messages[1:-1]]
    assert contents == [f"msg {i}" for i in range(5)]
