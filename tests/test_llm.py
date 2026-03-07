import pytest
from services.llm import build_system_prompt


def make_user(overrides=None):
    user = {
        "id": "u1",
        "name": "홍길동",
        "title": "시니어 개발자",
        "bio": "10년차 풀스택",
        "persona_config": {
            "preset": "professional",
            "custom_prompt": None,
        },
    }
    if overrides:
        user.update(overrides)
    return user


def make_config(overrides=None):
    config = {
        "language": "ko",
        "speechStyle": "formal",
        "answerLength": "medium",
        "questionStyle": "free",
    }
    if overrides:
        config.update(overrides)
    return config


def test_includes_user_name_title_bio():
    prompt = build_system_prompt(make_user(), make_config(), "")
    assert "홍길동" in prompt
    assert "시니어 개발자" in prompt
    assert "10년차 풀스택" in prompt


def test_context_included_when_provided():
    prompt = build_system_prompt(make_user(), make_config(), "관련 문서 내용")
    assert "참고 자료 (관련 문서에서 발췌)" in prompt
    assert "관련 문서 내용" in prompt


def test_fallback_message_when_no_context():
    prompt = build_system_prompt(make_user(), make_config(), "")
    assert "등록된 문서가 없습니다" in prompt


def test_korean_language_reflected():
    prompt = build_system_prompt(make_user(), make_config({"language": "ko"}), "")
    assert "한국어" in prompt


def test_english_language_reflected():
    prompt = build_system_prompt(make_user(), make_config({"language": "en"}), "")
    assert "English" in prompt


def test_formal_speech_reflected():
    prompt = build_system_prompt(make_user(), make_config({"speechStyle": "formal"}), "")
    assert "합니다/입니다" in prompt


def test_casual_speech_reflected():
    prompt = build_system_prompt(make_user(), make_config({"speechStyle": "casual"}), "")
    assert "반말" in prompt


def test_custom_prompt_included_when_set():
    user = make_user()
    user["persona_config"]["custom_prompt"] = "항상 유머를 섞어서 답변"
    prompt = build_system_prompt(user, make_config(), "")
    assert "항상 유머를 섞어서 답변" in prompt


def test_custom_prompt_omitted_when_none():
    prompt = build_system_prompt(make_user(), make_config(), "")
    assert "추가 지시" not in prompt


def test_first_person_instruction_included():
    prompt = build_system_prompt(make_user(), make_config(), "")
    assert "1인칭으로 답변" in prompt


def test_no_markdown_instruction_included():
    prompt = build_system_prompt(make_user(), make_config(), "")
    assert "마크다운 문법은 사용하지 마세요" in prompt
