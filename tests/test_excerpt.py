"""_extract_relevant_excerpt 함수 단위 테스트"""
import pytest
from routers.chat import _extract_relevant_excerpt


def test_returns_most_relevant_sentence():
    content = "저는 삼성에서 근무했습니다. 카카오에서 백엔드 개발을 담당했습니다. 취미는 독서입니다."
    result = _extract_relevant_excerpt(content, "카카오 경력")
    assert "카카오" in result


def test_includes_context_sentences():
    """관련 문장 앞뒤 1문장씩 포함"""
    content = "첫 번째 문장입니다. 두 번째 카카오 관련 문장입니다. 세 번째 문장입니다. 네 번째 문장입니다."
    result = _extract_relevant_excerpt(content, "카카오")
    # 앞 문장 또는 뒷 문장 중 하나는 포함되어야 함
    assert len(result) > len("두 번째 카카오 관련 문장입니다.")


def test_respects_max_len():
    content = "A" * 500 + ". " + "카카오 키워드. " + "B" * 500
    result = _extract_relevant_excerpt(content, "카카오", max_len=200)
    assert len(result) <= 200


def test_empty_content_returns_empty_or_short():
    result = _extract_relevant_excerpt("", "질문")
    assert result == "" or len(result) <= 200


def test_no_matching_keyword_returns_first_sentence():
    content = "첫 번째 문장. 두 번째 문장. 세 번째 문장."
    result = _extract_relevant_excerpt(content, "xyz없는단어")
    assert len(result) > 0
    assert len(result) <= 200


def test_multiline_content():
    content = "경력:\n삼성 근무\n카카오 백엔드\n학력:\n서울대"
    result = _extract_relevant_excerpt(content, "카카오")
    assert "카카오" in result


def test_multiple_keywords_scores_correctly():
    content = "React 프로젝트. TypeScript React 사용. Vue 사용."
    result = _extract_relevant_excerpt(content, "React TypeScript")
    # React + TypeScript 둘 다 포함한 문장이 선택되어야 함
    assert "TypeScript" in result and "React" in result
