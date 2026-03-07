import pytest
from services.search import build_context, SearchResult


def make_result(id: str, title: str, content: str, similarity: float) -> SearchResult:
    return SearchResult(id=id, title=title, content=content, similarity=similarity)


def test_empty_results_returns_empty_string():
    assert build_context([]) == ""


def test_all_below_threshold_returns_empty_string():
    results = [
        make_result("1", "A", "content", 0.1),
        make_result("2", "B", "content", 0.29),
    ]
    assert build_context(results) == ""


def test_only_above_threshold_included():
    results = [
        make_result("1", "Pass", "pass content", 0.5),
        make_result("2", "Fail", "fail content", 0.1),
    ]
    ctx = build_context(results)
    assert "Pass" in ctx
    assert "Fail" not in ctx


def test_exactly_at_threshold_included():
    results = [make_result("1", "Edge", "edge content", 0.3)]
    assert "Edge" in build_context(results)


def test_format_includes_numbered_title():
    results = [make_result("1", "My Doc", "some content", 0.8)]
    ctx = build_context(results)
    assert "[1] My Doc" in ctx


def test_multiple_results_separated_by_dashes():
    results = [
        make_result("1", "A", "content a", 0.9),
        make_result("2", "B", "content b", 0.8),
    ]
    ctx = build_context(results)
    assert "---" in ctx
    assert "[1] A" in ctx
    assert "[2] B" in ctx


def test_content_truncated_at_1500_chars():
    long_content = "x" * 2000
    results = [make_result("1", "Long", long_content, 0.9)]
    ctx = build_context(results)
    assert "x" * 1501 not in ctx
    assert "x" * 1500 in ctx


def test_custom_threshold_applied():
    results = [
        make_result("1", "High", "content", 0.8),
        make_result("2", "Low", "content", 0.4),
    ]
    ctx = build_context(results, threshold=0.6)
    assert "High" in ctx
    assert "Low" not in ctx
