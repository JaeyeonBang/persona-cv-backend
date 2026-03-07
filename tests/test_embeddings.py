import pytest
from services.embeddings import chunk_text, EMBEDDING_DIMS


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\t  ") == []


def test_less_than_chunk_size_returns_one_chunk():
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_exactly_500_words_returns_one_chunk():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words))
    assert len(chunks) == 1


def test_501_words_returns_two_chunks():
    words = [f"w{i}" for i in range(501)]
    chunks = chunk_text(" ".join(words))
    assert len(chunks) == 2


def test_second_chunk_starts_with_overlap():
    # 600단어: 첫 청크 w0~w499, 두 번째 청크 시작 w450
    words = [f"w{i}" for i in range(600)]
    chunks = chunk_text(" ".join(words))
    assert chunks[1].startswith("w450")


def test_custom_chunk_size_and_overlap():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert chunks[0] == " ".join(words[:10])


def test_multiple_spaces_treated_as_single_separator():
    chunks = chunk_text("a  b\n\nc   d")
    assert len(chunks) == 1
    assert chunks[0] == "a b c d"


def test_embedding_dims_is_1536():
    assert EMBEDDING_DIMS == 1536
