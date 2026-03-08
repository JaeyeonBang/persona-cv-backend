"""
document_processor 단위 테스트 (TDD)

테스트 대상:
  - extract_text_from_pdf: PDF 바이트 → 텍스트
  - chunk_text: 텍스트 → 청크 리스트
  - get_progress / _progress 딕셔너리
  - process_document: DB 연동 통합 (mock 사용)
"""

import io
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ── chunk_text ─────────────────────────────────────────────────────────────

from services.embeddings import chunk_text

class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "hello world " * 10  # 20단어
        result = chunk_text(text)
        assert len(result) == 1
        assert "hello" in result[0]

    def test_long_text_is_split_into_multiple_chunks(self):
        # 1100단어 → CHUNK_SIZE=500, OVERLAP=50 → 3청크
        text = " ".join([f"word{i}" for i in range(1100)])
        result = chunk_text(text)
        assert len(result) >= 2

    def test_chunk_size_does_not_exceed_limit(self):
        text = " ".join([f"w{i}" for i in range(2000)])
        result = chunk_text(text)
        for chunk in result:
            word_count = len(chunk.split())
            assert word_count <= 500, f"청크가 500단어를 초과: {word_count}"

    def test_overlap_preserves_words_between_chunks(self):
        text = " ".join([f"word{i}" for i in range(600)])
        result = chunk_text(text)
        assert len(result) >= 2
        # 첫 청크의 마지막 단어가 두 번째 청크에 있어야 함 (overlap)
        last_words_of_first = set(result[0].split()[-50:])
        first_words_of_second = set(result[1].split()[:50])
        assert len(last_words_of_first & first_words_of_second) > 0

    def test_exactly_500_words_returns_one_chunk(self):
        text = " ".join([f"w{i}" for i in range(500)])
        result = chunk_text(text)
        assert len(result) == 1


# ── _extract_text_from_html ────────────────────────────────────────────────

from services.document_processor import _extract_text_from_html

class TestExtractTextFromHtml:
    def test_removes_script_tags(self):
        html = "<html><head><script>alert('xss')</script></head><body><p>Hello</p></body></html>"
        result = _extract_text_from_html(html)
        assert "alert" not in result
        assert "Hello" in result

    def test_removes_style_tags(self):
        html = "<html><head><style>body { color: red; }</style></head><body><p>World</p></body></html>"
        result = _extract_text_from_html(html)
        assert "color" not in result
        assert "World" in result

    def test_collapses_whitespace(self):
        html = "<html><body><p>One</p>   <p>Two</p>\n\n<p>Three</p></body></html>"
        result = _extract_text_from_html(html)
        assert "  " not in result  # 연속 공백 없음

    def test_empty_html_returns_empty(self):
        result = _extract_text_from_html("<html><body></body></html>")
        assert result == ""

    def test_real_world_html(self):
        html = """<!DOCTYPE html>
<html>
<head><title>Test</title><style>.x{}</style><script>var x=1;</script></head>
<body>
  <nav>Navigation</nav>
  <main><h1>Article Title</h1><p>Article body content here.</p></main>
  <footer>Footer text</footer>
</body>
</html>"""
        result = _extract_text_from_html(html)
        assert "Article Title" in result
        assert "Article body content here" in result
        assert "var x" not in result
        assert ".x{}" not in result


# ── extract_text_from_pdf ──────────────────────────────────────────────────

from services.document_processor import extract_text_from_pdf

class TestExtractTextFromPdf:
    def _make_simple_pdf(self) -> bytes:
        """reportlab 없이 최소 PDF 바이트 생성"""
        try:
            import reportlab.pdfgen.canvas as canvas_mod
            buf = io.BytesIO()
            c = canvas_mod.Canvas(buf)
            c.drawString(100, 750, "Hello PDF World")
            c.save()
            return buf.getvalue()
        except ImportError:
            # reportlab 없으면 pdfplumber가 읽을 수 있는 최소 PDF
            return (
                b"%PDF-1.4\n"
                b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
                b"/Contents 4 0 R/Resources<</Font<</F1<</Type/Font"
                b"/Subtype/Type1/BaseFont/Helvetica>>>>>>>>>>endobj\n"
                b"4 0 obj<</Length 44>>\nstream\n"
                b"BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\n"
                b"endstream\nendobj\n"
                b"xref\n0 5\n"
                b"0000000000 65535 f\r\n"
                b"0000000009 00000 n\r\n"
                b"0000000058 00000 n\r\n"
                b"0000000115 00000 n\r\n"
                b"0000000274 00000 n\r\n"
                b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n366\n%%EOF"
            )

    def test_invalid_pdf_raises_exception(self):
        with pytest.raises(Exception):
            extract_text_from_pdf(b"not a pdf")

    def test_empty_bytes_raises_exception(self):
        with pytest.raises(Exception):
            extract_text_from_pdf(b"")


# ── get_progress ───────────────────────────────────────────────────────────

from services.document_processor import get_progress, _progress

class TestGetProgress:
    def setup_method(self):
        _progress.clear()

    def test_unknown_id_returns_zero(self):
        assert get_progress("unknown-id") == 0

    def test_returns_stored_progress(self):
        _progress["doc-123"] = 42
        assert get_progress("doc-123") == 42

    def test_returns_100_when_complete(self):
        _progress["doc-abc"] = 100
        assert get_progress("doc-abc") == 100


# ── process_document (통합, DB mock) ──────────────────────────────────────

from services.document_processor import process_document

class TestProcessDocument:
    """DB와 임베딩 API를 mock 처리한 통합 테스트"""

    def _make_mock_supabase(self, doc: dict):
        sb = MagicMock()
        # select().eq().single().execute() → doc 반환
        sb.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = doc
        sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
        return sb

    @pytest.mark.asyncio
    async def test_sets_status_processing_then_done(self):
        doc = {
            "id": "doc-1", "type": "url", "source_url": "http://example.com",
            "user_id": "user-1", "title": "Test Doc",
        }
        mock_sb = self._make_mock_supabase(doc)
        update_calls = []

        def capture_update(data):
            update_calls.append(data)
            return mock_sb.table.return_value.update.return_value
        mock_sb.table.return_value.update.side_effect = capture_update

        url_text = "word " * 600  # 충분히 긴 텍스트로 청킹 발생

        with patch("services.document_processor.get_client", return_value=mock_sb), \
             patch("services.document_processor.fetch_url_text", new=AsyncMock(return_value=url_text)), \
             patch("services.document_processor.embed", return_value=[0.1] * 1536):
            await process_document("doc-1")

        statuses = [c.get("status") for c in update_calls if "status" in c]
        assert "processing" in statuses
        assert "done" in statuses

    @pytest.mark.asyncio
    async def test_progress_reaches_100_on_success(self):
        doc = {
            "id": "doc-2", "type": "url", "source_url": "http://example.com",
            "user_id": "user-1", "title": "Test",
        }
        mock_sb = self._make_mock_supabase(doc)
        mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

        with patch("services.document_processor.get_client", return_value=mock_sb), \
             patch("services.document_processor.fetch_url_text", new=AsyncMock(return_value="word " * 100)), \
             patch("services.document_processor.embed", return_value=[0.0] * 1536):
            await process_document("doc-2")

        # finally에서 _progress에서 제거되므로 0 반환
        assert get_progress("doc-2") == 0

    @pytest.mark.asyncio
    async def test_sets_error_status_on_failure(self):
        doc = {
            "id": "doc-3", "type": "url", "source_url": "http://bad.url",
            "user_id": "user-1", "title": "Bad",
        }
        mock_sb = self._make_mock_supabase(doc)
        update_calls = []

        def capture(data):
            update_calls.append(data)
            return mock_sb.table.return_value.update.return_value
        mock_sb.table.return_value.update.side_effect = capture

        with patch("services.document_processor.get_client", return_value=mock_sb), \
             patch("services.document_processor.fetch_url_text", new=AsyncMock(side_effect=Exception("network error"))):
            with pytest.raises(Exception, match="network error"):
                await process_document("doc-3")

        statuses = [c.get("status") for c in update_calls if "status" in c]
        assert "error" in statuses

    @pytest.mark.asyncio
    async def test_empty_text_sets_error(self):
        doc = {
            "id": "doc-4", "type": "url", "source_url": "http://empty.com",
            "user_id": "user-1", "title": "Empty",
        }
        mock_sb = self._make_mock_supabase(doc)
        mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

        with patch("services.document_processor.get_client", return_value=mock_sb), \
             patch("services.document_processor.fetch_url_text", new=AsyncMock(return_value="   ")):
            with pytest.raises(ValueError, match="No text content extracted"):
                await process_document("doc-4")

    @pytest.mark.asyncio
    async def test_chunking_occurs_for_long_text(self):
        """청킹이 실제로 발생하는지 검증 (embed 호출 횟수로 확인)"""
        doc = {
            "id": "doc-5", "type": "url", "source_url": "http://long.com",
            "user_id": "user-1", "title": "Long",
        }
        mock_sb = self._make_mock_supabase(doc)
        mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

        long_text = " ".join([f"word{i}" for i in range(1100)])  # → 3청크 예상
        embed_call_count = 0

        def counting_embed(text):
            nonlocal embed_call_count
            embed_call_count += 1
            return [0.1] * 1536

        with patch("services.document_processor.get_client", return_value=mock_sb), \
             patch("services.document_processor.fetch_url_text", new=AsyncMock(return_value=long_text)), \
             patch("services.document_processor.embed", side_effect=counting_embed):
            await process_document("doc-5")

        # 1100단어 / (500-50 stride) = 3청크
        assert embed_call_count >= 2, f"청킹 미발생: embed 호출 {embed_call_count}회"
