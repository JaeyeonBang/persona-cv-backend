import httpx
import pdfplumber
import io
from db.supabase import get_client
from services.embeddings import embed, chunk_text


async def fetch_pdf_bytes(storage_path: str) -> bytes:
    """Supabase Storage에서 PDF 파일을 다운로드합니다."""
    supabase = get_client()
    response = supabase.storage.from_("documents").download(storage_path)
    return response


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """PDF 바이트에서 텍스트를 추출합니다."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


async def fetch_url_text(url: str) -> str:
    """URL에서 텍스트 콘텐츠를 가져옵니다."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    # 간단히 텍스트로 반환 (HTML 파싱 없음 — MVP)
    return response.text


async def process_document(document_id: str) -> None:
    """
    단일 문서를 처리합니다:
    1. DB에서 문서 정보 조회
    2. 소스 타입에 따라 텍스트 추출
    3. 청크 분할 → 임베딩 생성
    4. document_chunks 테이블에 저장
    5. 문서 status를 'done'으로 업데이트
    """
    supabase = get_client()

    # 1. 문서 조회
    result = supabase.table("documents").select("*").eq("id", document_id).single().execute()
    doc = result.data
    if not doc:
        raise ValueError(f"Document not found: {document_id}")

    try:
        # 2. 텍스트 추출
        if doc["source_type"] == "pdf":
            pdf_bytes = await fetch_pdf_bytes(doc["storage_path"])
            text = extract_text_from_pdf(pdf_bytes)
        else:
            text = await fetch_url_text(doc["source_url"])

        if not text.strip():
            raise ValueError("No text content extracted")

        # 3. 청크 분할 + 임베딩
        chunks = chunk_text(text)
        chunk_records = []
        for i, chunk in enumerate(chunks):
            embedding = embed(chunk)
            chunk_records.append(
                {
                    "document_id": document_id,
                    "user_id": doc["user_id"],
                    "title": doc["title"],
                    "content": chunk,
                    "embedding": embedding,
                    "chunk_index": i,
                }
            )

        # 4. document_chunks 저장
        if chunk_records:
            supabase.table("document_chunks").insert(chunk_records).execute()

        # 5. 상태 업데이트
        supabase.table("documents").update({"status": "done"}).eq("id", document_id).execute()

    except Exception as e:
        supabase.table("documents").update({"status": "error"}).eq("id", document_id).execute()
        raise e
