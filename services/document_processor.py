import httpx
import pdfplumber
import io
import os
import re
from urllib.parse import unquote
from bs4 import BeautifulSoup
from db.supabase import get_client
from services.embeddings import embed, chunk_text

_progress: dict[str, int] = {}


def get_progress(document_id: str) -> int:
    return _progress.get(document_id, 0)


def _storage_path_from_url(url: str) -> str | None:
    """Supabase storage public URL에서 버킷 내 경로를 추출한다.

    URL 패턴: https://<project>.supabase.co/storage/v1/object/public/documents/<path>
    SUPABASE_URL 환경변수 대신 URL 패턴 매칭으로 추출해 환경변수 로딩 순서에 무관하게 동작한다.
    """
    match = re.search(r'/storage/v1/object/public/documents/(.+)', url)
    if match:
        return match.group(1)
    return None


async def fetch_pdf_bytes(url: str) -> bytes:
    """Supabase storage URL인 경우 인증된 엔드포인트를 통해 직접 다운로드.
    
    documents 버킷이 private이며, 파일명 인코딩 문제로 SDK의 download나
    create_signed_url 조차 실패하는 경우가 있음.
    따라서 Service Role Key를 사용해 /object/authenticated/ URL로 직접 GET 요청.
    """
    path = _storage_path_from_url(url)
    if path:
        decoded_path = unquote(path)
        supabase = get_client()
        base_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        
        # Authenticated storage endpoint
        auth_url = f"{base_url}/storage/v1/object/authenticated/documents/{decoded_path}"
        headers = {
            "Authorization": f"Bearer {key}",
            "apikey": key
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(auth_url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                return response.content
            except Exception as e:
                import logging
                logging.warning(f"Authenticated download failed ({e}), falling back to public url: {url}")

    # 일반 URL이나 fallback의 경우 직접 다운로드 시도
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content




def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


def _extract_text_from_html(html: str) -> str:
    """HTML에서 스크립트/스타일 제거 후 순수 텍스트 추출."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "link", "noscript", "svg", "img"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # 연속 공백 정리
    return re.sub(r"\s+", " ", text).strip()


async def fetch_url_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type or url.endswith((".html", ".htm")):
        return _extract_text_from_html(response.text)
    # plain text (markdown, txt 등)
    return response.text


async def process_document(document_id: str) -> None:
    supabase = get_client()
    _progress[document_id] = 0

    result = supabase.table("documents").select("*").eq("id", document_id).single().execute()
    doc = result.data
    if not doc:
        _progress.pop(document_id, None)
        raise ValueError(f"Document not found: {document_id}")

    try:
        supabase.table("documents").update({
            "status": "processing",
            "error_message": None  # 이전 에러 메시지 초기화
        }).eq("id", document_id).execute()
        _progress[document_id] = 5

        # 텍스트 추출
        if doc["type"] == "pdf":
            pdf_bytes = await fetch_pdf_bytes(doc["source_url"])
            text = extract_text_from_pdf(pdf_bytes)
        else:
            text = await fetch_url_text(doc["source_url"])


        if not text.strip():
            raise ValueError("No text content extracted")

        _progress[document_id] = 20

        # 청크 분할 + 임베딩 (20~90%)
        chunks = chunk_text(text)
        total = len(chunks)
        chunk_records = []

        for i, chunk in enumerate(chunks):
            embedding = embed(chunk)
            chunk_records.append({
                "document_id": document_id,
                "user_id": doc["user_id"],
                "title": doc["title"],
                "content": chunk,
                "embedding": embedding,
                "chunk_index": i,
            })
            _progress[document_id] = 20 + int((i + 1) / total * 70)

        # document_chunks에 청크 저장
        if chunk_records:
            supabase.table("document_chunks").insert(chunk_records).execute()

        _progress[document_id] = 95

        # documents 테이블 상태 업데이트
        supabase.table("documents").update({
            "content": text,
            "status": "done",
        }).eq("id", document_id).execute()

        _progress[document_id] = 100

    except Exception as e:
        supabase.table("documents").update({
            "status": "error",
            "error_message": str(e),
        }).eq("id", document_id).execute()
        raise e
    finally:
        _progress.pop(document_id, None)
