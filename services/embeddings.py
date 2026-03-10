import logging
import os
import time
from openai import OpenAI

logger = logging.getLogger(__name__)

EMBEDDING_DIMS = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def embed(text: str) -> list[float]:
    """텍스트를 1536차원 임베딩 벡터로 변환합니다. 실패 시 최대 3회 재시도."""
    client = _get_client()
    model = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=model,
                input=text[:24000],
            )
            return response.data[0].embedding
        except Exception as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                logger.warning("embed() attempt %d/%d failed: %s — retrying in %.1fs", attempt, _MAX_RETRIES, e, _RETRY_DELAY)
                time.sleep(_RETRY_DELAY)
            else:
                logger.error("embed() failed after %d attempts: %s", _MAX_RETRIES, e)

    raise last_exc  # type: ignore[misc]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """텍스트를 청크로 분할합니다. 단어 기반, 500단어 청크, 50단어 오버랩."""
    words = [w for w in text.split() if w]
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks
