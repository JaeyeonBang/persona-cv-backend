# persona-cv-backend

FastAPI 백엔드. Next.js 프론트엔드의 thin proxy를 통해 호출됨.

## 실행

```bash
uv sync
uv run uvicorn main:app --port 8001 --reload
```

## 환경변수 (`backend/.env`)

```
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=z-ai/glm-4.7-flash
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

## API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| POST | `/api/chat` | SSE 스트리밍 채팅 (RAG 포함) |
| POST | `/api/documents/process` | 문서 임베딩 처리 (백그라운드) |
| GET | `/health` | 헬스체크 |

## 구조

```
main.py                    — FastAPI 앱 진입점
routers/
  chat.py                  — POST /api/chat
  documents.py             — POST /api/documents/process
services/
  embeddings.py            — embed(), chunk_text()
  search.py                — search_documents(), build_context()
  llm.py                   — build_system_prompt(), stream_chat()
  document_processor.py    — process_document() (PDF/URL → pgvector)
db/
  supabase.py              — Supabase Admin 클라이언트 (싱글턴)
```

## 주요 사항

- GLM-4.7 Flash는 reasoning 모델 → `max_tokens: 2000` 필수
- 임베딩: OpenRouter ZDR(Zero Data Retention) 반드시 비활성화
- PDF 파싱: `pdfplumber` 사용
- 문서 처리는 `BackgroundTasks`로 비동기 실행
- Supabase RPC `match_documents` 함수로 pgvector 검색

## 테스트

```bash
# pytest (예정)
pytest tests/
```
