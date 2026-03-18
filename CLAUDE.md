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
NEO4J_URI=bolt://localhost:7687       # Graphiti 사용 시
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
ALLOWED_ORIGINS=http://localhost:3000  # 콤마 구분 다중 허용
```

## API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| POST | `/api/chat` | SSE 스트리밍 채팅 (RAG 포함) |
| POST | `/api/documents/process` | 문서 임베딩 처리 (백그라운드) |
| GET  | `/api/document-progress/{id}` | 문서 처리 진행률 조회 |
| GET  | `/api/pinned-qa?username=xxx` | 예상 Q&A 목록 공개 조회 |
| POST | `/api/pinned-qa` | 예상 Q&A 저장 |
| PUT  | `/api/pinned-qa/{id}` | 예상 Q&A 수정 |
| DELETE | `/api/pinned-qa/{id}` | 예상 Q&A 삭제 |
| POST | `/api/pinned-qa/generate` | Q&A AI 답변 초안 생성 |
| POST | `/api/pinned-qa/suggest` | 프로필 기반 면접 Q&A 자동 제안 |
| POST | `/api/conversations/{id}/feedback` | 대화 피드백 저장 (1/-1) |
| GET  | `/api/graphiti/status` | Graphiti 연결 상태 |
| POST | `/api/graphiti/search` | Graphiti 그래프 검색 |
| POST | `/api/graphiti/ingest` | Graphiti 에피소드 인덱싱 |
| GET  | `/health` | 헬스체크 |

## 구조

```
main.py                    — FastAPI 앱 진입점
routers/
  chat.py                  — POST /api/chat (SSE + conversation_id 이벤트 포함)
  documents.py             — POST /api/documents/process
  progress.py              — GET /api/document-progress/{id}
  pinned_qa.py             — 예상 Q&A CRUD + AI 생성
  conversations.py         — POST /api/conversations/{id}/feedback
  graphiti.py              — Graphiti 브릿지
services/
  embeddings.py            — embed(), chunk_text()
  search.py                — hybrid_search(), build_context()
  llm.py                   — build_system_prompt(), stream_chat()
  document_processor.py    — process_document() (PDF/URL → pgvector)
  conversations.py         — check_cache(), save_conversation()
  graphiti_client.py       — Graphiti 래퍼 (graceful fallback)
  agents/                  — LangGraph 노드 (retrieval, persona, factcheck)
db/
  supabase.py              — Supabase Admin 클라이언트 (싱글턴)
```

## 주요 사항

- GLM-4.7 Flash는 reasoning 모델 → `max_tokens: 2000` 필수
- 임베딩: OpenRouter ZDR(Zero Data Retention) 반드시 비활성화
- PDF 파싱: `pdfplumber` 사용
- 문서 처리는 `BackgroundTasks`로 비동기 실행
- Supabase RPC `match_documents` 함수로 pgvector 검색
- `/api/chat` SSE 이벤트: `conversation_id`, `cache_hit`, `graph_fallback`, `status`, `citations`, `text`, `fact_check_warn`, `error`, `[DONE]`
- Graphiti: Neo4j 미연결 시 graceful fallback (vector 검색만 사용)

## 테스트

```bash
uv run pytest tests/
```
