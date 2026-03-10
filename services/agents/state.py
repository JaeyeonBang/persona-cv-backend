from __future__ import annotations

from typing import Any, TypedDict


class ChatState(TypedDict):
    # ─── Input ────────────────────────────────────────────────────────────
    username: str
    question: str
    config: dict
    history: list[dict]
    session_id: str | None

    # ─── DB ───────────────────────────────────────────────────────────────
    user_data: dict

    # ─── Retrieval Node ───────────────────────────────────────────────────
    query_embedding: list[float]        # 사전 계산 시 재사용 (캐시 체크용)
    search_results: list[Any]           # list[SearchResult]
    context: str
    graph_fallback: bool
    citations: list[dict]
    pinned_qa_context: str              # 사전 준비된 Q&A 컨텍스트

    # ─── Persona Node ─────────────────────────────────────────────────────
    draft_answer: str

    # ─── Fact-Check Node ──────────────────────────────────────────────────
    fact_check_passed: bool
    fact_check_notes: str
    final_answer: str
