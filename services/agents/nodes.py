"""
LangGraph 노드 정의.

retrieval_node  — 임베딩 + 하이브리드 검색
persona_node    — 페르소나 답변 생성 (LLM)
factcheck_node  — 문서 근거 팩트체크 (LLM)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from services.agents.state import ChatState
from services.embeddings import embed
from services.llm import build_system_prompt
from services.search import build_context, hybrid_search

logger = logging.getLogger(__name__)


# ─── 공통 LLM 팩토리 ───────────────────────────────────────────────────────

def _get_llm(max_tokens: int = 2000) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "z-ai/glm-4.7-flash"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        max_tokens=max_tokens,
    )


def _extract_relevant_excerpt(content: str, query: str, max_len: int = 200) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+|(?<=\n)", content) if s.strip()]
    if not sentences:
        return content[:max_len]
    query_words = set(re.sub(r"[^\w]", " ", query.lower()).split())
    if not query_words:
        return sentences[0][:max_len]

    def score(sentence: str) -> int:
        words = set(re.sub(r"[^\w]", " ", sentence.lower()).split())
        return len(query_words & words)

    best = max(sentences, key=score)
    idx = sentences.index(best)
    parts = sentences[max(0, idx - 1) : idx + 2]
    excerpt = " ".join(parts)
    return excerpt[:max_len] if len(excerpt) > max_len else excerpt


# ─── 노드 1: 검색 ──────────────────────────────────────────────────────────

async def retrieval_node(state: ChatState) -> dict[str, Any]:
    """질문 임베딩 + pgvector / Graphiti 하이브리드 검색."""
    question = state["question"]
    user_id = state["user_data"]["id"]

    # 사전 계산된 임베딩 재사용 (캐시 체크 시 이미 계산한 경우)
    query_embedding: list[float] = state.get("query_embedding") or []
    if not query_embedding:
        query_embedding = embed(question)

    search_results, graph_context = await hybrid_search(
        user_id=user_id,
        query=question,
        query_embedding=query_embedding,
    )

    context = build_context(search_results)
    if graph_context:
        context = f"{context}\n\n{graph_context}".strip() if context else graph_context

    citations = [
        {
            "index": i + 1,
            "title": r.title,
            "excerpt": _extract_relevant_excerpt(r.content, question),
            "url": r.source_url,
            "source": r.source,
        }
        for i, r in enumerate(search_results)
        if r.similarity >= 0.25
    ]

    return {
        "query_embedding": query_embedding,
        "search_results": search_results,
        "context": context,
        "graph_fallback": bool(graph_context),
        "citations": citations,
    }


# ─── 노드 2: 페르소나 답변 생성 ────────────────────────────────────────────

async def persona_node(state: ChatState) -> dict[str, Any]:
    """페르소나 에이전트: 인물처럼 답변 생성."""
    system_prompt = build_system_prompt(
        state["user_data"],
        state["config"],
        state["context"],
        state["search_results"],
    )

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for msg in state.get("history", []):
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=state["question"]))

    response = await _get_llm().ainvoke(messages)
    return {"draft_answer": response.content}


# ─── 노드 3: 팩트체크 ──────────────────────────────────────────────────────

FACTCHECK_SYSTEM_PROMPT = """당신은 팩트체커입니다.
아래 참고 자료와 답변을 비교하여, 답변에 문서에 없는 사실적 주장이 포함되어 있는지 확인합니다.

반드시 JSON 형식으로만 응답하세요:
{"verdict": "pass", "notes": "근거 충분"}
{"verdict": "warn", "notes": "구체적으로 어떤 내용이 문서에 없는지"}

규칙:
- 참고 자료에 없는 수치·날짜·고유명사가 답변에 등장하면 "warn"
- 일반적 상식이나 자기소개 수준의 내용은 "pass"
- 참고 자료가 부족해 검증 불가능한 경우도 "pass" (과도한 경고 방지)"""


async def factcheck_node(state: ChatState) -> dict[str, Any]:
    """팩트체크 에이전트: 답변이 문서 근거에 충실한지 검증."""
    context = state["context"]
    draft = state["draft_answer"]

    # 검색 컨텍스트 없으면 팩트체크 생략
    if not context or not context.strip():
        return {
            "fact_check_passed": True,
            "fact_check_notes": "참고 문서 없음 — 팩트체크 생략",
            "final_answer": draft,
        }

    # 답변이 짧으면 팩트체크 불필요
    if len(draft.strip()) < 100:
        return {
            "fact_check_passed": True,
            "fact_check_notes": "짧은 답변 — 팩트체크 생략",
            "final_answer": draft,
        }

    user_message = (
        f"참고 자료:\n{context[:2000]}\n\n"
        f"질문: {state['question']}\n\n"
        f"답변:\n{draft}"
    )

    response = await _get_llm(max_tokens=300).ainvoke(
        [
            SystemMessage(content=FACTCHECK_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )

    verdict = "pass"
    notes = ""
    try:
        raw = response.content.strip()
        # ```json ... ``` 래핑 제거
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "pass")
        notes = parsed.get("notes", "")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Fact-check parse failed: %s", response.content[:120])

    return {
        "fact_check_passed": verdict == "pass",
        "fact_check_notes": notes,
        "final_answer": draft,
    }
