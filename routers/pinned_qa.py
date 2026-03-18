"""
Pinned Q&A 라우터.

GET  /api/pinned-qa?username=xxx   — 공개 조회
POST /api/pinned-qa                — 새 항목 저장
PUT  /api/pinned-qa/{id}           — 답변 수정
DELETE /api/pinned-qa/{id}         — 삭제
POST /api/pinned-qa/generate       — AI 답변 초안 생성
POST /api/pinned-qa/suggest        — 프로필+문서 기반 면접 Q&A 자동 생성
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from db.supabase import get_client

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic models ──────────────────────────────────────────────────────────

class PinnedQACreate(BaseModel):
    username: str
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=10000)
    display_order: int = 0


class PinnedQAUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    display_order: int | None = None


class GenerateRequest(BaseModel):
    username: str
    question: str = Field(..., min_length=1, max_length=2000)


class SuggestRequest(BaseModel):
    username: str


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_llm(max_tokens: int = 4000) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "z-ai/glm-4.7-flash"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        max_tokens=max_tokens,
    )


def _get_user_by_username(username: str) -> dict:
    supabase = get_client()
    try:
        result = supabase.table("users").select("*").eq("username", username).single().execute()
    except Exception:
        raise HTTPException(status_code=404, detail="User not found")
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")
    return result.data


# ── endpoints ────────────────────────────────────────────────────────────────

@router.get("/api/pinned-qa")
def list_pinned_qa(username: str):
    """해당 사용자의 예상 Q&A 목록을 표시 순서대로 반환."""
    user = _get_user_by_username(username)
    supabase = get_client()
    result = (
        supabase.table("pinned_qa")
        .select("*")
        .eq("user_id", user["id"])
        .order("display_order")
        .execute()
    )
    return result.data or []


@router.post("/api/pinned-qa", status_code=201)
def create_pinned_qa(body: PinnedQACreate):
    """새 예상 Q&A 저장."""
    user = _get_user_by_username(body.username)
    supabase = get_client()
    result = (
        supabase.table("pinned_qa")
        .insert({
            "user_id": user["id"],
            "question": body.question,
            "answer": body.answer,
            "display_order": body.display_order,
        })
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=500, detail="Insert failed")
    return result.data[0]


@router.put("/api/pinned-qa/{item_id}")
def update_pinned_qa(item_id: str, body: PinnedQAUpdate):
    """예상 Q&A 수정 (부분 업데이트)."""
    supabase = get_client()
    updates: dict = {}
    if body.question is not None:
        updates["question"] = body.question
    if body.answer is not None:
        updates["answer"] = body.answer
    if body.display_order is not None:
        updates["display_order"] = body.display_order

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = (
        supabase.table("pinned_qa")
        .update(updates)
        .eq("id", item_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Item not found")
    return result.data[0]


@router.delete("/api/pinned-qa/{item_id}", status_code=204)
def delete_pinned_qa(item_id: str):
    """예상 Q&A 삭제."""
    supabase = get_client()
    supabase.table("pinned_qa").delete().eq("id", item_id).execute()
    return None


@router.post("/api/pinned-qa/generate")
async def generate_answer(body: GenerateRequest):
    """질문에 대한 AI 답변 초안을 생성한다."""
    user = _get_user_by_username(body.username)

    system_prompt = (
        f"당신은 {user['name']}입니다.\n"
        f"직책: {user.get('title', '')}\n"
        f"소개: {user.get('bio', '')}\n\n"
        "아래 질문에 대해 본인의 입장에서 1인칭으로 자연스럽게 답변하세요.\n"
        "마크다운 문법은 사용하지 말고, 5~7문장 정도로 답변하세요."
    )

    try:
        llm = _get_llm(max_tokens=4000)
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=body.question),
        ])
        return {"answer": response.content}
    except Exception as e:
        logger.error("generate_answer failed: %s", e)
        raise HTTPException(status_code=503, detail="AI 답변 생성에 실패했습니다.")


@router.post("/api/pinned-qa/suggest")
async def suggest_qa(body: SuggestRequest):
    """프로필과 업로드 문서를 분석해 면접관이 물어볼 만한 Q&A 5개를 자동 생성한다."""
    user = _get_user_by_username(body.username)
    supabase = get_client()

    # 사용자 문서 조회
    docs_result = (
        supabase.table("documents")
        .select("id, title")
        .eq("user_id", user["id"])
        .limit(5)
        .execute()
    )
    doc_ids = [d["id"] for d in (docs_result.data or [])]

    # 문서별 첫 청크를 [N] title 형식으로 번호화 (채팅 RAG 방식과 동일)
    numbered_docs: list[tuple[int, str, str]] = []  # (index, title, content)
    if doc_ids:
        chunks_result = (
            supabase.table("document_chunks")
            .select("content, document_id")
            .in_("document_id", doc_ids)
            .limit(15)
            .execute()
        )
        doc_title_map = {d["id"]: d["title"] for d in (docs_result.data or [])}
        seen: set[str] = set()
        for chunk in (chunks_result.data or []):
            doc_id = chunk["document_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                title = doc_title_map.get(doc_id, "문서")
                numbered_docs.append((len(numbered_docs) + 1, title, chunk["content"][:800]))

    doc_context = "\n\n---\n\n".join(
        f"[{idx}] {title}\n{content}" for idx, title, content in numbered_docs
    )

    persona_intro = (
        f"당신은 {user.get('name', '')}입니다.\n"
        f"직책: {user.get('title', '')}\n"
        f"소개: {user.get('bio', '')}\n"
        f"참고 자료:\n{doc_context[:2000] if doc_context else '(없음)'}"
    )

    try:
        import asyncio as _asyncio
        llm = _get_llm(max_tokens=8000)

        # 1단계: 면접 질문 5개 생성 (단순 목록 형식)
        q_response = await llm.ainvoke([
            SystemMessage(content=persona_intro),
            HumanMessage(
                content=(
                    "회사 면접관이 당신에게 물어볼 만한 핵심 질문 5개를 작성하세요. "
                    "한 줄에 하나씩, 번호 없이 질문만 나열하세요."
                )
            ),
        ])
        raw_questions = q_response.content.strip()
        logger.info("suggest_qa questions raw: %s", raw_questions[:300])

        questions_list = [
            line.strip().lstrip("0123456789.-). ").strip()
            for line in raw_questions.splitlines()
            if line.strip() and len(line.strip()) > 5
        ][:5]

        if not questions_list:
            raise ValueError(f"질문 생성 실패: {raw_questions[:100]}")

        # 2단계: 각 질문에 대한 답변을 병렬 생성 (채팅 RAG 방식과 동일하게 출처 표기)
        citation_rule = (
            "\n\n## 출처 표기 규칙 (반드시 준수)\n"
            "참고 자료의 내용을 활용한 문장 끝에는 반드시 [출처: 정확한_제목] 형식으로 출처를 표기하세요.\n"
            "예시: '저는 해당 프로젝트에서 MAU 10만을 달성했습니다. [출처: 방재연_이력서.pdf]'\n"
            "참고 자료를 사용하지 않은 일반적인 내용에는 출처를 붙이지 마세요."
        ) if numbered_docs else ""

        answer_system = (
            f"{persona_intro}{citation_rule}\n\n"
            "아래 질문에 대해 본인의 입장에서 1인칭으로 자연스럽게 3~5문장으로 답변하세요. "
            "마크다운 문법은 사용하지 마세요."
        )

        async def gen_answer(question: str) -> str:
            r = await llm.ainvoke([
                SystemMessage(content=answer_system),
                HumanMessage(content=question),
            ])
            return r.content.strip()

        answers = await _asyncio.gather(*[gen_answer(q) for q in questions_list])

        sources = [{"id": d["id"], "title": d["title"]} for d in (docs_result.data or [])]
        pairs = [
            {"question": q, "answer": a, "sources": sources}
            for q, a in zip(questions_list, answers)
            if a
        ]
        if not pairs:
            raise ValueError("답변 생성 결과 없음")
        logger.info("suggest_qa produced %d pairs", len(pairs))
        return pairs
    except Exception as e:
        logger.error("suggest_qa failed: %s", e)
        raise HTTPException(status_code=503, detail="AI 질문 생성에 실패했습니다.")
