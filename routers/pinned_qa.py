"""
Pinned Q&A 라우터.

GET  /api/pinned-qa?username=xxx   — 공개 조회
POST /api/pinned-qa                — 새 항목 저장
PUT  /api/pinned-qa/{id}           — 답변 수정
DELETE /api/pinned-qa/{id}         — 삭제
POST /api/pinned-qa/generate       — AI 답변 초안 생성
"""

from __future__ import annotations

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


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "z-ai/glm-4.7-flash"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        max_tokens=1000,
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
        llm = _get_llm()
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=body.question),
        ])
        return {"answer": response.content}
    except Exception as e:
        logger.error("generate_answer failed: %s", e)
        raise HTTPException(status_code=503, detail="AI 답변 생성에 실패했습니다.")
