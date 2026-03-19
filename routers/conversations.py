"""
Conversations 라우터.

POST   /api/conversations/{id}/feedback — 피드백 저장 (1 = 좋아요, -1 = 별로예요)
DELETE /api/conversations/{id}          — 단일 대화 삭제
POST   /api/conversations/clear         — 전체 대화 삭제 (body: {user_id})
PATCH  /api/conversations/{id}          — 답변 수정 (body: {answer})
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.supabase import get_client

router = APIRouter()


class FeedbackRequest(BaseModel):
    feedback: int  # 1 or -1


class ClearRequest(BaseModel):
    user_id: str


class UpdateAnswerRequest(BaseModel):
    answer: str


@router.post("/api/conversations/{conversation_id}/feedback", status_code=204)
def update_feedback(conversation_id: str, body: FeedbackRequest):
    """대화 피드백을 저장한다."""
    if body.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
    supabase = get_client()
    supabase.table("conversations").update({"feedback": body.feedback}).eq("id", conversation_id).execute()
    return None


@router.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str):
    """단일 대화를 삭제한다."""
    supabase = get_client()
    supabase.table("conversations").delete().eq("id", conversation_id).execute()
    return None


@router.post("/api/conversations/clear", status_code=204)
def clear_conversations(body: ClearRequest):
    """특정 유저의 전체 대화를 삭제한다."""
    if not body.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    supabase = get_client()
    supabase.table("conversations").delete().eq("user_id", body.user_id).execute()
    return None


@router.patch("/api/conversations/{conversation_id}", status_code=200)
def update_answer(conversation_id: str, body: UpdateAnswerRequest):
    """대화 답변을 수정한다. 수정 후 해당 대화가 캐시 히트 시 수정된 답변이 반환됨."""
    if not body.answer.strip():
        raise HTTPException(status_code=400, detail="answer must not be empty")
    supabase = get_client()
    supabase.table("conversations").update({"answer": body.answer}).eq("id", conversation_id).execute()
    return {"id": conversation_id, "answer": body.answer}
