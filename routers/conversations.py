"""
Conversations 라우터.

POST /api/conversations/{id}/feedback — 피드백 저장 (1 = 좋아요, -1 = 별로예요)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.supabase import get_client

router = APIRouter()


class FeedbackRequest(BaseModel):
    feedback: int  # 1 or -1


@router.post("/api/conversations/{conversation_id}/feedback", status_code=204)
def update_feedback(conversation_id: str, body: FeedbackRequest):
    """대화 피드백을 저장한다."""
    if body.feedback not in (1, -1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1")
    supabase = get_client()
    supabase.table("conversations").update({"feedback": body.feedback}).eq("id", conversation_id).execute()
    return None
