"""
방문자 페이지 뷰 카운터
POST /api/views/{username}  — 뷰 카운트 원자적 증가
"""
import logging
from fastapi import APIRouter, HTTPException
from db.supabase import get_client

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/views/{username}", status_code=204)
def increment_view(username: str):
    try:
        supabase = get_client()
        supabase.rpc("increment_view_count", {"p_username": username}).execute()
    except Exception as e:
        logger.warning("view count increment failed for %s: %s", username, e)
        raise HTTPException(status_code=500, detail="view count error")
    return None
