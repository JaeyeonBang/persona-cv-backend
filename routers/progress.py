from fastapi import APIRouter, HTTPException
from db.supabase import get_client
from services.document_processor import get_progress

router = APIRouter()


@router.get("/api/documents/{document_id}/progress")
def document_progress(document_id: str):
    supabase = get_client()
    result = supabase.table("documents").select("status").eq("id", document_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Document not found")

    status = result.data["status"]

    if status == "done":
        progress = 100
    elif status == "error":
        progress = 0
    elif status == "pending":
        progress = 0
    else:
        progress = get_progress(document_id)

    return {"status": status, "progress": progress}
