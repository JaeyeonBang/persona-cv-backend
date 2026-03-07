from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from services.document_processor import process_document

router = APIRouter()


class ProcessRequest(BaseModel):
    documentIds: list[str]


@router.post("/api/documents/process")
async def process_documents(req: ProcessRequest, background_tasks: BackgroundTasks):
    if not req.documentIds:
        raise HTTPException(status_code=400, detail="documentIds is required")

    for doc_id in req.documentIds:
        background_tasks.add_task(process_document, doc_id)

    return {"queued": len(req.documentIds)}
