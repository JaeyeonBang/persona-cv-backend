"""
Graphiti 브릿지 라우터.

Next.js에서 직접 Graphiti 상태를 조회하거나
수동으로 Knowledge Graph를 갱신할 때 사용.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.graphiti_client import is_available, search, add_episode, graph_results_to_context

router = APIRouter(prefix="/api/graphiti")


@router.get("/status")
def graphiti_status():
    """Graphiti 연결 상태 확인."""
    return {"available": is_available()}


class GraphSearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 5


@router.post("/search")
async def graphiti_search(req: GraphSearchRequest):
    """Graphiti 그래프 검색."""
    if not is_available():
        raise HTTPException(status_code=503, detail="Graphiti not available (Neo4j not connected)")

    results = await search(group_id=req.user_id, query=req.query, limit=req.limit)
    return {
        "results": [{"fact": r.fact, "score": r.score} for r in results],
        "context": graph_results_to_context(results),
    }


class GraphIngestRequest(BaseModel):
    user_id: str
    name: str
    content: str
    source: str = "manual"


@router.post("/ingest")
async def graphiti_ingest(req: GraphIngestRequest):
    """수동으로 텍스트를 Knowledge Graph에 추가."""
    if not is_available():
        raise HTTPException(status_code=503, detail="Graphiti not available (Neo4j not connected)")

    success = await add_episode(
        group_id=req.user_id,
        name=req.name,
        content=req.content,
        source=req.source,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Graphiti ingestion failed")

    return {"success": True}
