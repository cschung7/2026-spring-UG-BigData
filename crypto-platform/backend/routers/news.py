from fastapi import APIRouter, HTTPException, Query
from services.news_aggregator import get_news, get_policy_news

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("/")
async def latest_news(limit: int = Query(40, ge=1, le=100)):
    try:
        return await get_news(limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/policy")
async def policy_news(limit: int = Query(30, ge=1, le=100)):
    try:
        return await get_policy_news(limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
