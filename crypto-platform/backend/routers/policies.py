from fastapi import APIRouter, Query
from services.policies import get_all_policies
from services.news_aggregator import get_policy_news

router = APIRouter(prefix="/api/policies", tags=["policies"])


@router.get("/")
async def list_policies(
    country: str = Query(None),
    impact: str = Query(None, regex="^(bullish|bearish|neutral)$"),
):
    return get_all_policies(country=country, impact=impact)


@router.get("/news")
async def policy_news_feed(limit: int = Query(20, ge=1, le=60)):
    return await get_policy_news(limit)
