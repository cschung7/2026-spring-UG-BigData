from fastapi import APIRouter, HTTPException, Query
from services.coingecko import get_top_coins, get_price_history, get_ohlc, search_coins

router = APIRouter(prefix="/api/prices", tags=["prices"])


@router.get("/top")
async def top_coins(limit: int = Query(20, ge=1, le=100)):
    try:
        return await get_top_coins(limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/{coin_id}/history")
async def price_history(coin_id: str, days: int = Query(30, ge=1, le=365)):
    try:
        return await get_price_history(coin_id, days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/{coin_id}/ohlc")
async def ohlc(coin_id: str, days: int = Query(30, ge=1, le=365)):
    try:
        return await get_ohlc(coin_id, days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/search/{query}")
async def search(query: str):
    try:
        return await search_coins(query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
