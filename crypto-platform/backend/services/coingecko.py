import httpx
from typing import Optional

BASE_URL = "https://api.coingecko.com/api/v3"

async def get_top_coins(limit: int = 20) -> list:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h,7d",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def get_price_history(coin_id: str, days: int = 30) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{BASE_URL}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days},
        )
        resp.raise_for_status()
        return resp.json()


async def get_ohlc(coin_id: str, days: int = 30) -> list:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{BASE_URL}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
        )
        resp.raise_for_status()
        return resp.json()


async def search_coins(query: str) -> list:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{BASE_URL}/search", params={"query": query})
        resp.raise_for_status()
        data = resp.json()
        return data.get("coins", [])[:10]
