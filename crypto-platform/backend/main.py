from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import prices, news, policies

app = FastAPI(title="Crypto Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prices.router)
app.include_router(news.router)
app.include_router(policies.router)


@app.get("/")
async def root():
    return {"message": "Crypto Platform API", "docs": "/docs"}
