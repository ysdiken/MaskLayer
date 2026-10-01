import logging
import os
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from app.api.routes import router
from app.api.upload_routes import router as upload_router
from app.api.chat_routes import router as chat_router
from app.api.correction_routes import router as correction_router
from app.api.ablation_routes import router as ablation_router
from app.api.admin_routes import router as admin_router
from app.db.session import init_db, close_db

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    app.state.redis = Redis.from_url(redis_url, decode_responses=False)
    print(f"MaskLayer backend starting — Redis: {redis_url.split('@')[-1]}")

    # PostgreSQL — create tables if not exists
    try:
        await init_db()
    except Exception as e:
        # DB unavailable: log and continue — masking still works without audit
        print(f"WARNING: PostgreSQL unavailable ({e}). Audit logging disabled.")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await app.state.redis.aclose()
    print("MaskLayer backend stopped — Redis connection closed")

    await close_db()


app = FastAPI(title="MaskLayer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(correction_router)
app.include_router(ablation_router)
app.include_router(admin_router)
