"""ExamGuard — AI Forensic Intelligence for Examination Integrity.

FastAPI application entry point with CORS, lifespan management,
database initialization, and route registration.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from models.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize DB and directories on startup."""
    settings.ensure_dirs()
    await init_db()
    yield


app = FastAPI(
    title="ExamGuard API",
    description="AI Forensic Intelligence for Examination Integrity — "
                "8 detection engines, GPU-accelerated analysis, LLM-narrated reports",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ──
from api.routes.analysis import router as analysis_router
from api.routes.generator import router as generator_router
from api.routes.health import router as health_router
from api.websocket import router as ws_router

app.include_router(analysis_router, prefix=settings.API_PREFIX, tags=["Analysis"])
app.include_router(generator_router, prefix=settings.API_PREFIX, tags=["Generator"])
app.include_router(health_router, tags=["System"])
app.include_router(ws_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
