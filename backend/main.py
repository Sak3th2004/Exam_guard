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

# ── Health endpoint ──
@app.get("/health", tags=["System"])
async def health_check():
    """Check system health including GPU and Ollama availability."""
    import httpx

    gpu_available = False
    gpu_name = None
    gpu_memory_mb = None
    ollama_available = False
    ollama_model = None

    # GPU check
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    except ImportError:
        pass

    # Ollama check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                ollama_available = True
                models = resp.json().get("models", [])
                for m in models:
                    if "mistral" in m.get("name", "").lower():
                        ollama_model = m["name"]
                        break
    except Exception:
        pass

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_memory_mb,
        "ollama_available": ollama_available,
        "ollama_model": ollama_model,
    }


# ── Import and register routes (will be added in Phase 6) ──
# from api.routes.analysis import router as analysis_router
# from api.routes.generator import router as generator_router
# from api.routes.health import router as health_router
# from api.websocket import router as ws_router
# app.include_router(analysis_router, prefix=settings.API_PREFIX)
# app.include_router(generator_router, prefix=settings.API_PREFIX)
# app.include_router(health_router)
# app.include_router(ws_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
