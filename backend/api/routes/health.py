"""Health check endpoint with GPU and Ollama status."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    """Check system health including GPU and Ollama availability."""
    import httpx

    gpu_available = False
    gpu_name = None
    gpu_memory_mb = None
    ollama_available = False
    ollama_model = None

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:
        pass

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
