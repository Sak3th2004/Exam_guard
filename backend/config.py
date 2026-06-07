"""ExamGuard configuration module.

Centralizes all settings: paths, database, GPU, Ollama, and analysis defaults.
Uses Pydantic BaseSettings for environment variable support.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    # ── Paths ──
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    REPORT_DIR: Path = BASE_DIR / "reports"
    MODEL_CACHE_DIR: Path = BASE_DIR / "models_cache"
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'examguard.db'}"

    # ── API ──
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # ── GPU ──
    USE_GPU: bool = True
    CUDA_DEVICE: str = "cuda:0"

    # ── Ollama LLM ──
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL: str = "mistral:7b-instruct-v0.2-q4_K_M"
    OLLAMA_TIMEOUT: float = 60.0

    # ── Analysis Defaults ──
    DEFAULT_N_STUDENTS: int = 100_000
    DEFAULT_N_QUESTIONS: int = 200
    DEFAULT_N_CENTERS: int = 450
    DEFAULT_N_OPTIONS: int = 4
    LSH_THRESHOLD: float = 0.7
    LSH_NUM_PERM: int = 128
    SIMILARITY_EDGE_THRESHOLD: float = 0.75
    WAA_WEIGHT: float = 0.6
    JACCARD_WEIGHT: float = 0.4
    ISOLATION_FOREST_CONTAMINATION: float = 0.05
    ISOLATION_FOREST_ESTIMATORS: int = 200
    LEAK_GRADIENT_THRESHOLD: float = -0.05
    SPEED_RATIO_THRESHOLD: float = 0.2
    VAE_LATENT_DIM: int = 32
    VAE_EPOCHS: int = 50
    VAE_BATCH_SIZE: int = 512
    VAE_ANOMALY_STD_MULTIPLIER: float = 2.5
    GNN_HIDDEN_DIM: int = 64
    GNN_EPOCHS: int = 100
    GNN_PATIENCE: int = 15
    GNN_LEARNING_RATE: float = 0.01
    GNN_DROPOUT: float = 0.3
    GNN_FRAUD_THRESHOLD: float = 0.5
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_N_ESTIMATORS: int = 200
    XGBOOST_LEARNING_RATE: float = 0.1
    QUESTION_SIMILARITY_THRESHOLD: float = 0.85
    CROSS_EXAM_SIMILARITY_THRESHOLD: float = 0.80
    BONFERRONI_ALPHA: float = 0.05

    # ── Risk Tiers ──
    RISK_CRITICAL_THRESHOLD: float = 0.8
    RISK_HIGH_THRESHOLD: float = 0.6
    RISK_MEDIUM_THRESHOLD: float = 0.3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
