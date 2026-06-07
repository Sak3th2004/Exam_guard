"""Pydantic schemas for ExamGuard API.

Defines request/response models for all endpoints, engine results,
and internal data structures.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──

class ExamType(str, Enum):
    """Examination type."""
    PAPER_BASED = "paper_based"
    COMPUTER_BASED = "computer_based"


class AnalysisStatus(str, Enum):
    """Analysis processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class EngineStatus(str, Enum):
    """Individual engine status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class EntityType(str, Enum):
    """Type of flagged entity."""
    STUDENT = "student"
    CENTER = "center"
    PAIR = "pair"
    QUESTION = "question"


class Severity(str, Enum):
    """Fraud severity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskTier(str, Enum):
    """XGBoost risk tier classification."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ── Request Models ──

class ExamConfig(BaseModel):
    """Configuration for exam analysis."""
    exam_name: str = Field(default="Exam Analysis", description="Name of the examination")
    exam_type: ExamType = Field(default=ExamType.PAPER_BASED, description="Type of examination")
    n_options: int = Field(default=4, ge=2, le=6, description="Number of answer options per question")
    has_answer_key: bool = Field(default=False, description="Whether answer key is provided")
    has_timing_data: bool = Field(default=False, description="Whether response timing data is available")
    has_question_text: bool = Field(default=False, description="Whether question text is available")


class GenerateRequest(BaseModel):
    """Request to generate synthetic exam data."""
    n_students: int = Field(default=100_000, ge=100, le=500_000)
    n_questions: int = Field(default=200, ge=10, le=500)
    n_centers: int = Field(default=450, ge=10, le=2000)
    n_options: int = Field(default=4, ge=2, le=6)
    include_timing: bool = Field(default=True)
    include_question_text: bool = Field(default=True)
    exam_name: str = Field(default="NEET 2026 Forensic Simulation")
    fraud_config: Optional[FraudConfig] = None


class FraudConfig(BaseModel):
    """Configuration for planted fraud patterns in synthetic data."""
    copy_rings: list[CopyRingConfig] = Field(default_factory=list)
    leak_students: int = Field(default=340)
    leak_question_range: tuple[int, int] = Field(default=(45, 120))
    anomalous_centers: int = Field(default=3)
    timing_cheaters: int = Field(default=89)


class CopyRingConfig(BaseModel):
    """Configuration for a single copy ring."""
    size: int = Field(default=20, ge=3)
    overlap_rate: float = Field(default=0.85, ge=0.5, le=1.0)
    center_id: Optional[str] = None


class CompareRequest(BaseModel):
    """Request to compare two students."""
    student_a: str
    student_b: str


# ── Response Models ──

class AnalysisResponse(BaseModel):
    """Response for analysis status."""
    id: str
    created_at: str
    exam_name: str
    exam_type: ExamType
    total_students: int
    total_questions: int
    total_centers: int
    status: AnalysisStatus
    overall_score: Optional[float] = None
    engine_summaries: dict[str, EngineSummary] = Field(default_factory=dict)
    total_flagged: int = 0
    cross_referenced_count: int = 0


class EngineSummary(BaseModel):
    """Summary of an engine's results for the dashboard."""
    engine_name: str
    status: EngineStatus
    duration_ms: Optional[int] = None
    flagged_count: int = 0
    summary_text: str = ""
    confidence: Optional[float] = None


class EngineDetailResponse(BaseModel):
    """Full engine result details."""
    engine_name: str
    status: EngineStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    flagged_count: int = 0
    result_data: dict[str, Any] = Field(default_factory=dict)


class FlaggedEntityResponse(BaseModel):
    """Response for a flagged entity."""
    id: str
    analysis_id: str
    engine_name: str
    entity_type: EntityType
    entity_id: str
    confidence: float
    evidence: dict[str, Any] = Field(default_factory=dict)
    severity: Severity


class PaginatedFlagged(BaseModel):
    """Paginated list of flagged entities."""
    items: list[FlaggedEntityResponse]
    total: int
    page: int
    pages: int


class GraphNode(BaseModel):
    """Node in the network graph."""
    id: str
    label: str
    cluster: Optional[int] = None
    fraud_prob: Optional[float] = None
    center_id: Optional[str] = None
    score: Optional[int] = None
    size: float = 5.0
    color: str = "#8888aa"


class GraphEdge(BaseModel):
    """Edge in the network graph."""
    source: str
    target: str
    weight: float
    color: str = "#2a2a3e"


class GraphResponse(BaseModel):
    """Network graph data for visualization."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class HeatmapPoint(BaseModel):
    """Geographic data point for map visualization."""
    center_id: str
    lat: float
    lon: float
    anomaly_score: float
    student_count: int
    city: str
    state: str
    flags: list[str] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)


class DifficultyCurveResponse(BaseModel):
    """Difficulty curve chart data."""
    quartiles: list[str]
    normal_accuracy: list[float]
    flagged_accuracy: list[float]
    national_average: list[float]


class LatentSpacePoint(BaseModel):
    """A point in the VAE latent space t-SNE plot."""
    student_id: str
    x: float
    y: float
    anomaly_score: float
    is_flagged: bool


class EnsembleRanking(BaseModel):
    """Student ranking from XGBoost ensemble."""
    student_id: str
    fraud_probability: float
    risk_tier: RiskTier
    engines_flagged: list[str] = Field(default_factory=list)
    center_id: Optional[str] = None


class FeatureImportanceItem(BaseModel):
    """Feature importance from XGBoost."""
    feature: str
    importance: float


class ComparisonResponse(BaseModel):
    """Side-by-side student comparison."""
    student_a: str
    student_b: str
    total_questions: int
    matching_total: int
    matching_wrong: int
    jaccard: float
    waa: float
    p_value: Optional[float] = None
    human_readable: str = ""
    per_question: list[QuestionComparison] = Field(default_factory=list)


class QuestionComparison(BaseModel):
    """Per-question comparison between two students."""
    question: int
    answer_a: str
    answer_b: str
    correct_answer: Optional[str] = None
    is_match: bool
    is_both_wrong: bool
    is_both_correct: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    ollama_available: bool = False
    ollama_model: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket progress message."""
    engine: str
    progress: int = Field(ge=0, le=100)
    message: str = ""
    status: EngineStatus = EngineStatus.RUNNING


class GenerateResponse(BaseModel):
    """Response from data generation."""
    analysis_id: str
    message: str
    total_students: int
    total_questions: int
    total_centers: int


# Fix forward references
GenerateRequest.model_rebuild()
