"""SQLAlchemy database models for ExamGuard.

Three tables: analyses, engine_results, flagged_entities.
Uses async SQLite via aiosqlite.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, ForeignKey, Integer, Real, String, Text, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class Analysis(Base):
    """Examination analysis record."""

    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    exam_name = Column(String, nullable=False, default="Exam Analysis")
    exam_type = Column(String, nullable=False, default="paper_based")
    total_students = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    total_centers = Column(Integer, default=0)
    status = Column(String, default="uploading")
    overall_score = Column(Real, nullable=True)
    config = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    report_path = Column(Text, nullable=True)

    # Relationships
    engine_results = relationship("EngineResult", back_populates="analysis", cascade="all, delete-orphan")
    flagged_entities = relationship("FlaggedEntity", back_populates="analysis", cascade="all, delete-orphan")


class EngineResult(Base):
    """Result from a single detection engine."""

    __tablename__ = "engine_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    engine_name = Column(String, nullable=False)
    status = Column(String, default="pending")
    started_at = Column(String, nullable=True)
    completed_at = Column(String, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    result_data = Column(Text, nullable=True)  # JSON
    summary = Column(Text, nullable=True)  # JSON
    flagged_count = Column(Integer, default=0)

    # Relationships
    analysis = relationship("Analysis", back_populates="engine_results")


class FlaggedEntity(Base):
    """A flagged student, center, pair, or question."""

    __tablename__ = "flagged_entities"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    engine_name = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)  # student | center | pair | question
    entity_id = Column(String, nullable=False)
    confidence = Column(Real, default=0.0)
    evidence = Column(Text, nullable=True)  # JSON
    severity = Column(String, default="low")

    # Relationships
    analysis = relationship("Analysis", back_populates="flagged_entities")


# ── Database Engine & Session ──

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Dependency for FastAPI routes."""
    async with async_session() as session:
        yield session
