"""Generator endpoint — creates synthetic exam data and auto-starts analysis."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from data.generator import generate_exam_data, data_to_csv_bytes
from data.schemas import GenerateRequest, GenerateResponse
from models.database import Analysis, EngineResult, FlaggedEntity, get_session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, tags=["Generator"])
async def generate_data(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """Generate synthetic exam data with planted fraud patterns and start analysis."""
    analysis_id = str(uuid.uuid4())

    # Create analysis record
    analysis = Analysis(
        id=analysis_id,
        exam_name=request.exam_name,
        exam_type="computer_based" if request.include_timing else "paper_based",
        total_students=request.n_students,
        total_questions=request.n_questions,
        total_centers=request.n_centers,
        status="processing",
        config=json.dumps(request.model_dump()),
    )
    session.add(analysis)
    await session.commit()

    # Run generation and analysis in background
    background_tasks.add_task(
        _generate_and_analyze,
        analysis_id,
        request,
    )

    return GenerateResponse(
        analysis_id=analysis_id,
        message=f"Generating {request.n_students:,} students with fraud patterns, then analyzing...",
        total_students=request.n_students,
        total_questions=request.n_questions,
        total_centers=request.n_centers,
    )


async def _generate_and_analyze(analysis_id: str, request: GenerateRequest):
    """Background task: generate data then run full analysis."""
    from engines.orchestrator import AnalysisOrchestrator
    from api.websocket import manager
    from models.database import async_session

    try:
        async with async_session() as session:
            # Generate data
            logger.info(f"[{analysis_id}] Generating synthetic data...")
            data = generate_exam_data(
                n_students=request.n_students,
                n_questions=request.n_questions,
                n_centers=request.n_centers,
                n_options=request.n_options,
                include_timing=request.include_timing,
                include_question_text=request.include_question_text,
            )

            # Save CSV
            csv_bytes = data_to_csv_bytes(data)
            csv_path = settings.UPLOAD_DIR / f"{analysis_id}.csv"
            csv_path.write_bytes(csv_bytes)

            # Run analysis
            orchestrator = AnalysisOrchestrator()

            async def ws_broadcast(engine, progress, message, status):
                await manager.broadcast(analysis_id, engine, progress, message, status)

            results = await orchestrator.run_analysis(
                answers=data.answers,
                answer_key=data.answer_key,
                student_ids=data.student_ids,
                center_ids=[data.student_centers[sid] for sid in data.student_ids],
                timing_data=data.timing_data,
                question_texts=data.question_texts,
                ground_truth=data.ground_truth,
                fraud_labels=data.fraud_labels,
                center_metadata=data.center_metadata,
                ws_broadcast=ws_broadcast,
                analysis_id=analysis_id,
            )

            # Save results to database
            meta = results.pop("_meta", None)
            overall_score = meta.result_data.get("integrity_score", 50) if meta else 50

            # Update analysis
            analysis = await session.get(Analysis, analysis_id)
            if analysis:
                analysis.status = "complete"
                analysis.overall_score = overall_score
                analysis.file_path = str(csv_path)

                # Save engine results
                for engine_name, result in results.items():
                    er = EngineResult(
                        analysis_id=analysis_id,
                        engine_name=engine_name,
                        status=result.status,
                        duration_ms=result.duration_ms,
                        result_data=json.dumps(result.result_data, default=str),
                        summary=json.dumps(result.summary, default=str),
                        flagged_count=result.flagged_count,
                    )
                    session.add(er)

                    # Save flagged entities
                    for sid in result.flagged_student_ids[:1000]:  # Cap per engine
                        fe = FlaggedEntity(
                            analysis_id=analysis_id,
                            engine_name=engine_name,
                            entity_type="student",
                            entity_id=sid,
                            confidence=0.8,
                            severity="high" if result.flagged_count > 100 else "medium",
                        )
                        session.add(fe)

                await session.commit()

            # Broadcast completion
            await manager.broadcast(analysis_id, "complete", 100, "Analysis complete", "complete")

            logger.info(f"[{analysis_id}] Analysis complete — score={overall_score}")

    except Exception as e:
        logger.error(f"[{analysis_id}] Analysis failed: {e}", exc_info=True)
        async with async_session() as session:
            analysis = await session.get(Analysis, analysis_id)
            if analysis:
                analysis.status = "failed"
                await session.commit()
        await manager.broadcast(analysis_id, "error", 0, str(e), "failed")
