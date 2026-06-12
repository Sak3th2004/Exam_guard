"""Analysis API routes — CRUD, engine results, visualizations, report download."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from data.ingestion import ingest_csv
from data.schemas import (
    AnalysisResponse, CompareRequest, ComparisonResponse, EngineDetailResponse,
    EngineSummary, EngineStatus, PaginatedFlagged, FlaggedEntityResponse,
    GraphResponse, HeatmapPoint, DifficultyCurveResponse,
    LatentSpacePoint, EnsembleRanking, FeatureImportanceItem,
)
from models.database import Analysis, EngineResult, FlaggedEntity, get_session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyses", tags=["Analysis"])
async def create_analysis(
    file: UploadFile = File(...),
    config: str = Form(default='{}'),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
):
    """Upload CSV + config and start analysis."""
    analysis_id = str(uuid.uuid4())

    # Parse config
    try:
        config_data = json.loads(config)
    except json.JSONDecodeError:
        config_data = {}

    # Read file
    content = await file.read()
    file_path = settings.UPLOAD_DIR / f"{analysis_id}.csv"
    file_path.write_bytes(content)

    # Quick ingestion to get counts
    try:
        ingested = ingest_csv(content, n_options=config_data.get("n_options", 4))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analysis = Analysis(
        id=analysis_id,
        exam_name=config_data.get("exam_name", "Exam Analysis"),
        exam_type=config_data.get("exam_type", "paper_based"),
        total_students=ingested.n_students,
        total_questions=ingested.n_questions,
        total_centers=ingested.n_centers,
        status="processing",
        config=json.dumps(config_data),
        file_path=str(file_path),
    )
    session.add(analysis)
    await session.commit()

    # Run analysis in background
    background_tasks.add_task(_run_analysis_background, analysis_id, content, config_data)

    return {"analysis_id": analysis_id, "status": "processing"}


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse, tags=["Analysis"])
async def get_analysis(analysis_id: str, session: AsyncSession = Depends(get_session)):
    """Get analysis status and summary."""
    analysis = await session.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Get engine summaries
    result = await session.execute(
        select(EngineResult).where(EngineResult.analysis_id == analysis_id)
    )
    engine_results = result.scalars().all()

    engine_summaries = {}
    for er in engine_results:
        summary_data = json.loads(er.summary) if er.summary else {}
        engine_summaries[er.engine_name] = EngineSummary(
            engine_name=er.engine_name,
            status=EngineStatus(er.status) if er.status in [e.value for e in EngineStatus] else EngineStatus.PENDING,
            duration_ms=er.duration_ms,
            flagged_count=er.flagged_count or 0,
            summary_text=json.dumps(summary_data),
        )

    # Count flagged
    flagged_count = await session.execute(
        select(func.count()).select_from(FlaggedEntity).where(
            FlaggedEntity.analysis_id == analysis_id
        )
    )
    total_flagged = flagged_count.scalar() or 0

    return AnalysisResponse(
        id=analysis.id,
        created_at=analysis.created_at or "",
        exam_name=analysis.exam_name or "Exam Analysis",
        exam_type=analysis.exam_type or "paper_based",
        total_students=analysis.total_students or 0,
        total_questions=analysis.total_questions or 0,
        total_centers=analysis.total_centers or 0,
        status=analysis.status or "uploading",
        overall_score=analysis.overall_score,
        engine_summaries=engine_summaries,
        total_flagged=total_flagged,
    )


@router.get("/analyses/{analysis_id}/engines/{engine_name}", tags=["Analysis"])
async def get_engine_detail(
    analysis_id: str, engine_name: str,
    session: AsyncSession = Depends(get_session),
):
    """Get detailed engine results."""
    result = await session.execute(
        select(EngineResult).where(
            EngineResult.analysis_id == analysis_id,
            EngineResult.engine_name == engine_name,
        )
    )
    er = result.scalar_one_or_none()
    if not er:
        raise HTTPException(status_code=404, detail=f"Engine {engine_name} not found")

    return {
        "engine_name": er.engine_name,
        "status": er.status,
        "duration_ms": er.duration_ms,
        "flagged_count": er.flagged_count,
        "result_data": json.loads(er.result_data) if er.result_data else {},
    }


@router.get("/analyses/{analysis_id}/flagged", tags=["Analysis"])
async def get_flagged(
    analysis_id: str,
    severity: Optional[str] = None,
    engine: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Get paginated flagged entities."""
    query = select(FlaggedEntity).where(FlaggedEntity.analysis_id == analysis_id)
    count_query = select(func.count()).select_from(FlaggedEntity).where(
        FlaggedEntity.analysis_id == analysis_id
    )

    if severity:
        query = query.where(FlaggedEntity.severity == severity)
        count_query = count_query.where(FlaggedEntity.severity == severity)
    if engine:
        query = query.where(FlaggedEntity.engine_name == engine)
        count_query = count_query.where(FlaggedEntity.engine_name == engine)

    total = (await session.execute(count_query)).scalar() or 0
    pages = max(1, (total + limit - 1) // limit)

    query = query.offset((page - 1) * limit).limit(limit)
    result = await session.execute(query)
    items = result.scalars().all()

    return {
        "items": [
            {
                "id": fe.id,
                "analysis_id": fe.analysis_id,
                "engine_name": fe.engine_name,
                "entity_type": fe.entity_type,
                "entity_id": fe.entity_id,
                "confidence": fe.confidence,
                "evidence": json.loads(fe.evidence) if fe.evidence else {},
                "severity": fe.severity,
            }
            for fe in items
        ],
        "total": total,
        "page": page,
        "pages": pages,
    }


@router.get("/analyses/{analysis_id}/graph", tags=["Visualization"])
async def get_graph(analysis_id: str, session: AsyncSession = Depends(get_session)):
    """Get network graph data for visualization."""
    # Try GNN enhanced graph first, fall back to copy ring
    for engine_name in ["gnn_copy_ring", "copy_ring"]:
        result = await session.execute(
            select(EngineResult).where(
                EngineResult.analysis_id == analysis_id,
                EngineResult.engine_name == engine_name,
            )
        )
        er = result.scalar_one_or_none()
        if er and er.result_data:
            data = json.loads(er.result_data)
            graph = data.get("enhanced_graph_data") or data.get("graph_data")
            if graph:
                return graph
    return {"nodes": [], "edges": []}


@router.get("/analyses/{analysis_id}/heatmap", tags=["Visualization"])
async def get_heatmap(analysis_id: str, session: AsyncSession = Depends(get_session)):
    """Get geographic heatmap data."""
    result = await session.execute(
        select(EngineResult).where(
            EngineResult.analysis_id == analysis_id,
            EngineResult.engine_name == "center_anomaly",
        )
    )
    er = result.scalar_one_or_none()
    if er and er.result_data:
        data = json.loads(er.result_data)
        return data.get("heatmap_data", [])
    return []


@router.get("/analyses/{analysis_id}/difficulty-curve", tags=["Visualization"])
async def get_difficulty_curve(analysis_id: str, session: AsyncSession = Depends(get_session)):
    """Get difficulty curve chart data."""
    result = await session.execute(
        select(EngineResult).where(
            EngineResult.analysis_id == analysis_id,
            EngineResult.engine_name == "leak_signature",
        )
    )
    er = result.scalar_one_or_none()
    if er and er.result_data:
        data = json.loads(er.result_data)
        return data.get("difficulty_curve_data", {})
    return {"quartiles": [], "normal_accuracy": [], "flagged_accuracy": [], "national_average": []}


@router.get("/analyses/{analysis_id}/latent-space", tags=["Visualization"])
async def get_latent_space(analysis_id: str, session: AsyncSession = Depends(get_session)):
    """Get VAE t-SNE coordinates."""
    result = await session.execute(
        select(EngineResult).where(
            EngineResult.analysis_id == analysis_id,
            EngineResult.engine_name == "vae_anomaly",
        )
    )
    er = result.scalar_one_or_none()
    if er and er.result_data:
        data = json.loads(er.result_data)
        return data.get("latent_visualization", [])
    return []


@router.get("/analyses/{analysis_id}/ensemble-rankings", tags=["Visualization"])
async def get_rankings(
    analysis_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Get XGBoost ensemble fraud rankings."""
    result = await session.execute(
        select(EngineResult).where(
            EngineResult.analysis_id == analysis_id,
            EngineResult.engine_name == "xgboost_ensemble",
        )
    )
    er = result.scalar_one_or_none()
    if er and er.result_data:
        data = json.loads(er.result_data)
        rankings = data.get("final_rankings", [])
        return rankings[:limit]
    return []


@router.get("/analyses/{analysis_id}/feature-importance", tags=["Visualization"])
async def get_feature_importance(analysis_id: str, session: AsyncSession = Depends(get_session)):
    """Get XGBoost feature importance."""
    result = await session.execute(
        select(EngineResult).where(
            EngineResult.analysis_id == analysis_id,
            EngineResult.engine_name == "xgboost_ensemble",
        )
    )
    er = result.scalar_one_or_none()
    if er and er.result_data:
        data = json.loads(er.result_data)
        return data.get("feature_importance", [])
    return []


@router.post("/analyses/{analysis_id}/compare", tags=["Analysis"])
async def compare_students(
    analysis_id: str,
    request: CompareRequest,
    session: AsyncSession = Depends(get_session),
):
    """Side-by-side comparison of two students."""
    from services.comparison import compare_students as do_compare

    analysis = await session.get(Analysis, analysis_id)
    if not analysis or not analysis.file_path:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        import pathlib
        csv_content = pathlib.Path(analysis.file_path).read_bytes()
        result = do_compare(csv_content, request.student_a, request.student_b)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analyses/{analysis_id}/report", tags=["Analysis"])
async def download_report(analysis_id: str, session: AsyncSession = Depends(get_session)):
    """Download PDF forensic report."""
    analysis = await session.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    report_path = settings.REPORT_DIR / f"{analysis_id}.pdf"
    if not report_path.exists():
        # Generate report on-demand
        from services.report_generator import generate_report

        engine_data = {}
        result = await session.execute(
            select(EngineResult).where(EngineResult.analysis_id == analysis_id)
        )
        for er in result.scalars().all():
            rd = json.loads(er.result_data) if er.result_data else {}
            # Also get flagged student IDs from FlaggedEntity table
            flagged_q = await session.execute(
                select(FlaggedEntity.entity_id).where(
                    FlaggedEntity.analysis_id == analysis_id,
                    FlaggedEntity.engine_name == er.engine_name,
                )
            )
            fids = [r[0] for r in flagged_q.all()]
            engine_data[er.engine_name] = {
                "result_data": rd,
                "flagged_student_ids": fids,
                "flagged_count": er.flagged_count or len(fids),
                "status": er.status or "complete",
                "duration_ms": er.duration_ms,
                "summary": json.loads(er.summary) if er.summary else {},
            }

        await generate_report(
            analysis_id=analysis_id,
            exam_name=analysis.exam_name or "Exam",
            total_students=analysis.total_students or 0,
            total_centers=analysis.total_centers or 0,
            overall_score=analysis.overall_score or 50,
            engine_data=engine_data,
            output_path=report_path,
        )

    return FileResponse(
        path=str(report_path),
        filename=f"ExamGuard_Report_{analysis_id[:8]}.pdf",
        media_type="application/pdf",
    )


async def _run_analysis_background(analysis_id: str, csv_content: bytes, config: dict):
    """Background task to run analysis on uploaded CSV."""
    from engines.orchestrator import AnalysisOrchestrator
    from api.websocket import manager
    from models.database import async_session

    try:
        ingested = ingest_csv(csv_content, n_options=config.get("n_options", 4))

        orchestrator = AnalysisOrchestrator()

        async def ws_broadcast(engine, progress, message, status):
            await manager.broadcast(analysis_id, engine, progress, message, status)

        results = await orchestrator.run_analysis(
            answers=ingested.answers,
            answer_key=ingested.answer_key,
            student_ids=ingested.student_ids,
            center_ids=ingested.center_ids,
            timing_data=ingested.timing_data,
            question_texts=ingested.question_texts,
            ws_broadcast=ws_broadcast,
            analysis_id=analysis_id,
        )

        # Save to DB
        async with async_session() as session:
            meta = results.pop("_meta", None)
            overall_score = meta.result_data.get("integrity_score", 50) if meta else 50

            analysis = await session.get(Analysis, analysis_id)
            if analysis:
                analysis.status = "complete"
                analysis.overall_score = overall_score

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

                    for sid in result.flagged_student_ids[:1000]:
                        fe = FlaggedEntity(
                            analysis_id=analysis_id,
                            engine_name=engine_name,
                            entity_type="student",
                            entity_id=sid,
                            confidence=0.8,
                            severity="high",
                        )
                        session.add(fe)

                await session.commit()

        await manager.broadcast(analysis_id, "complete", 100, "Analysis complete", "complete")

    except Exception as e:
        logger.error(f"[{analysis_id}] Failed: {e}", exc_info=True)
        async with async_session() as session:
            analysis = await session.get(Analysis, analysis_id)
            if analysis:
                analysis.status = "failed"
                await session.commit()
