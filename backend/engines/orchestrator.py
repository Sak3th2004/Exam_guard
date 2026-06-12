"""Analysis Orchestrator — Parallel Engine Execution.

Manages execution of all 8 detection engines + XGBoost ensemble:
  - CPU engines run in parallel (asyncio)
  - GPU engines run sequentially (shared VRAM)
  - Progress streamed via WebSocket callbacks
  - Graceful degradation if any engine fails
  - Computes overall integrity score from all results

Architecture principles:
  1. Separation: Detection (Layers 1-3) is math/ML. Narration (Layer 4) is LLM.
  2. Graceful degradation: If GPU fails, classical engines still work.
  3. Parallel execution: CPU engines concurrent, GPU sequential.
  4. Real-time streaming: Per-engine progress via WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from engines.base_engine import EngineOutput, ProgressCallback
from engines.copy_ring import CopyRingEngine
from engines.stat_impossibility import StatImpossibilityEngine
from engines.center_anomaly import CenterAnomalyEngine
from engines.leak_signature import LeakSignatureEngine
from engines.response_time import ResponseTimeEngine
from engines.gnn_fraud import GNNFraudEngine
from engines.vae_anomaly import VAEAnomalyEngine
from engines.question_similarity import QuestionSimilarityEngine
from engines.xgboost_ensemble import XGBoostEnsembleEngine
from engines.benford import BenfordEngine

logger = logging.getLogger(__name__)

# Type for WebSocket broadcast function
WSBroadcast = Callable[[str, int, str, str], Any]  # engine, progress, message, status


class AnalysisOrchestrator:
    """Orchestrates parallel execution of all detection engines."""

    def __init__(self):
        # Layer 1: Classical (CPU)
        self.e1_copy_ring = CopyRingEngine()
        self.e2_stat_impossibility = StatImpossibilityEngine()
        self.e3_center_anomaly = CenterAnomalyEngine()
        self.e4_leak_signature = LeakSignatureEngine()
        self.e5_response_time = ResponseTimeEngine()

        # Layer 2: Deep Learning (GPU)
        self.e6_gnn = GNNFraudEngine()
        self.e7_vae = VAEAnomalyEngine()
        self.e8_question_sim = QuestionSimilarityEngine()

        # Layer 3: Ensemble (GPU)
        self.xgboost_ensemble = XGBoostEnsembleEngine()

        # Bonus: Forensic
        self.e9_benford = BenfordEngine()

    async def run_analysis(
        self,
        answers: np.ndarray,
        answer_key: Optional[np.ndarray] = None,
        student_ids: Optional[list[str]] = None,
        center_ids: Optional[list[str]] = None,
        timing_data: Optional[np.ndarray] = None,
        question_texts: Optional[list[str]] = None,
        ground_truth: Optional[dict[str, Any]] = None,
        fraud_labels: Optional[np.ndarray] = None,
        center_metadata: Optional[list[dict]] = None,
        ws_broadcast: Optional[WSBroadcast] = None,
        db_session: Optional[AsyncSession] = None,
        analysis_id: Optional[str] = None,
    ) -> dict[str, EngineOutput]:
        """Run all 8 engines + ensemble, streaming progress.

        Returns dict mapping engine_name → EngineOutput.
        """
        start_time = time.time()
        results: dict[str, EngineOutput] = {}

        def make_progress_cb(engine_name: str) -> ProgressCallback:
            def callback(pct: int, msg: str) -> None:
                if ws_broadcast:
                    asyncio.ensure_future(
                        _safe_broadcast(ws_broadcast, engine_name, pct, msg, "running")
                    )
            return callback

        common_kwargs = {
            "answers": answers,
            "answer_key": answer_key,
            "student_ids": student_ids,
            "center_ids": center_ids,
            "timing_data": timing_data,
            "question_texts": question_texts,
            "ground_truth": ground_truth,
        }

        # ═══════════════════════════════════════
        # LAYER 1: Classical Engines (parallel)
        # ═══════════════════════════════════════
        logger.info("Starting Layer 1: Classical Detection Engines")

        layer1_tasks = [
            self._run_engine(self.e1_copy_ring, common_kwargs, {
                "center_metadata": center_metadata,
            }, make_progress_cb("copy_ring")),
            self._run_engine(self.e3_center_anomaly, common_kwargs, {
                "center_metadata": center_metadata or [],
            }, make_progress_cb("center_anomaly")),
            self._run_engine(self.e4_leak_signature, common_kwargs, {},
                           make_progress_cb("leak_signature")),
            self._run_engine(self.e5_response_time, common_kwargs, {},
                           make_progress_cb("response_time")),
            self._run_engine(self.e9_benford, common_kwargs, {},
                           make_progress_cb("benford_law")),
        ]

        layer1_results = await asyncio.gather(*layer1_tasks, return_exceptions=True)

        # Process results
        for engine_name, result in zip(
            ["copy_ring", "center_anomaly", "leak_signature", "response_time", "benford_law"],
            layer1_results,
        ):
            if isinstance(result, Exception):
                results[engine_name] = EngineOutput(
                    engine_name=engine_name, status="failed",
                    error_message=str(result),
                )
            else:
                results[engine_name] = result

        # E2 depends on E1 results (uses candidate pairs)
        e1_result = results.get("copy_ring", EngineOutput(engine_name="copy_ring"))
        e2_result = await self._run_engine(
            self.e2_stat_impossibility, common_kwargs, {
                "copy_ring_edges": e1_result.result_data.get("graph_data", {}).get("edges", []),
            }, make_progress_cb("stat_impossibility"),
        )
        results["stat_impossibility"] = e2_result

        logger.info("Layer 1 complete. Starting Layer 2: Deep Learning")

        # ═══════════════════════════════════════
        # LAYER 2: Deep Learning (sequential — shared GPU)
        # ═══════════════════════════════════════

        # E6: GNN (depends on E1 graph)
        e6_result = await self._run_engine(
            self.e6_gnn, common_kwargs, {
                "copy_ring_result": {
                    "graph_data": e1_result.result_data.get("graph_data", {"nodes": [], "edges": []}),
                    "clusters": e1_result.result_data.get("clusters", []),
                },
            }, make_progress_cb("gnn_copy_ring"),
        )
        results["gnn_copy_ring"] = e6_result

        # E7: VAE
        e7_result = await self._run_engine(
            self.e7_vae, common_kwargs, {
                "copy_ring_flagged": e1_result.flagged_student_ids,
                "leak_flagged": results.get("leak_signature", EngineOutput(engine_name="leak_signature")).flagged_student_ids,
            }, make_progress_cb("vae_anomaly"),
        )
        results["vae_anomaly"] = e7_result

        # E8: Question Similarity
        e8_result = await self._run_engine(
            self.e8_question_sim, common_kwargs, {},
            make_progress_cb("question_similarity"),
        )
        results["question_similarity"] = e8_result

        logger.info("Layer 2 complete. Starting Layer 3: XGBoost Ensemble")

        # ═══════════════════════════════════════
        # LAYER 3: XGBoost Ensemble
        # ═══════════════════════════════════════

        ensemble_result = await self._run_engine(
            self.xgboost_ensemble, common_kwargs, {
                "engine_results": {
                    name: {"flagged_student_ids": r.flagged_student_ids, "result_data": r.result_data}
                    for name, r in results.items()
                },
                "fraud_labels": fraud_labels,
            }, make_progress_cb("xgboost_ensemble"),
        )
        results["xgboost_ensemble"] = ensemble_result

        # ═══════════════════════════════════════
        # Compute Overall Integrity Score
        # ═══════════════════════════════════════
        total_flagged = len(set().union(*(
            set(r.flagged_student_ids) for r in results.values()
        )))
        n_students = answers.shape[0]
        fraud_rate = total_flagged / max(n_students, 1)

        # Integrity score: 100 = clean, 0 = heavily compromised
        if fraud_rate < 0.001:
            integrity_score = 95 + 5 * (1 - fraud_rate / 0.001)
        elif fraud_rate < 0.005:
            integrity_score = 85 + 10 * (1 - (fraud_rate - 0.001) / 0.004)
        elif fraud_rate < 0.01:
            integrity_score = 70 + 15 * (1 - (fraud_rate - 0.005) / 0.005)
        elif fraud_rate < 0.05:
            integrity_score = 40 + 30 * (1 - (fraud_rate - 0.01) / 0.04)
        else:
            integrity_score = max(5, 40 * (1 - fraud_rate))

        results["_meta"] = EngineOutput(
            engine_name="_meta",
            result_data={
                "integrity_score": round(integrity_score, 1),
                "total_flagged": total_flagged,
                "fraud_rate": round(fraud_rate * 100, 2),
                "total_time_ms": int((time.time() - start_time) * 1000),
                "engines_completed": sum(1 for r in results.values() if r.status == "complete"),
                "engines_failed": sum(1 for r in results.values() if r.status == "failed"),
                "engines_skipped": sum(1 for r in results.values() if r.status == "skipped"),
                "cross_referenced": _count_cross_referenced(results, student_ids or []),
            },
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Analysis complete in {elapsed:.1f}s — "
            f"integrity={integrity_score:.1f}, flagged={total_flagged:,}"
        )

        return results

    async def _run_engine(
        self,
        engine,
        common_kwargs: dict,
        extra_kwargs: dict,
        progress_cb: ProgressCallback,
    ) -> EngineOutput:
        """Run a single engine with error handling."""
        try:
            return await engine.analyze(
                **common_kwargs,
                **extra_kwargs,
                progress_callback=progress_cb,
            )
        except Exception as e:
            logger.error(f"Engine {engine.engine_name} failed: {e}", exc_info=True)
            return EngineOutput(
                engine_name=engine.engine_name,
                status="failed",
                error_message=str(e),
            )


def _count_cross_referenced(
    results: dict[str, EngineOutput], student_ids: list[str]
) -> int:
    """Count students flagged by 2+ engines."""
    student_flags: dict[str, int] = {}
    for name, result in results.items():
        if name.startswith("_"):
            continue
        for sid in result.flagged_student_ids:
            student_flags[sid] = student_flags.get(sid, 0) + 1
    return sum(1 for count in student_flags.values() if count >= 2)


async def _safe_broadcast(ws_broadcast: WSBroadcast, engine: str, pct: int, msg: str, status: str):
    """Safely broadcast WebSocket message."""
    try:
        await ws_broadcast(engine, pct, msg, status)
    except Exception:
        pass
