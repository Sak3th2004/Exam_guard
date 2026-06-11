"""ExamGuard Detection Engines Package.

Exports all 8 detection engines + XGBoost ensemble + orchestrator.
Each engine implements the BaseEngine interface with async _run_analysis().
"""

from engines.base_engine import BaseEngine
from engines.copy_ring import CopyRingEngine
from engines.stat_impossibility import StatImpossibilityEngine
from engines.center_anomaly import CenterAnomalyEngine
from engines.leak_signature import LeakSignatureEngine
from engines.response_time import ResponseTimeEngine
from engines.gnn_fraud import GNNFraudEngine
from engines.vae_anomaly import VAEAnomalyEngine
from engines.question_similarity import QuestionSimilarityEngine
from engines.xgboost_ensemble import XGBoostEnsembleEngine
from engines.orchestrator import AnalysisOrchestrator

# Engine registry for dynamic instantiation
ENGINE_REGISTRY: dict[str, type[BaseEngine]] = {
    "copy_ring": CopyRingEngine,
    "stat_impossibility": StatImpossibilityEngine,
    "center_anomaly": CenterAnomalyEngine,
    "leak_signature": LeakSignatureEngine,
    "response_time": ResponseTimeEngine,
    "gnn_copy_ring": GNNFraudEngine,
    "vae_anomaly": VAEAnomalyEngine,
    "question_similarity": QuestionSimilarityEngine,
    "xgboost_ensemble": XGBoostEnsembleEngine,
}

__all__ = [
    "BaseEngine",
    "CopyRingEngine",
    "StatImpossibilityEngine",
    "CenterAnomalyEngine",
    "LeakSignatureEngine",
    "ResponseTimeEngine",
    "GNNFraudEngine",
    "VAEAnomalyEngine",
    "QuestionSimilarityEngine",
    "XGBoostEnsembleEngine",
    "AnalysisOrchestrator",
    "ENGINE_REGISTRY",
]
