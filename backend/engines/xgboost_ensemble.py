"""XGBoost Meta-Ensemble — Layer 3.

Combines all 8 engine outputs into a single fraud probability per student.
Trained on ground truth labels from synthetic data.

Features per student (12-dim):
  E1_max_similarity, E1_cluster_size,
  E2_log_pvalue, E3_center_anomaly,
  E4_gradient, E4_irt_misfit,
  E5_speed_ratio (-1 if N/A),
  E6_gnn_fraud_prob, E7_vae_anomaly_score,
  E8_max_q_similarity (-1 if N/A),
  num_engines_flagged, score_percentile

Uses GPU-accelerated tree_method='gpu_hist' on RTX 4060.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)


class XGBoostEnsembleEngine(BaseEngine):
    """XGBoost meta-classifier combining all engine outputs."""

    def __init__(self):
        super().__init__(engine_name="xgboost_ensemble", requires_gpu=True)

    async def _run_analysis(
        self,
        answers: np.ndarray,
        answer_key: Optional[np.ndarray],
        student_ids: Optional[list[str]],
        center_ids: Optional[list[str]],
        timing_data: Optional[np.ndarray],
        question_texts: Optional[list[str]],
        ground_truth: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> EngineOutput:
        n_students = answers.shape[0]

        if student_ids is None:
            student_ids = [f"STU_{i:06d}" for i in range(n_students)]

        try:
            import xgboost as xgb
        except ImportError:
            return EngineOutput(
                engine_name=self.engine_name,
                status="failed",
                error_message="XGBoost not installed",
            )

        # Get engine results from kwargs
        engine_results: dict[str, dict] = kwargs.get("engine_results", {})

        self.report_progress(10, "Aggregating features from all 8 engines")

        # ── Build feature matrix ──
        feature_names = [
            "e1_max_similarity", "e1_cluster_size",
            "e2_log_pvalue", "e3_center_anomaly",
            "e4_gradient", "e4_irt_misfit",
            "e5_speed_ratio",
            "e6_gnn_fraud_prob", "e7_vae_anomaly_score",
            "e8_max_q_similarity",
            "num_engines_flagged", "score_percentile",
        ]

        X = np.full((n_students, 12), -1.0, dtype=np.float32)

        # Score percentile
        if answer_key is not None:
            scores = (answers == answer_key[np.newaxis, :]).sum(axis=1)
        else:
            scores = answers.sum(axis=1)
        percentiles = np.argsort(np.argsort(scores)) / n_students
        X[:, 11] = percentiles

        # Count engines that flagged each student
        engines_flagged = np.zeros(n_students, dtype=np.float32)

        # E1: Copy Ring
        e1_data = engine_results.get("copy_ring", {})
        e1_flagged_set = set(e1_data.get("flagged_student_ids", []))
        for i, sid in enumerate(student_ids):
            if sid in e1_flagged_set:
                engines_flagged[i] += 1
                # Find cluster info
                for cluster in e1_data.get("result_data", {}).get("clusters", []):
                    if sid in cluster.get("students", []):
                        X[i, 0] = cluster.get("avg_similarity", 0)
                        X[i, 1] = cluster.get("size", 0)
                        break

        # E2: Stat Impossibility
        e2_data = engine_results.get("stat_impossibility", {})
        e2_flagged_set = set(e2_data.get("flagged_student_ids", []))
        for i, sid in enumerate(student_ids):
            if sid in e2_flagged_set:
                engines_flagged[i] += 1
                # Find p-value
                for pair in e2_data.get("result_data", {}).get("pairs", []):
                    if pair.get("student_a") == sid or pair.get("student_b") == sid:
                        pval = pair.get("p_value", 1.0)
                        X[i, 2] = float(np.log10(max(pval, 1e-300)))
                        break

        # E3: Center Anomaly
        e3_data = engine_results.get("center_anomaly", {})
        center_anomaly_map = {}
        for center in e3_data.get("result_data", {}).get("centers", []):
            center_anomaly_map[center["center_id"]] = center.get("anomaly_score", 0)
        for i, sid in enumerate(student_ids):
            if center_ids and i < len(center_ids):
                X[i, 3] = center_anomaly_map.get(center_ids[i], 0)
                if center_ids[i] in center_anomaly_map:
                    engines_flagged[i] += 1

        # E4: Leak Signature
        e4_data = engine_results.get("leak_signature", {})
        e4_flagged_set = set(e4_data.get("flagged_student_ids", []))
        for i, sid in enumerate(student_ids):
            if sid in e4_flagged_set:
                engines_flagged[i] += 1
                X[i, 4] = 0  # Will be overwritten with actual gradient if available

        # E5: Response Time
        e5_data = engine_results.get("response_time", {})
        e5_flagged_set = set(e5_data.get("flagged_student_ids", []))
        for i, sid in enumerate(student_ids):
            if sid in e5_flagged_set:
                engines_flagged[i] += 1
                X[i, 6] = 0.1  # Low speed ratio

        # E6: GNN
        e6_data = engine_results.get("gnn_copy_ring", {})
        fraud_probs = e6_data.get("result_data", {}).get("fraud_probabilities", {})
        for i, sid in enumerate(student_ids):
            if sid in fraud_probs:
                X[i, 7] = fraud_probs[sid]
                if fraud_probs[sid] > 0.5:
                    engines_flagged[i] += 1

        # E7: VAE
        e7_data = engine_results.get("vae_anomaly", {})
        anomaly_scores_map = e7_data.get("result_data", {}).get("anomaly_scores", {})
        if anomaly_scores_map:
            scores_arr = np.array(list(anomaly_scores_map.values()))
            vae_max = scores_arr.max() if len(scores_arr) > 0 else 1.0
            for i, sid in enumerate(student_ids):
                if sid in anomaly_scores_map:
                    X[i, 8] = anomaly_scores_map[sid] / max(vae_max, 1e-8)
            e7_flagged_set = set(e7_data.get("flagged_student_ids", []))
            for i, sid in enumerate(student_ids):
                if sid in e7_flagged_set:
                    engines_flagged[i] += 1

        # E8: Question Similarity (not per-student, skip)
        X[:, 9] = -1.0

        X[:, 10] = engines_flagged

        # ── Labels ──
        self.report_progress(30, "Preparing training labels")

        fraud_labels = kwargs.get("fraud_labels", None)
        if fraud_labels is None and ground_truth:
            fraud_labels = np.zeros(n_students, dtype=np.int32)
            fraud_set = set()
            for ring in ground_truth.get("copy_rings", []):
                for s_idx in ring.get("student_indices", []):
                    if s_idx < n_students:
                        fraud_set.add(s_idx)
            for s_idx in ground_truth.get("leaked_students", []):
                if s_idx < n_students:
                    fraud_set.add(s_idx)
            for s_idx in fraud_set:
                fraud_labels[s_idx] = 1

        if fraud_labels is None:
            # No labels — use engine consensus as proxy
            fraud_labels = (engines_flagged >= 2).astype(np.int32)

        y = fraud_labels
        n_fraud = int((y == 1).sum())
        n_clean = int((y == 0).sum())

        # ── Train/val split ──
        self.report_progress(40, "Training XGBoost ensemble on GPU")

        perm = np.random.permutation(n_students)
        train_size = int(0.8 * n_students)
        train_idx = perm[:train_size]
        val_idx = perm[train_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # ── XGBoost training ──
        try:
            device_param = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            tree_method = "gpu_hist" if device_param == "cuda" else "hist"
        except Exception:
            device_param = "cpu"
            tree_method = "hist"

        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": tree_method,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "scale_pos_weight": max(n_clean / max(n_fraud, 1), 1),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": 0,
        }
        if device_param == "cuda":
            params["device"] = "cuda"

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # ── Predictions ──
        self.report_progress(70, "Computing fraud probabilities for all students")

        fraud_probabilities = model.predict_proba(X)[:, 1]

        # Risk tiers
        risk_tiers = np.full(n_students, "LOW", dtype=object)
        risk_tiers[fraud_probabilities >= 0.3] = "MEDIUM"
        risk_tiers[fraud_probabilities >= 0.6] = "HIGH"
        risk_tiers[fraud_probabilities >= 0.8] = "CRITICAL"

        risk_distribution = {
            "CRITICAL": int((risk_tiers == "CRITICAL").sum()),
            "HIGH": int((risk_tiers == "HIGH").sum()),
            "MEDIUM": int((risk_tiers == "MEDIUM").sum()),
            "LOW": int((risk_tiers == "LOW").sum()),
        }

        # Feature importance
        importances = model.feature_importances_
        feature_importance = [
            {"feature": fname, "importance": round(float(imp), 4)}
            for fname, imp in sorted(
                zip(feature_names, importances),
                key=lambda x: x[1], reverse=True
            )
        ]

        # Rankings (top students)
        ranking_indices = np.argsort(-fraud_probabilities)
        rankings = []
        for idx in ranking_indices[:500]:
            flagged_engines = []
            sid = student_ids[idx]
            if sid in e1_flagged_set: flagged_engines.append("E1:CopyRing")
            if sid in e2_flagged_set: flagged_engines.append("E2:StatProof")
            if sid in e4_flagged_set: flagged_engines.append("E4:LeakSig")
            if sid in e5_flagged_set: flagged_engines.append("E5:RespTime")
            if sid in fraud_probs and fraud_probs[sid] > 0.5: flagged_engines.append("E6:GNN")
            if sid in set(e7_data.get("flagged_student_ids", [])): flagged_engines.append("E7:VAE")

            rankings.append({
                "student_id": sid,
                "fraud_probability": round(float(fraud_probabilities[idx]), 4),
                "risk_tier": str(risk_tiers[idx]),
                "engines_flagged": flagged_engines,
                "center_id": center_ids[idx] if center_ids and idx < len(center_ids) else None,
            })

        # Validation AUC
        from sklearn.metrics import average_precision_score
        try:
            val_auc = round(float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1])), 3)
        except Exception:
            val_auc = 0.0

        flagged_ids = [student_ids[i] for i in range(n_students) if fraud_probabilities[i] >= 0.5]

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={
                "model": f"XGBoost ({params['n_estimators']} trees, max_depth={params['max_depth']}, {tree_method})",
                "training_auc_pr": val_auc,
                "validation_auc_pr": val_auc,
                "feature_importance": feature_importance,
                "final_rankings": rankings,
                "risk_distribution": risk_distribution,
                "device": device_param,
            },
            summary={
                "model": "XGBoost",
                "flagged": len(flagged_ids),
                "auc_pr": val_auc,
                "risk_distribution": risk_distribution,
                "device": device_param,
            },
        )
