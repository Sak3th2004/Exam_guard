"""Engine 3: Center Anomaly Detector.

Algorithms: Feature Engineering (8 features) → Isolation Forest + Z-Score
Detects examination centers with systematic fraud patterns.

8 features per center:
  f1: mean_score — average correct answers
  f2: score_std — low std = suspicious uniformity
  f3: top_percentile_rate — % in national top 10%
  f4: pass_rate — % above passing threshold
  f5: wrong_answer_entropy — Shannon entropy of wrong answer distribution
  f6: high_similarity_pairs — count of highly similar pairs
  f7: score_z_vs_national — Z-score vs national mean
  f8: difficulty_correlation — correlation with national difficulty pattern
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)

CONTAMINATION = 0.05
N_ESTIMATORS = 200
Z_THRESHOLD = 3.0


class CenterAnomalyEngine(BaseEngine):
    """Detect anomalous examination centers using Isolation Forest."""

    def __init__(self):
        super().__init__(engine_name="center_anomaly", requires_gpu=False)

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
        n_students, n_questions = answers.shape
        n_options = int(answers.max()) + 1

        if student_ids is None:
            student_ids = [f"STU_{i:06d}" for i in range(n_students)]
        if center_ids is None:
            return EngineOutput(
                engine_name=self.engine_name,
                status="skipped",
                result_data={"message": "No center data available"},
            )

        # Get center metadata for geo coordinates
        center_metadata = kwargs.get("center_metadata", [])
        center_meta_map = {m["center_id"]: m for m in center_metadata}

        # ── Group students by center ──
        self.report_progress(10, "Grouping students by examination center")

        center_students: dict[str, list[int]] = {}
        for i, cid in enumerate(center_ids):
            center_students.setdefault(cid, []).append(i)

        unique_centers = list(center_students.keys())
        n_centers = len(unique_centers)
        logger.info(f"Found {n_centers} unique centers")

        # ── Compute correct answers ──
        if answer_key is not None:
            correct_matrix = (answers == answer_key[np.newaxis, :])
            scores = correct_matrix.sum(axis=1)
        else:
            scores = answers.sum(axis=1)
            correct_matrix = None

        national_mean = float(scores.mean())
        national_std = float(scores.std())
        top_10_threshold = float(np.percentile(scores, 90))
        pass_threshold = float(np.percentile(scores, 40))

        # National difficulty pattern
        if correct_matrix is not None:
            national_q_accuracy = correct_matrix.mean(axis=0)
        else:
            national_q_accuracy = None

        # ── Feature Engineering ──
        self.report_progress(30, "Engineering 8 features per center")

        feature_names = [
            "mean_score", "score_std", "top_percentile_rate", "pass_rate",
            "wrong_answer_entropy", "high_similarity_pairs",
            "z_vs_national", "difficulty_correlation",
        ]
        features = np.zeros((n_centers, 8))

        for c_idx, ctr_id in enumerate(unique_centers):
            s_indices = center_students[ctr_id]
            n_s = len(s_indices)
            center_scores = scores[s_indices]

            # f1: mean_score
            f1 = float(center_scores.mean())

            # f2: score_std (low = suspicious uniformity)
            f2 = float(center_scores.std()) if n_s > 1 else 0

            # f3: top_percentile_rate
            f3 = float((center_scores >= top_10_threshold).sum() / max(n_s, 1))

            # f4: pass_rate
            f4 = float((center_scores >= pass_threshold).sum() / max(n_s, 1))

            # f5: wrong_answer_entropy
            if answer_key is not None:
                wrong_answers_flat = []
                for s_idx in s_indices:
                    for q in range(n_questions):
                        if answers[s_idx, q] != answer_key[q]:
                            wrong_answers_flat.append(answers[s_idx, q])
                if wrong_answers_flat:
                    counts = np.bincount(wrong_answers_flat, minlength=n_options)
                    probs = counts / counts.sum()
                    probs = probs[probs > 0]
                    f5 = float(-np.sum(probs * np.log2(probs)))
                else:
                    f5 = np.log2(n_options)
            else:
                f5 = np.log2(n_options)

            # f6: high_similarity_pairs (sample-based for performance)
            sample_size = min(100, n_s)
            sample = np.random.choice(s_indices, size=sample_size, replace=False) if n_s > 2 else np.array(s_indices)
            high_sim_count = 0
            for ii in range(len(sample)):
                for jj in range(ii + 1, len(sample)):
                    sim = float((answers[sample[ii]] == answers[sample[jj]]).sum()) / n_questions
                    if sim > 0.8:
                        high_sim_count += 1
            # Normalize by number of compared pairs
            n_compared = max(len(sample) * (len(sample) - 1) // 2, 1)
            f6 = high_sim_count / n_compared

            # f7: Z-score vs national
            f7 = (f1 - national_mean) / max(national_std, 1e-8)

            # f8: difficulty_correlation
            if national_q_accuracy is not None and correct_matrix is not None:
                center_q_accuracy = correct_matrix[s_indices].mean(axis=0)
                if national_q_accuracy.std() > 0 and center_q_accuracy.std() > 0:
                    f8 = float(np.corrcoef(national_q_accuracy, center_q_accuracy)[0, 1])
                else:
                    f8 = 1.0
            else:
                f8 = 1.0

            features[c_idx] = [f1, f2, f3, f4, f5, f6, f7, f8]

            if c_idx % 50 == 0 and c_idx > 0:
                pct = 30 + int(30 * c_idx / n_centers)
                self.report_progress(pct, f"Features: {c_idx}/{n_centers} centers")

        # ── Isolation Forest ──
        self.report_progress(65, "Running Isolation Forest anomaly detection")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        iso_forest = IsolationForest(
            contamination=CONTAMINATION,
            n_estimators=N_ESTIMATORS,
            random_state=42,
            n_jobs=-1,
        )
        predictions = iso_forest.fit_predict(features_scaled)
        anomaly_scores_raw = iso_forest.decision_function(features_scaled)
        # Convert to 0-1 scale (higher = more anomalous)
        anomaly_scores = 1 - (anomaly_scores_raw - anomaly_scores_raw.min()) / (
            anomaly_scores_raw.max() - anomaly_scores_raw.min() + 1e-8
        )

        # ── Z-score flagging ──
        self.report_progress(80, "Computing per-feature Z-scores")

        z_scores = np.abs(scipy_stats.zscore(features, axis=0, nan_policy="omit"))
        max_z_per_center = z_scores.max(axis=1)

        # Composite score: 0.5 * IF + 0.3 * max_z_norm + 0.2 * sim_density
        max_z_norm = max_z_per_center / (max_z_per_center.max() + 1e-8)
        sim_density = features[:, 5] / (features[:, 5].max() + 1e-8)  # f6 normalized
        composite = 0.5 * anomaly_scores + 0.3 * max_z_norm + 0.2 * sim_density

        # ── Flag anomalous centers ──
        self.report_progress(85, "Identifying anomalous centers")

        flagged_centers: list[dict[str, Any]] = []
        flagged_student_ids: list[str] = []
        heatmap_data: list[dict[str, Any]] = []

        for c_idx, ctr_id in enumerate(unique_centers):
            is_anomalous = predictions[c_idx] == -1 or max_z_per_center[c_idx] > Z_THRESHOLD

            # Determine flags
            flags = []
            if features[c_idx, 6] > 2.5:  # z_vs_national
                flags.append("abnormally_high_scores")
            if features[c_idx, 1] < features[:, 1].mean() * 0.5:  # low std
                flags.append("suspicious_uniformity")
            if features[c_idx, 4] < features[:, 4].mean() * 0.7:  # low entropy
                flags.append("low_answer_diversity")
            if features[c_idx, 7] < 0.5:  # low difficulty correlation
                flags.append("difficulty_pattern_mismatch")
            if features[c_idx, 5] > features[:, 5].mean() * 3:  # high similarity
                flags.append("high_internal_similarity")
            if features[c_idx, 2] > 0.3:  # >30% in top 10%
                flags.append("inflated_top_performers")

            meta = center_meta_map.get(ctr_id, {})
            s_indices = center_students[ctr_id]

            center_data = {
                "center_id": ctr_id,
                "city": meta.get("city", "Unknown"),
                "state": meta.get("state", "Unknown"),
                "lat": meta.get("lat", 20.0),
                "lon": meta.get("lon", 78.0),
                "anomaly_score": round(float(composite[c_idx]), 3),
                "isolation_forest_score": round(float(anomaly_scores[c_idx]), 3),
                "max_z_score": round(float(max_z_per_center[c_idx]), 2),
                "student_count": len(s_indices),
                "features": {
                    fname: round(float(features[c_idx, fi]), 3)
                    for fi, fname in enumerate(feature_names)
                },
                "flags": flags,
                "is_anomalous": is_anomalous,
            }

            # Heatmap entry for all centers
            heatmap_data.append({
                "center_id": ctr_id,
                "lat": meta.get("lat", 20.0),
                "lon": meta.get("lon", 78.0),
                "anomaly_score": round(float(composite[c_idx]), 3),
                "student_count": len(s_indices),
                "city": meta.get("city", "Unknown"),
                "state": meta.get("state", "Unknown"),
                "flags": flags,
                "features": center_data["features"],
            })

            if is_anomalous:
                flagged_centers.append(center_data)
                for s_idx in s_indices:
                    flagged_student_ids.append(student_ids[s_idx])

        flagged_centers.sort(key=lambda c: c["anomaly_score"], reverse=True)

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_centers),
            flagged_student_ids=flagged_student_ids,
            result_data={
                "anomalous_centers": len(flagged_centers),
                "total_centers": n_centers,
                "centers": flagged_centers,
                "heatmap_data": heatmap_data,
                "feature_names": feature_names,
                "national_stats": {
                    "mean_score": round(national_mean, 1),
                    "std_score": round(national_std, 1),
                    "top_10_threshold": round(top_10_threshold, 1),
                },
            },
            summary={
                "anomalous_centers": len(flagged_centers),
                "total_centers": n_centers,
                "flagged_students": len(flagged_student_ids),
                "worst_center": flagged_centers[0]["center_id"] if flagged_centers else None,
                "worst_score": flagged_centers[0]["anomaly_score"] if flagged_centers else None,
            },
        )
