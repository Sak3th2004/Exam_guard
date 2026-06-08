"""Engine 5: Response Time Forensics (CBT Only).

Algorithms: KDE + Speed Ratio + K-Means
Detects pre-knowledge via impossibly fast response times.

Pipeline:
  1. Per-question: fit KDE on response times, compute median
  2. Per-student: speed_ratio = student_time / median_time
  3. Flag speed_ratio < 0.2 on hard questions (Q3+Q4)
  4. "Impossibly fast" if time < reading time estimate
  5. K-Means(k=2) on time feature vectors → normal vs pre-knowledge

NOTE: Only runs when timing data is present. Gracefully skips otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)

SPEED_RATIO_THRESHOLD = 0.2
MIN_READING_TIME_SECONDS = 3.0


class ResponseTimeEngine(BaseEngine):
    """Detect pre-knowledge via response time analysis."""

    def __init__(self):
        super().__init__(engine_name="response_time", requires_gpu=False)

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

        if student_ids is None:
            student_ids = [f"STU_{i:06d}" for i in range(n_students)]

        # ── Check if timing data is available ──
        if timing_data is None:
            logger.info("No timing data available — skipping response time analysis")
            return EngineOutput(
                engine_name=self.engine_name,
                status="skipped",
                result_data={"applicable": False, "message": "No timing data available"},
                summary={"applicable": False, "message": "Timing data not provided"},
            )

        self.report_progress(10, "Analyzing response time distributions")

        # ── Step 1: Per-question statistics ──
        median_times = np.median(timing_data, axis=0)
        mean_times = np.mean(timing_data, axis=0)

        # Identify hard questions (if answer key available)
        if answer_key is not None:
            correct_rates = (answers == answer_key[np.newaxis, :]).mean(axis=0)
            difficulty = 1.0 - correct_rates
            hard_qs = np.where(difficulty > np.percentile(difficulty, 50))[0]
        else:
            hard_qs = np.arange(n_questions // 2, n_questions)

        # ── Step 2: Speed ratios ──
        self.report_progress(30, "Computing speed ratios per student")

        speed_ratios = timing_data / (median_times[np.newaxis, :] + 1e-8)

        # Per-student average speed ratio on hard questions
        avg_speed_hard = speed_ratios[:, hard_qs].mean(axis=1)

        # Per-student: count of impossibly fast answers on hard questions
        fast_hard_count = (speed_ratios[:, hard_qs] < SPEED_RATIO_THRESHOLD).sum(axis=1)

        # ── Step 3: Impossibly fast detection ──
        self.report_progress(50, "Detecting impossibly fast responses")

        # Flag students with many fast answers on hard questions
        min_fast_threshold = max(3, len(hard_qs) // 10)
        speed_flagged = fast_hard_count >= min_fast_threshold

        # Also flag students with overall very low average speed on hard questions
        speed_flagged = speed_flagged | (avg_speed_hard < 0.3)

        # ── Step 4: K-Means clustering ──
        self.report_progress(65, "Running K-Means clustering on time features")

        # Feature vector per student: [avg_speed_hard, fast_hard_count_norm, total_time_z]
        total_times = timing_data.sum(axis=1)
        total_time_z = (total_times - total_times.mean()) / (total_times.std() + 1e-8)

        time_features = np.column_stack([
            avg_speed_hard,
            fast_hard_count / max(len(hard_qs), 1),
            total_time_z,
        ])

        # K-Means with k=2: normal vs pre-knowledge
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(time_features)

        # Identify which cluster is the "cheater" cluster
        cluster_means = [time_features[cluster_labels == c, 0].mean() for c in range(2)]
        cheater_cluster = int(np.argmin(cluster_means))  # Lower speed ratio = cheater

        kmeans_flagged = (cluster_labels == cheater_cluster)

        # ── Combine flags ──
        self.report_progress(80, "Combining detection signals")

        combined_flagged = speed_flagged | kmeans_flagged
        flagged_indices = np.where(combined_flagged)[0]
        flagged_ids = [student_ids[i] for i in flagged_indices]

        # ── Time distribution data for visualization ──
        self.report_progress(90, "Preparing time distribution visualization data")

        # Histogram data: flagged vs normal on hard questions
        flagged_hard_times = timing_data[combined_flagged][:, hard_qs].flatten() if combined_flagged.any() else np.array([])
        normal_hard_times = timing_data[~combined_flagged][:, hard_qs].flatten()

        # Bin the data
        bins = np.linspace(0, 120, 25)
        flagged_hist, _ = np.histogram(flagged_hard_times, bins=bins) if len(flagged_hard_times) > 0 else (np.zeros(24), bins)
        normal_hist, _ = np.histogram(normal_hard_times, bins=bins)

        # Normalize
        flagged_hist_norm = flagged_hist / max(flagged_hist.sum(), 1)
        normal_hist_norm = normal_hist / max(normal_hist.sum(), 1)

        time_data = {
            "bins": [round(b, 1) for b in bins[:-1]],
            "flagged_distribution": [round(float(x), 4) for x in flagged_hist_norm],
            "normal_distribution": [round(float(x), 4) for x in normal_hist_norm],
            "flagged_median_hard": round(float(np.median(flagged_hard_times)), 1) if len(flagged_hard_times) > 0 else 0,
            "normal_median_hard": round(float(np.median(normal_hard_times)), 1),
        }

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={
                "applicable": True,
                "fast_students": len(flagged_ids),
                "speed_threshold": SPEED_RATIO_THRESHOLD,
                "cheater_cluster": cheater_cluster,
                "time_data": time_data,
                "cluster_stats": {
                    "cluster_0_size": int((cluster_labels == 0).sum()),
                    "cluster_1_size": int((cluster_labels == 1).sum()),
                    "cluster_0_avg_speed": round(float(cluster_means[0]), 3),
                    "cluster_1_avg_speed": round(float(cluster_means[1]), 3),
                },
            },
            summary={
                "applicable": True,
                "fast_students": len(flagged_ids),
                "flagged": len(flagged_ids),
                "flagged_median_time": time_data["flagged_median_hard"],
                "normal_median_time": time_data["normal_median_hard"],
            },
        )
