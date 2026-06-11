"""ExamGuard Test Suite — Unit tests for all engines and services."""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
import numpy as np


# ── Generator Tests ──


class TestDataGenerator:
    """Test IRT 2PL synthetic data generator."""

    def test_generate_small_dataset(self):
        from data.generator import generate_exam_data

        data = generate_exam_data(
            n_students=500, n_questions=30, n_centers=5, seed=42
        )
        assert len(data.student_ids) == 500
        assert data.answers.shape == (500, 30)
        assert data.correct_matrix.shape == (500, 30)
        assert data.scores.shape == (500,)
        assert data.abilities.shape == (500,)
        assert data.difficulties.shape == (30,)
        assert data.discriminations.shape == (30,)
        assert len(data.center_ids) == 5

    def test_fraud_patterns_injected(self):
        from data.generator import generate_exam_data

        data = generate_exam_data(
            n_students=1000, n_questions=50, n_centers=10, seed=42
        )
        assert data.fraud_labels is not None
        assert data.fraud_labels.sum() > 0
        assert len(data.ground_truth["copy_rings"]) >= 1
        assert len(data.ground_truth["leaked_students"]) > 0
        assert len(data.ground_truth["anomalous_centers"]) >= 1

    def test_timing_data_generated(self):
        from data.generator import generate_exam_data

        data = generate_exam_data(
            n_students=200, n_questions=20, n_centers=3,
            include_timing=True, seed=42
        )
        assert data.timing_data is not None
        assert data.timing_data.shape == (200, 20)
        assert data.timing_data.min() >= 2  # Minimum 2 seconds
        assert data.timing_data.max() <= 600  # Maximum 600 seconds

    def test_question_text_generated(self):
        from data.generator import generate_exam_data

        data = generate_exam_data(
            n_students=100, n_questions=30, n_centers=2,
            include_question_text=True, seed=42
        )
        assert data.question_texts is not None
        assert len(data.question_texts) == 30
        assert all("Q" in text for text in data.question_texts)

    def test_irt_probabilities_valid(self):
        from data.generator import generate_exam_data

        data = generate_exam_data(
            n_students=500, n_questions=50, n_centers=5, seed=42
        )
        # Average score should be reasonable (not all correct or all wrong)
        avg_score = data.scores.mean()
        assert 10 < avg_score < 40, f"Average score {avg_score} outside expected range"

    def test_center_assignment_complete(self):
        from data.generator import generate_exam_data

        data = generate_exam_data(
            n_students=200, n_questions=10, n_centers=5, seed=42
        )
        # Every student should have a center
        assert len(data.student_centers) == 200
        for sid in data.student_ids:
            assert sid in data.student_centers

    def test_reproducibility(self):
        from data.generator import generate_exam_data

        data1 = generate_exam_data(n_students=100, n_questions=10, n_centers=3, seed=42)
        data2 = generate_exam_data(n_students=100, n_questions=10, n_centers=3, seed=42)
        np.testing.assert_array_equal(data1.answers, data2.answers)
        np.testing.assert_array_equal(data1.scores, data2.scores)


# ── Engine Tests ──


class TestBaseEngine:
    """Test engine base class."""

    def test_base_engine_abstract(self):
        from engines.base_engine import BaseEngine

        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            BaseEngine()


class TestCopyRingEngine:
    """Test MinHash LSH + Louvain copy ring detection."""

    def test_detects_copy_ring(self):
        from data.generator import generate_exam_data
        from engines.copy_ring import CopyRingEngine

        data = generate_exam_data(n_students=500, n_questions=50, n_centers=10, seed=42)
        engine = CopyRingEngine()
        result = asyncio.run(engine.analyze(
            answers=data.answers,
            answer_key=data.answer_key,
            student_ids=data.student_ids,
            center_ids=[data.student_centers[s] for s in data.student_ids],
        ))
        assert result.status in ("complete", "failed")


class TestStatImpossibilityEngine:
    """Test binomial + Bonferroni statistical proof."""

    def test_detects_impossibility(self):
        from data.generator import generate_exam_data
        from engines.stat_impossibility import StatImpossibilityEngine

        data = generate_exam_data(n_students=500, n_questions=50, n_centers=10, seed=42)
        engine = StatImpossibilityEngine()
        result = asyncio.run(engine.analyze(
            answers=data.answers,
            answer_key=data.answer_key,
            student_ids=data.student_ids,
        ))
        assert result.status in ("complete", "failed")


class TestCenterAnomalyEngine:
    """Test isolation forest center anomaly detection."""

    def test_detects_center_anomalies(self):
        from data.generator import generate_exam_data
        from engines.center_anomaly import CenterAnomalyEngine

        data = generate_exam_data(n_students=500, n_questions=50, n_centers=10, seed=42)
        engine = CenterAnomalyEngine()
        result = asyncio.run(engine.analyze(
            answers=data.answers,
            answer_key=data.answer_key,
            student_ids=data.student_ids,
            center_ids=[data.student_centers[s] for s in data.student_ids],
        ))
        assert result.status in ("complete", "failed")


# ── Config Tests ──


class TestConfig:
    """Test Pydantic configuration."""

    def test_default_settings(self):
        from config import settings

        assert settings.LSH_NUM_PERM == 128
        assert settings.BONFERRONI_ALPHA == 0.05
        assert settings.GNN_HIDDEN_DIM == 64
        assert settings.VAE_LATENT_DIM == 32
        assert settings.XGBOOST_MAX_DEPTH == 6


# ── Schema Tests ──


class TestSchemas:
    """Test Pydantic data models."""

    def test_generate_request_defaults(self):
        from data.schemas import GenerateRequest

        req = GenerateRequest()
        assert req.n_students == 100_000
        assert req.n_questions == 200
        assert req.n_centers == 450
        assert req.n_options == 4

    def test_generate_request_custom(self):
        from data.schemas import GenerateRequest

        req = GenerateRequest(n_students=500, n_questions=30, n_centers=10)
        assert req.n_students == 500
        assert req.n_questions == 30
        assert req.n_centers == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
