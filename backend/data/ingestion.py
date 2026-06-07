"""CSV data ingestion and validation for ExamGuard.

Parses uploaded CSV files, validates structure, detects column formats,
and extracts answer data, timing data, and metadata.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IngestedData:
    """Parsed and validated exam data from CSV."""
    student_ids: list[str]
    center_ids: list[str]
    unique_centers: list[str]
    answers: np.ndarray  # (n_students, n_questions) int
    answer_key: Optional[np.ndarray] = None  # (n_questions,) int
    timing_data: Optional[np.ndarray] = None  # (n_students, n_questions) float
    question_texts: Optional[list[str]] = None
    n_students: int = 0
    n_questions: int = 0
    n_centers: int = 0
    n_options: int = 4
    has_timing: bool = False
    has_question_text: bool = False
    validation_warnings: list[str] = field(default_factory=list)


def ingest_csv(
    file_content: bytes,
    answer_key_content: Optional[bytes] = None,
    question_text_content: Optional[bytes] = None,
    n_options: int = 4,
) -> IngestedData:
    """Parse and validate an uploaded exam CSV file.

    Expected CSV format:
        student_id, center_id, q_1, q_2, ..., q_N
        Optional: time_q_1, time_q_2, ..., time_q_N

    Args:
        file_content: Raw CSV bytes
        answer_key_content: Optional CSV with answer key
        question_text_content: Optional CSV with question texts
        n_options: Number of answer options per question

    Returns:
        IngestedData with validated and parsed data

    Raises:
        ValueError: If CSV format is invalid or required columns missing
    """
    logger.info("Starting CSV ingestion")

    # Read CSV
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    warnings: list[str] = []

    # ── Detect Column Structure ──
    columns = df.columns.tolist()

    # Find student_id column
    student_col = _find_column(columns, ["student_id", "studentid", "student", "id", "roll_no", "roll_number"])
    if student_col is None:
        raise ValueError("Could not find student_id column. Expected: student_id, studentid, student, id, roll_no")

    # Find center_id column
    center_col = _find_column(columns, ["center_id", "centerid", "center", "centre_id", "exam_center"])
    if center_col is None:
        warnings.append("No center_id column found — treating all students as single center")

    # Find answer columns (q_1, q_2, ... or q1, q2, ...)
    answer_cols = [c for c in columns if _is_answer_col(c)]
    if not answer_cols:
        raise ValueError("Could not find answer columns. Expected: q_1, q_2, ... or q1, q2, ...")

    n_questions = len(answer_cols)
    logger.info(f"Detected {n_questions} answer columns")

    # Find timing columns
    timing_cols = [c for c in columns if _is_timing_col(c)]
    has_timing = len(timing_cols) == n_questions
    if timing_cols and not has_timing:
        warnings.append(f"Found {len(timing_cols)} timing columns but expected {n_questions} — ignoring timing data")
        has_timing = False

    # ── Extract Data ──
    student_ids = df[student_col].astype(str).tolist()
    n_students = len(student_ids)

    center_ids: list[str] = []
    if center_col:
        center_ids = df[center_col].astype(str).tolist()
    else:
        center_ids = ["CTR_000"] * n_students

    unique_centers = sorted(set(center_ids))
    n_centers = len(unique_centers)

    # Parse answers
    try:
        answers = df[answer_cols].values.astype(np.int32)
    except (ValueError, TypeError):
        # Try mapping string options to integers
        answer_df = df[answer_cols].copy()
        for col in answer_cols:
            answer_df[col] = _map_answers_to_int(answer_df[col])
        answers = answer_df.values.astype(np.int32)

    # Validate answer range
    max_answer = answers.max()
    min_answer = answers.min()
    if min_answer < 0:
        warnings.append(f"Found negative answer values (min={min_answer}), clipping to 0")
        answers = np.clip(answers, 0, n_options - 1)
    if max_answer >= n_options:
        detected_options = max_answer + 1
        warnings.append(f"Detected {detected_options} answer options (max value = {max_answer})")
        n_options = detected_options

    # Parse timing data
    timing_data = None
    if has_timing:
        try:
            timing_data = df[timing_cols].values.astype(np.float64)
            # Replace NaN with median
            for q in range(n_questions):
                col_data = timing_data[:, q]
                nan_mask = np.isnan(col_data)
                if nan_mask.any():
                    median_val = np.nanmedian(col_data)
                    timing_data[nan_mask, q] = median_val
            logger.info("Timing data loaded successfully")
        except (ValueError, TypeError):
            warnings.append("Failed to parse timing data — ignoring")
            timing_data = None
            has_timing = False

    # Parse answer key
    answer_key = None
    if answer_key_content:
        try:
            key_df = pd.read_csv(io.BytesIO(answer_key_content))
            if len(key_df.columns) == 1:
                answer_key = key_df.iloc[:, 0].values.astype(np.int32)
            elif "answer" in key_df.columns:
                answer_key = key_df["answer"].values.astype(np.int32)
            else:
                answer_key = key_df.iloc[:, 0].values.astype(np.int32)
            if len(answer_key) != n_questions:
                warnings.append(f"Answer key has {len(answer_key)} answers but {n_questions} questions — ignoring")
                answer_key = None
            else:
                logger.info("Answer key loaded")
        except Exception as e:
            warnings.append(f"Failed to parse answer key: {e}")

    # Parse question texts
    question_texts = None
    if question_text_content:
        try:
            qt_df = pd.read_csv(io.BytesIO(question_text_content))
            question_texts = qt_df.iloc[:, 0].astype(str).tolist()
            if len(question_texts) != n_questions:
                warnings.append(f"Question text count ({len(question_texts)}) != questions ({n_questions})")
                question_texts = None
        except Exception as e:
            warnings.append(f"Failed to parse question texts: {e}")

    logger.info(
        f"Ingestion complete: {n_students:,} students, {n_questions} questions, "
        f"{n_centers} centers, timing={'yes' if has_timing else 'no'}"
    )

    if warnings:
        for w in warnings:
            logger.warning(f"Ingestion warning: {w}")

    return IngestedData(
        student_ids=student_ids,
        center_ids=center_ids,
        unique_centers=unique_centers,
        answers=answers,
        answer_key=answer_key,
        timing_data=timing_data,
        question_texts=question_texts,
        n_students=n_students,
        n_questions=n_questions,
        n_centers=n_centers,
        n_options=n_options,
        has_timing=has_timing,
        has_question_text=question_texts is not None,
        validation_warnings=warnings,
    )


# ── Helpers ──

def _find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Find a column name from candidates (case-insensitive)."""
    lower_cols = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_cols:
            return lower_cols[candidate.lower()]
    return None


def _is_answer_col(col: str) -> bool:
    """Check if column name matches answer pattern (q_1, q1, etc.)."""
    col_lower = col.lower().strip()
    if col_lower.startswith("q_") or col_lower.startswith("q"):
        suffix = col_lower.replace("q_", "").replace("q", "")
        return suffix.isdigit()
    return False


def _is_timing_col(col: str) -> bool:
    """Check if column name matches timing pattern (time_q_1, etc.)."""
    col_lower = col.lower().strip()
    return col_lower.startswith("time_q_") or col_lower.startswith("time_q")


def _map_answers_to_int(series: pd.Series) -> pd.Series:
    """Map string answers (A, B, C, D) to integers (0, 1, 2, 3)."""
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    return series.map(lambda x: mapping.get(str(x).strip(), 0))
