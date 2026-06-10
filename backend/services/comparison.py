"""Student comparison logic — per-question diff, matching stats, WAA."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from data.ingestion import ingest_csv


def compare_students(csv_content: bytes, student_a: str, student_b: str) -> dict[str, Any]:
    """Compare two students' answer sheets side-by-side."""
    ingested = ingest_csv(csv_content)

    idx_a = None
    idx_b = None
    for i, sid in enumerate(ingested.student_ids):
        if sid == student_a:
            idx_a = i
        if sid == student_b:
            idx_b = i

    if idx_a is None:
        raise ValueError(f"Student {student_a} not found")
    if idx_b is None:
        raise ValueError(f"Student {student_b} not found")

    answers_a = ingested.answers[idx_a]
    answers_b = ingested.answers[idx_b]
    n_questions = len(answers_a)

    matches = (answers_a == answers_b)
    matching_total = int(matches.sum())

    matching_wrong = 0
    per_question = []

    for q in range(n_questions):
        is_match = bool(answers_a[q] == answers_b[q])
        correct_a = bool(answers_a[q] == ingested.answer_key[q]) if ingested.answer_key is not None else None
        correct_b = bool(answers_b[q] == ingested.answer_key[q]) if ingested.answer_key is not None else None
        is_both_wrong = False
        is_both_correct = False

        if ingested.answer_key is not None:
            is_both_correct = correct_a and correct_b
            is_both_wrong = not correct_a and not correct_b
            if is_match and is_both_wrong:
                matching_wrong += 1

        per_question.append({
            "question": q + 1,
            "answer_a": int(answers_a[q]),
            "answer_b": int(answers_b[q]),
            "correct_answer": int(ingested.answer_key[q]) if ingested.answer_key is not None else None,
            "is_match": is_match,
            "is_both_wrong": is_both_wrong,
            "is_both_correct": is_both_correct,
        })

    # Jaccard
    jaccard = matching_total / n_questions if n_questions > 0 else 0

    # WAA
    if ingested.answer_key is not None:
        wrong_a = set(q for q in range(n_questions) if answers_a[q] != ingested.answer_key[q])
        wrong_b = set(q for q in range(n_questions) if answers_b[q] != ingested.answer_key[q])
        wrong_union = wrong_a | wrong_b
        waa = matching_wrong / len(wrong_union) if wrong_union else 0
    else:
        waa = 0

    # P-value (simplified)
    from scipy import stats
    n_options = int(ingested.answers.max()) + 1
    p_avg = 1.0 / n_options
    p_value = float(stats.binom.sf(matching_total - 1, n_questions, p_avg))

    from engines.stat_impossibility import _format_probability
    human_readable = _format_probability(p_value) if p_value < 0.01 else f"1 in {int(1/max(p_value,1e-10))}"

    return {
        "student_a": student_a,
        "student_b": student_b,
        "total_questions": n_questions,
        "matching_total": matching_total,
        "matching_wrong": matching_wrong,
        "jaccard": round(jaccard, 4),
        "waa": round(waa, 4),
        "p_value": p_value,
        "human_readable": human_readable,
        "per_question": per_question,
    }
