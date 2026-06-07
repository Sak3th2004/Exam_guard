"""IRT-Based Synthetic Exam Data Generator.

Generates statistically realistic examination data using Item Response Theory
(2-Parameter Logistic model) with deliberately planted fraud patterns.

Fraud Patterns:
  A) Copy rings — clusters of students sharing answers
  B) Paper leak — students with leaked question answers
  C) Center anomalies — centers with systematic fraud
  D) Response time cheating — impossibly fast answers

All data follows real psychometric distributions. NO random/fake data.
Ground truth labels enable supervised training of GNN + XGBoost.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.special import expit

logger = logging.getLogger(__name__)

# ── Indian Cities for Center Metadata ──

INDIAN_CITIES: list[dict[str, Any]] = [
    {"city": "Mumbai", "state": "Maharashtra", "lat": 19.076, "lon": 72.877},
    {"city": "Delhi", "state": "Delhi", "lat": 28.704, "lon": 77.102},
    {"city": "Bangalore", "state": "Karnataka", "lat": 12.972, "lon": 77.594},
    {"city": "Hyderabad", "state": "Telangana", "lat": 17.385, "lon": 78.486},
    {"city": "Chennai", "state": "Tamil Nadu", "lat": 13.083, "lon": 80.270},
    {"city": "Kolkata", "state": "West Bengal", "lat": 22.572, "lon": 88.363},
    {"city": "Pune", "state": "Maharashtra", "lat": 18.520, "lon": 73.856},
    {"city": "Ahmedabad", "state": "Gujarat", "lat": 23.023, "lon": 72.571},
    {"city": "Jaipur", "state": "Rajasthan", "lat": 26.912, "lon": 75.787},
    {"city": "Lucknow", "state": "Uttar Pradesh", "lat": 26.846, "lon": 80.946},
    {"city": "Patna", "state": "Bihar", "lat": 25.611, "lon": 85.144},
    {"city": "Bhopal", "state": "Madhya Pradesh", "lat": 23.259, "lon": 77.412},
    {"city": "Chandigarh", "state": "Chandigarh", "lat": 30.733, "lon": 76.779},
    {"city": "Ranchi", "state": "Jharkhand", "lat": 23.344, "lon": 85.309},
    {"city": "Thiruvananthapuram", "state": "Kerala", "lat": 8.524, "lon": 76.936},
    {"city": "Guwahati", "state": "Assam", "lat": 26.144, "lon": 91.736},
    {"city": "Bhubaneswar", "state": "Odisha", "lat": 20.296, "lon": 85.824},
    {"city": "Dehradun", "state": "Uttarakhand", "lat": 30.316, "lon": 78.032},
    {"city": "Raipur", "state": "Chhattisgarh", "lat": 21.251, "lon": 81.629},
    {"city": "Indore", "state": "Madhya Pradesh", "lat": 22.719, "lon": 75.857},
    {"city": "Nagpur", "state": "Maharashtra", "lat": 21.145, "lon": 79.088},
    {"city": "Varanasi", "state": "Uttar Pradesh", "lat": 25.317, "lon": 82.991},
    {"city": "Coimbatore", "state": "Tamil Nadu", "lat": 11.016, "lon": 76.955},
    {"city": "Kochi", "state": "Kerala", "lat": 9.931, "lon": 76.267},
    {"city": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.686, "lon": 83.218},
    {"city": "Surat", "state": "Gujarat", "lat": 21.170, "lon": 72.831},
    {"city": "Kanpur", "state": "Uttar Pradesh", "lat": 26.449, "lon": 80.331},
    {"city": "Agra", "state": "Uttar Pradesh", "lat": 27.176, "lon": 78.008},
    {"city": "Allahabad", "state": "Uttar Pradesh", "lat": 25.431, "lon": 81.846},
    {"city": "Amritsar", "state": "Punjab", "lat": 31.633, "lon": 74.872},
    {"city": "Gwalior", "state": "Madhya Pradesh", "lat": 26.218, "lon": 78.182},
    {"city": "Jodhpur", "state": "Rajasthan", "lat": 26.238, "lon": 73.024},
    {"city": "Mysore", "state": "Karnataka", "lat": 12.295, "lon": 76.639},
    {"city": "Nashik", "state": "Maharashtra", "lat": 19.997, "lon": 73.789},
    {"city": "Vijayawada", "state": "Andhra Pradesh", "lat": 16.506, "lon": 80.648},
    {"city": "Madurai", "state": "Tamil Nadu", "lat": 9.925, "lon": 78.119},
    {"city": "Meerut", "state": "Uttar Pradesh", "lat": 28.984, "lon": 77.706},
    {"city": "Rajkot", "state": "Gujarat", "lat": 22.303, "lon": 70.802},
    {"city": "Jabalpur", "state": "Madhya Pradesh", "lat": 23.181, "lon": 79.986},
    {"city": "Dhanbad", "state": "Jharkhand", "lat": 23.795, "lon": 86.430},
    {"city": "Shimla", "state": "Himachal Pradesh", "lat": 31.104, "lon": 77.172},
    {"city": "Jammu", "state": "Jammu & Kashmir", "lat": 32.726, "lon": 74.857},
    {"city": "Goa", "state": "Goa", "lat": 15.299, "lon": 74.124},
    {"city": "Udaipur", "state": "Rajasthan", "lat": 24.585, "lon": 73.712},
    {"city": "Tiruchirappalli", "state": "Tamil Nadu", "lat": 10.790, "lon": 78.704},
    {"city": "Salem", "state": "Tamil Nadu", "lat": 11.664, "lon": 78.146},
    {"city": "Bareilly", "state": "Uttar Pradesh", "lat": 28.367, "lon": 79.432},
    {"city": "Aligarh", "state": "Uttar Pradesh", "lat": 27.897, "lon": 78.088},
    {"city": "Moradabad", "state": "Uttar Pradesh", "lat": 28.838, "lon": 78.776},
    {"city": "Gorakhpur", "state": "Uttar Pradesh", "lat": 26.760, "lon": 83.373},
]

# ── Sample Question Text Templates ──

QUESTION_SUBJECTS: list[dict[str, list[str]]] = [
    {
        "subject": "Physics",
        "templates": [
            "A body of mass {m} kg is moving with velocity {v} m/s. Calculate the kinetic energy.",
            "An object is thrown vertically upward with initial velocity {v} m/s. Find the maximum height reached.",
            "A wire of length {l} m and area of cross-section {a} m² has resistance {r} Ω. Find the resistivity.",
            "Two parallel plates are separated by distance {d} cm. If the potential difference is {v} V, find the electric field.",
            "A concave mirror has focal length {f} cm. Find the image position for an object at distance {u} cm.",
            "A transformer has {n1} turns in primary and {n2} turns in secondary coil. Find the voltage ratio.",
            "A capacitor of capacitance {c} μF is charged to potential {v} V. Calculate the energy stored.",
            "A satellite orbits Earth at height {h} km. Calculate the orbital velocity.",
            "In Young's double slit experiment, slit separation is {d} mm and screen distance is {D} m. Find fringe width for wavelength {λ} nm.",
            "A body starts from rest and accelerates at {a} m/s² for {t} seconds. Find the distance covered.",
        ],
    },
    {
        "subject": "Chemistry",
        "templates": [
            "Calculate the molarity of a solution containing {g} g of {compound} dissolved in {v} mL of solution.",
            "What is the hybridization of the central atom in {molecule}?",
            "Calculate the pH of a {c} M solution of {acid}.",
            "Balance the following reaction: {reactant1} + {reactant2} → {product1} + {product2}",
            "How many moles of {gas} are present in {v} L at STP?",
            "What is the oxidation state of {element} in {compound}?",
            "Calculate the bond order of {molecule} using molecular orbital theory.",
            "Determine the crystal field splitting energy for {complex} in octahedral field.",
            "What is the IUPAC name of {compound}?",
            "Calculate the cell potential for the galvanic cell: {anode} | {cathode}",
        ],
    },
    {
        "subject": "Biology",
        "templates": [
            "Which enzyme is responsible for {process} in {organelle}?",
            "Describe the role of {hormone} in {system}.",
            "What is the function of {structure} in the cell?",
            "In which phase of cell division does {event} occur?",
            "What type of inheritance pattern is shown by {trait} in humans?",
            "Name the disease caused by deficiency of {vitamin}.",
            "Which part of the brain controls {function}?",
            "Describe the process of {process} in plants.",
            "What is the taxonomic classification of {organism}?",
            "Explain the mechanism of {process} in the immune system.",
        ],
    },
]


@dataclass
class GeneratedData:
    """Container for all generated exam data."""
    student_ids: list[str]
    center_ids: list[str]
    student_centers: dict[str, str]  # student_id → center_id
    answers: np.ndarray  # (n_students, n_questions)
    answer_key: np.ndarray  # (n_questions,)
    correct_matrix: np.ndarray  # (n_students, n_questions) boolean
    scores: np.ndarray  # (n_students,) total correct
    abilities: np.ndarray  # (n_students,) IRT theta
    difficulties: np.ndarray  # (n_questions,) IRT difficulty
    discriminations: np.ndarray  # (n_questions,) IRT discrimination
    center_metadata: list[dict[str, Any]]
    question_texts: Optional[list[str]] = None
    timing_data: Optional[np.ndarray] = None  # (n_students, n_questions) seconds

    # Ground truth labels
    ground_truth: dict[str, Any] = field(default_factory=dict)
    fraud_labels: Optional[np.ndarray] = None  # (n_students,) 0=clean, 1=fraud


def generate_exam_data(
    n_students: int = 100_000,
    n_questions: int = 200,
    n_centers: int = 450,
    n_options: int = 4,
    include_timing: bool = True,
    include_question_text: bool = True,
    seed: int = 42,
    progress_callback: Optional[callable] = None,
) -> GeneratedData:
    """Generate complete synthetic exam data with planted fraud patterns.

    Uses IRT 2PL model for realistic answer generation:
        P(correct | θ, a, b) = 1 / (1 + exp(-a * (θ - b)))

    Args:
        n_students: Number of students (default 100,000)
        n_questions: Number of questions (default 200)
        n_centers: Number of exam centers (default 450)
        n_options: Number of answer options per question (default 4)
        include_timing: Whether to generate response time data
        include_question_text: Whether to generate question text
        seed: Random seed for reproducibility
        progress_callback: Optional callback for progress updates

    Returns:
        GeneratedData with all fields populated including ground truth
    """
    rng = np.random.RandomState(seed)
    logger.info(
        f"Generating exam data: {n_students:,} students × {n_questions} questions × {n_centers} centers"
    )

    def report_progress(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)
        logger.info(f"[{pct}%] {msg}")

    report_progress(5, "Generating student abilities (IRT θ distribution)")

    # ── Step 1: Student Abilities ──
    # Normal distribution: most students are average, few are very high/low
    abilities = rng.normal(0, 1, n_students)

    # ── Step 2: Question Parameters (IRT 2PL) ──
    report_progress(8, "Generating question parameters (difficulty, discrimination)")

    # Difficulty: Beta(2,5) → right-skewed (more easy questions), scaled to [-3, 3]
    difficulties = rng.beta(2, 5, n_questions) * 6 - 3

    # Discrimination: LogNormal → always positive, clipped
    discriminations = np.clip(rng.lognormal(0, 0.5, n_questions), 0.5, 3.0)

    # ── Step 3: Answer Key ──
    # Random correct answers (0 to n_options-1)
    answer_key = rng.randint(0, n_options, n_questions)

    # ── Step 4: Center Assignment ──
    report_progress(10, "Assigning students to examination centers")

    center_ids = [f"CTR_{i:03d}" for i in range(n_centers)]
    student_ids = [f"STU_{i:06d}" for i in range(n_students)]

    # Students per center: variable (50-300), slightly ability-clustered
    # Coaching hubs have higher-ability students
    center_sizes = _generate_center_sizes(n_students, n_centers, rng)
    student_center_map: dict[str, str] = {}
    student_indices_by_center: dict[str, list[int]] = {}

    # Sort students by ability, then assign with some clustering
    sorted_indices = np.argsort(abilities)
    # Shuffle with a window to add noise while preserving clustering
    shuffled_indices = _windowed_shuffle(sorted_indices, window=n_students // 10, rng=rng)

    idx = 0
    for c_idx, size in enumerate(center_sizes):
        center_id = center_ids[c_idx]
        student_indices_by_center[center_id] = []
        for _ in range(size):
            if idx < n_students:
                s_id = student_ids[shuffled_indices[idx]]
                student_center_map[s_id] = center_id
                student_indices_by_center[center_id].append(shuffled_indices[idx])
                idx += 1

    # ── Step 5: Generate Answers Using IRT 2PL ──
    report_progress(15, "Computing IRT 2PL response probabilities (vectorized)")

    # P(correct) = expit(a * (θ - b)) — vectorized computation
    # Shape: (n_students, n_questions)
    logits = discriminations[np.newaxis, :] * (abilities[:, np.newaxis] - difficulties[np.newaxis, :])
    prob_correct = expit(logits)

    # Generate correct/incorrect for each student×question
    random_draws = rng.random((n_students, n_questions))
    correct_matrix = random_draws < prob_correct

    report_progress(20, "Generating answer choices with weighted distractors")

    # Build answer matrix
    answers = np.zeros((n_students, n_questions), dtype=np.int32)

    for q in range(n_questions):
        correct_ans = answer_key[q]
        # Correct students get the correct answer
        answers[correct_matrix[:, q], q] = correct_ans

        # Incorrect students get weighted wrong answers (Dirichlet-distributed)
        wrong_mask = ~correct_matrix[:, q]
        n_wrong = wrong_mask.sum()
        if n_wrong > 0:
            wrong_options = [o for o in range(n_options) if o != correct_ans]
            # Dirichlet weights for distractor attractiveness
            distractor_weights = rng.dirichlet(np.ones(len(wrong_options)) * 2.0)
            wrong_choices = rng.choice(wrong_options, size=n_wrong, p=distractor_weights)
            answers[wrong_mask, q] = wrong_choices

    scores = correct_matrix.sum(axis=1)

    # ── Step 6: Center Metadata ──
    report_progress(25, "Generating center metadata with Indian city geolocation")

    center_metadata = _generate_center_metadata(center_ids, student_indices_by_center, scores, rng)

    # ── Step 7: Plant Fraud Patterns ──
    report_progress(30, "Injecting fraud Pattern A: copy rings")

    ground_truth: dict[str, Any] = {
        "copy_rings": [],
        "leaked_students": [],
        "anomalous_centers": [],
        "timing_cheaters": [],
    }
    fraud_labels = np.zeros(n_students, dtype=np.int32)

    # Pattern A: Copy Rings
    ring_configs = [
        {"size": 23, "center_idx": 42, "overlap": 0.85},
        {"size": 15, "center_idx": 89, "overlap": 0.80},
        {"size": 8, "center_indices": [105, 106], "overlap": 0.82},
    ]

    for ring_config in ring_configs:
        ring_students = _inject_copy_ring(
            answers=answers,
            answer_key=answer_key,
            student_indices_by_center=student_indices_by_center,
            center_ids=center_ids,
            config=ring_config,
            n_questions=n_questions,
            n_options=n_options,
            rng=rng,
        )
        if ring_students:
            ground_truth["copy_rings"].append({
                "student_indices": ring_students,
                "center": ring_config.get("center_idx", ring_config.get("center_indices")),
                "overlap": ring_config["overlap"],
                "size": len(ring_students),
            })
            fraud_labels[ring_students] = 1

    # Pattern B: Paper Leak
    report_progress(40, "Injecting fraud Pattern B: paper leak (Q45-Q120)")

    leaked_indices = _inject_paper_leak(
        answers=answers,
        answer_key=answer_key,
        correct_matrix=correct_matrix,
        abilities=abilities,
        n_students=n_students,
        n_leak_students=340,
        leak_range=(44, 120),  # 0-indexed: Q45-Q120
        accuracy_range=(0.88, 0.95),
        rng=rng,
    )
    ground_truth["leaked_students"] = leaked_indices.tolist()
    fraud_labels[leaked_indices] = 1

    # Pattern C: Center Anomalies
    report_progress(50, "Injecting fraud Pattern C: center anomalies")

    anomalous_center_indices = _inject_center_anomalies(
        answers=answers,
        answer_key=answer_key,
        correct_matrix=correct_matrix,
        student_indices_by_center=student_indices_by_center,
        center_ids=center_ids,
        rng=rng,
    )
    ground_truth["anomalous_centers"] = anomalous_center_indices

    # Mark students at anomalous centers
    for ctr_idx in anomalous_center_indices:
        ctr_id = center_ids[ctr_idx]
        if ctr_id in student_indices_by_center:
            for s_idx in student_indices_by_center[ctr_id]:
                fraud_labels[s_idx] = 1

    # Recompute scores and correct_matrix after fraud injection
    correct_matrix = (answers == answer_key[np.newaxis, :])
    scores = correct_matrix.sum(axis=1)

    # ── Step 8: Timing Data ──
    timing_data = None
    if include_timing:
        report_progress(60, "Generating response time data with KDE distributions")
        timing_data = _generate_timing_data(
            n_students=n_students,
            n_questions=n_questions,
            difficulties=difficulties,
            correct_matrix=correct_matrix,
            rng=rng,
        )

        # Pattern D: Timing Cheaters
        report_progress(65, "Injecting fraud Pattern D: impossibly fast response times")
        timing_cheater_indices = _inject_timing_fraud(
            timing_data=timing_data,
            leaked_indices=leaked_indices,
            leak_range=(44, 120),
            n_cheaters=89,
            rng=rng,
        )
        ground_truth["timing_cheaters"] = timing_cheater_indices.tolist()
        fraud_labels[timing_cheater_indices] = 1

    # ── Step 9: Question Text ──
    question_texts = None
    if include_question_text:
        report_progress(70, "Generating simulated question text")
        question_texts = _generate_question_texts(n_questions, rng)

    report_progress(80, "Finalizing generated data")

    # Summary stats
    total_fraud = fraud_labels.sum()
    logger.info(
        f"Generated {n_students:,} students with {total_fraud:,} fraud labels "
        f"({total_fraud/n_students*100:.1f}%)"
    )
    logger.info(
        f"  Copy rings: {sum(len(r['student_indices']) for r in ground_truth['copy_rings'])} students"
    )
    logger.info(f"  Paper leak: {len(ground_truth['leaked_students'])} students")
    logger.info(f"  Anomalous centers: {len(ground_truth['anomalous_centers'])} centers")
    if include_timing:
        logger.info(f"  Timing cheaters: {len(ground_truth.get('timing_cheaters', []))} students")

    report_progress(100, "Data generation complete")

    return GeneratedData(
        student_ids=student_ids,
        center_ids=center_ids,
        student_centers=student_center_map,
        answers=answers,
        answer_key=answer_key,
        correct_matrix=correct_matrix,
        scores=scores,
        abilities=abilities,
        difficulties=difficulties,
        discriminations=discriminations,
        center_metadata=center_metadata,
        question_texts=question_texts,
        timing_data=timing_data,
        ground_truth=ground_truth,
        fraud_labels=fraud_labels,
    )


# ══════════════════════════════════════════════════════════════
# Internal Helper Functions
# ══════════════════════════════════════════════════════════════


def _generate_center_sizes(
    n_students: int, n_centers: int, rng: np.random.RandomState
) -> list[int]:
    """Generate variable center sizes (50-300 students each)."""
    raw_sizes = rng.randint(50, 301, n_centers)
    # Scale to match total students
    total = raw_sizes.sum()
    scaled = (raw_sizes / total * n_students).astype(int)
    # Fix rounding difference
    diff = n_students - scaled.sum()
    for i in range(abs(diff)):
        if diff > 0:
            scaled[i % n_centers] += 1
        else:
            scaled[i % n_centers] -= 1
    return scaled.tolist()


def _windowed_shuffle(
    indices: np.ndarray, window: int, rng: np.random.RandomState
) -> np.ndarray:
    """Shuffle indices within local windows to preserve loose ordering."""
    result = indices.copy()
    n = len(indices)
    for i in range(0, n, window):
        end = min(i + window, n)
        rng.shuffle(result[i:end])
    return result


def _generate_center_metadata(
    center_ids: list[str],
    student_indices_by_center: dict[str, list[int]],
    scores: np.ndarray,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    """Generate center metadata with Indian city geolocation."""
    metadata = []
    n_cities = len(INDIAN_CITIES)

    for i, ctr_id in enumerate(center_ids):
        city_info = INDIAN_CITIES[i % n_cities].copy()
        # Add slight randomness to coordinates (different centers in same city)
        city_info["lat"] += rng.uniform(-0.1, 0.1)
        city_info["lon"] += rng.uniform(-0.1, 0.1)

        student_idxs = student_indices_by_center.get(ctr_id, [])
        center_scores = scores[student_idxs] if student_idxs else np.array([0])

        metadata.append({
            "center_id": ctr_id,
            "city": city_info["city"],
            "state": city_info["state"],
            "lat": round(city_info["lat"], 3),
            "lon": round(city_info["lon"], 3),
            "student_count": len(student_idxs),
            "mean_score": round(float(center_scores.mean()), 1) if len(student_idxs) > 0 else 0,
            "std_score": round(float(center_scores.std()), 1) if len(student_idxs) > 0 else 0,
        })

    return metadata


def _inject_copy_ring(
    answers: np.ndarray,
    answer_key: np.ndarray,
    student_indices_by_center: dict[str, list[int]],
    center_ids: list[str],
    config: dict,
    n_questions: int,
    n_options: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Inject a copy ring: students copy from a source with noise."""
    size = config["size"]
    overlap = config["overlap"]

    # Get students from specified center(s)
    if "center_idx" in config:
        ctr_id = center_ids[config["center_idx"]]
        pool = student_indices_by_center.get(ctr_id, [])
    elif "center_indices" in config:
        pool = []
        for ci in config["center_indices"]:
            ctr_id = center_ids[ci]
            pool.extend(student_indices_by_center.get(ctr_id, []))
    else:
        return []

    if len(pool) < size:
        size = max(3, len(pool) // 2)

    selected = rng.choice(pool, size=size, replace=False).tolist()

    if not selected:
        return []

    # Source student: the first one
    source_idx = selected[0]
    source_answers = answers[source_idx].copy()

    # Copy with noise
    for s_idx in selected[1:]:
        n_copy = int(n_questions * overlap)
        copy_questions = rng.choice(n_questions, size=n_copy, replace=False)
        answers[s_idx, copy_questions] = source_answers[copy_questions]

        # Add small noise: flip a few answers
        n_noise = int(n_questions * (1 - overlap) * 0.3)
        noise_questions = rng.choice(
            [q for q in range(n_questions) if q not in copy_questions],
            size=min(n_noise, n_questions - n_copy),
            replace=False,
        )
        for q in noise_questions:
            wrong_opts = [o for o in range(n_options) if o != answer_key[q]]
            answers[s_idx, q] = rng.choice(wrong_opts)

    return selected


def _inject_paper_leak(
    answers: np.ndarray,
    answer_key: np.ndarray,
    correct_matrix: np.ndarray,
    abilities: np.ndarray,
    n_students: int,
    n_leak_students: int,
    leak_range: tuple[int, int],
    accuracy_range: tuple[float, float],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Inject paper leak: students answer leaked questions with high accuracy."""
    # Select students from middle-to-low ability range (makes detection more obvious)
    # Real leak beneficiaries are not top students
    ability_order = np.argsort(abilities)
    lower_half = ability_order[:n_students // 2]
    leak_indices = rng.choice(lower_half, size=min(n_leak_students, len(lower_half)), replace=False)

    leak_start, leak_end = leak_range

    for s_idx in leak_indices:
        # For leaked questions, set high accuracy
        target_accuracy = rng.uniform(*accuracy_range)
        for q in range(leak_start, leak_end):
            if rng.random() < target_accuracy:
                answers[s_idx, q] = answer_key[q]

    return leak_indices


def _inject_center_anomalies(
    answers: np.ndarray,
    answer_key: np.ndarray,
    correct_matrix: np.ndarray,
    student_indices_by_center: dict[str, list[int]],
    center_ids: list[str],
    rng: np.random.RandomState,
) -> list[int]:
    """Inject 3 types of center anomalies.

    CTR_156: Score inflation (wrong → correct flips)
    CTR_203: Low diversity (force same wrong answers)
    CTR_042: Already has copy ring (Pattern A)
    """
    anomalous = []

    # Center 156: Score inflation — 30% of wrong answers flipped to correct
    if 156 < len(center_ids):
        ctr_id = center_ids[156]
        students = student_indices_by_center.get(ctr_id, [])
        for s_idx in students:
            wrong_qs = np.where(answers[s_idx] != answer_key)[0]
            n_flip = int(len(wrong_qs) * 0.30)
            if n_flip > 0:
                flip_qs = rng.choice(wrong_qs, size=n_flip, replace=False)
                answers[s_idx, flip_qs] = answer_key[flip_qs]
        anomalous.append(156)

    # Center 203: Low diversity — suppress wrong answer entropy
    if 203 < len(center_ids):
        ctr_id = center_ids[203]
        students = student_indices_by_center.get(ctr_id, [])
        if students:
            # Force all wrong answers to be the same option
            for q in range(answers.shape[1]):
                wrong_option = (answer_key[q] + 1) % answers.shape[1]
                for s_idx in students:
                    if answers[s_idx, q] != answer_key[q]:
                        answers[s_idx, q] = wrong_option
        anomalous.append(203)

    # Center 42: Already has copy ring from Pattern A
    anomalous.append(42)

    return anomalous


def _generate_timing_data(
    n_students: int,
    n_questions: int,
    difficulties: np.ndarray,
    correct_matrix: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate realistic response times based on question difficulty."""
    # Base time: harder questions take longer
    # Normalize difficulty to [0, 1]
    norm_diff = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min() + 1e-8)

    # Median time per question: 30-120 seconds, scaling with difficulty
    median_times = 30 + 90 * norm_diff  # Easy: ~30s, Hard: ~120s

    timing = np.zeros((n_students, n_questions))
    for q in range(n_questions):
        # Log-normal distribution centered on median time
        mu = np.log(median_times[q])
        sigma = 0.5
        timing[:, q] = rng.lognormal(mu, sigma, n_students)

        # Correct answers tend to be slightly faster
        correct_mask = correct_matrix[:, q]
        timing[correct_mask, q] *= rng.uniform(0.7, 0.9)

    # Clip to reasonable range (2-600 seconds)
    timing = np.clip(timing, 2, 600)

    return timing


def _inject_timing_fraud(
    timing_data: np.ndarray,
    leaked_indices: np.ndarray,
    leak_range: tuple[int, int],
    n_cheaters: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Inject impossibly fast response times for leaked questions."""
    # Select a subset of leaked students
    cheater_indices = rng.choice(
        leaked_indices,
        size=min(n_cheaters, len(leaked_indices)),
        replace=False,
    )

    leak_start, leak_end = leak_range
    for s_idx in cheater_indices:
        # Leaked questions answered in 2-8 seconds (vs 30-120 normal)
        for q in range(leak_start, leak_end):
            timing_data[s_idx, q] = rng.uniform(2, 8)

    return cheater_indices


def _generate_question_texts(n_questions: int, rng: np.random.RandomState) -> list[str]:
    """Generate simulated question texts for NLP similarity analysis."""
    questions = []
    all_templates = []
    for subject_group in QUESTION_SUBJECTS:
        for tmpl in subject_group["templates"]:
            all_templates.append((subject_group["subject"], tmpl))

    for i in range(n_questions):
        subject, template = all_templates[i % len(all_templates)]
        # Fill in random values
        text = template
        replacements = {
            "{m}": str(rng.randint(1, 100)),
            "{v}": str(rng.randint(1, 500)),
            "{l}": str(round(rng.uniform(0.1, 10.0), 1)),
            "{a}": str(round(rng.uniform(0.01, 10.0), 2)),
            "{r}": str(rng.randint(1, 1000)),
            "{d}": str(round(rng.uniform(0.1, 50.0), 1)),
            "{f}": str(rng.randint(5, 50)),
            "{u}": str(rng.randint(10, 100)),
            "{n1}": str(rng.randint(100, 1000)),
            "{n2}": str(rng.randint(100, 1000)),
            "{c}": str(round(rng.uniform(0.001, 1.0), 3)),
            "{h}": str(rng.randint(200, 36000)),
            "{D}": str(round(rng.uniform(0.5, 5.0), 1)),
            "{t}": str(rng.randint(1, 60)),
            "{g}": str(round(rng.uniform(1, 100), 1)),
            "{λ}": str(rng.choice([400, 450, 500, 550, 600, 650, 700])),
            "{compound}": rng.choice(["NaCl", "H₂SO₄", "CaCO₃", "KMnO₄", "NaOH"]),
            "{molecule}": rng.choice(["H₂O", "NH₃", "CH₄", "CO₂", "SF₆", "PCl₅"]),
            "{acid}": rng.choice(["HCl", "H₂SO₄", "HNO₃", "CH₃COOH"]),
            "{gas}": rng.choice(["O₂", "N₂", "CO₂", "H₂", "He"]),
            "{element}": rng.choice(["Mn", "Cr", "Fe", "Cu", "Ni"]),
            "{complex}": rng.choice(["[Fe(CN)₆]³⁻", "[Co(NH₃)₆]³⁺", "[Ni(CO)₄]"]),
            "{reactant1}": rng.choice(["Fe", "Zn", "Cu", "Al"]),
            "{reactant2}": rng.choice(["HCl", "H₂SO₄", "NaOH"]),
            "{product1}": rng.choice(["FeCl₂", "ZnSO₄", "CuCl₂"]),
            "{product2}": rng.choice(["H₂", "H₂O", "NaCl"]),
            "{anode}": rng.choice(["Zn/Zn²⁺", "Fe/Fe²⁺", "Cu/Cu²⁺"]),
            "{cathode}": rng.choice(["Cu²⁺/Cu", "Ag⁺/Ag", "Pt/H⁺"]),
            "{process}": rng.choice(["photosynthesis", "glycolysis", "oxidative phosphorylation", "transcription"]),
            "{organelle}": rng.choice(["mitochondria", "chloroplast", "ribosome", "nucleus"]),
            "{hormone}": rng.choice(["insulin", "thyroxine", "adrenaline", "estrogen"]),
            "{system}": rng.choice(["endocrine system", "nervous system", "circulatory system"]),
            "{structure}": rng.choice(["Golgi apparatus", "endoplasmic reticulum", "lysosome"]),
            "{event}": rng.choice(["chromosome condensation", "spindle formation", "cytokinesis"]),
            "{trait}": rng.choice(["color blindness", "hemophilia", "sickle cell anemia"]),
            "{vitamin}": rng.choice(["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin B12"]),
            "{function}": rng.choice(["balance", "memory", "vision", "breathing"]),
            "{organism}": rng.choice(["Amoeba", "Paramecium", "Euglena", "Plasmodium"]),
        }
        for key, val in replacements.items():
            text = text.replace(key, str(val))
        questions.append(f"Q{i+1} [{subject}]: {text}")

    return questions


def data_to_csv_bytes(data: GeneratedData) -> bytes:
    """Convert GeneratedData to CSV bytes for upload simulation."""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["student_id", "center_id"]
    for q in range(data.answers.shape[1]):
        header.append(f"q_{q+1}")
    if data.timing_data is not None:
        for q in range(data.answers.shape[1]):
            header.append(f"time_q_{q+1}")
    writer.writerow(header)

    # Data rows
    for i in range(len(data.student_ids)):
        row = [data.student_ids[i], data.student_centers[data.student_ids[i]]]
        row.extend(data.answers[i].tolist())
        if data.timing_data is not None:
            row.extend([round(t, 2) for t in data.timing_data[i].tolist()])
        writer.writerow(row)

    return output.getvalue().encode("utf-8")
