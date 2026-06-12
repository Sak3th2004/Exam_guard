"""ExamGuard Full End-to-End Integration Test.

Runs the complete pipeline:
1. Generate synthetic data (IRT 2PL + fraud patterns)
2. Run all 8 engines + XGBoost ensemble
3. Validate results
4. Compute accuracy metrics against ground truth
5. Print benchmark report

Usage: python tests/test_e2e.py
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np


async def run_e2e():
    print("=" * 70)
    print("  ExamGuard v2.0 — Full End-to-End Integration Test")
    print("=" * 70)

    # ── Step 1: Generate Data ──
    print("\n[1/5] Generating synthetic exam data...")
    t0 = time.time()

    from data.generator import generate_exam_data
    data = generate_exam_data(
        n_students=1000,
        n_questions=50,
        n_centers=10,
        n_options=4,
        include_timing=True,
        include_question_text=True,
        seed=42,
    )

    gen_time = time.time() - t0
    print(f"  ✓ Generated {len(data.student_ids)} students × {data.answers.shape[1]} questions")
    print(f"  ✓ Fraud labels: {data.fraud_labels.sum()} / {len(data.fraud_labels)} ({data.fraud_labels.mean()*100:.1f}%)")
    print(f"  ✓ Copy rings: {len(data.ground_truth.get('copy_rings', []))}")
    print(f"  ✓ Leaked students: {len(data.ground_truth.get('leaked_students', []))}")
    print(f"  ✓ Anomalous centers: {len(data.ground_truth.get('anomalous_centers', []))}")
    print(f"  ✓ Timing cheaters: {len(data.ground_truth.get('timing_cheaters', []))}")
    print(f"  ⏱ Data generation: {gen_time:.2f}s")

    # Prepare center_ids list
    center_ids = [data.student_centers[s] for s in data.student_ids]

    # ── Step 2: Run Classical Engines (CPU) ──
    print("\n[2/5] Running classical CPU engines (E1-E5)...")
    t1 = time.time()

    from engines.copy_ring import CopyRingEngine
    from engines.stat_impossibility import StatImpossibilityEngine
    from engines.center_anomaly import CenterAnomalyEngine
    from engines.leak_signature import LeakSignatureEngine
    from engines.response_time import ResponseTimeEngine

    cpu_engines = [
        ("E1 Copy Ring", CopyRingEngine()),
        ("E2 Stat Impossibility", StatImpossibilityEngine()),
        ("E3 Center Anomaly", CenterAnomalyEngine()),
        ("E4 Leak Signature", LeakSignatureEngine()),
        ("E5 Response Time", ResponseTimeEngine()),
    ]

    engine_results = {}
    for name, engine in cpu_engines:
        t_eng = time.time()
        result = await engine.analyze(
            answers=data.answers,
            answer_key=data.answer_key,
            student_ids=data.student_ids,
            center_ids=center_ids,
            timing_data=data.timing_data,
            question_texts=data.question_texts,
            ground_truth=data.ground_truth,
        )
        dt = time.time() - t_eng
        engine_results[engine.engine_name] = result
        status_icon = "✓" if result.status == "complete" else "✗"
        print(f"  {status_icon} {name}: {result.flagged_count} flagged, {dt:.2f}s [{result.status}]")

    cpu_time = time.time() - t1

    # ── Step 3: Run GPU Engines (E6-E8) ──
    print("\n[3/5] Running GPU-accelerated engines (E6-E8)...")
    t2 = time.time()

    from engines.gnn_fraud import GNNFraudEngine
    from engines.vae_anomaly import VAEAnomalyEngine
    from engines.question_similarity import QuestionSimilarityEngine

    gpu_engines = [
        ("E6 GNN (GraphSAGE)", GNNFraudEngine()),
        ("E7 VAE (Autoencoder)", VAEAnomalyEngine()),
        ("E8 NLP (Sentence Transformer)", QuestionSimilarityEngine()),
    ]

    e1_result = engine_results.get("copy_ring")
    copy_ring_result = {}
    if e1_result and e1_result.result_data:
        copy_ring_result = {
            "graph_data": e1_result.result_data.get("graph_data", {"nodes": [], "edges": []}),
            "clusters": e1_result.result_data.get("clusters", []),
        }

    for name, engine in gpu_engines:
        t_eng = time.time()
        extra_kwargs = {}
        if engine.engine_name == "gnn_copy_ring":
            extra_kwargs["copy_ring_result"] = copy_ring_result
        result = await engine.analyze(
            answers=data.answers,
            answer_key=data.answer_key,
            student_ids=data.student_ids,
            center_ids=center_ids,
            timing_data=data.timing_data,
            question_texts=data.question_texts,
            ground_truth=data.ground_truth,
            **extra_kwargs,
        )
        dt = time.time() - t_eng
        engine_results[engine.engine_name] = result
        device = result.result_data.get("device", "cpu") if result.result_data else "cpu"
        status_icon = "✓" if result.status == "complete" else "✗"
        print(f"  {status_icon} {name}: {result.flagged_count} flagged, {dt:.2f}s [{device}]")

    gpu_time = time.time() - t2

    # ── Step 4: Run XGBoost Ensemble ──
    print("\n[4/5] Running XGBoost meta-ensemble...")
    t3 = time.time()

    from engines.xgboost_ensemble import XGBoostEnsembleEngine

    ensemble = XGBoostEnsembleEngine()
    ensemble_result = await ensemble.analyze(
        answers=data.answers,
        answer_key=data.answer_key,
        student_ids=data.student_ids,
        center_ids=center_ids,
        timing_data=data.timing_data,
        ground_truth=data.ground_truth,
        engine_results=engine_results,
    )
    engine_results["xgboost_ensemble"] = ensemble_result
    ens_time = time.time() - t3

    status_icon = "✓" if ensemble_result.status == "complete" else "✗"
    print(f"  {status_icon} XGBoost: {ensemble_result.flagged_count} flagged, {ens_time:.2f}s [{ensemble_result.status}]")

    # ── Step 5: Compute Accuracy ──
    print("\n[5/5] Computing accuracy benchmarks...")

    rankings = ensemble_result.result_data.get("final_rankings", [])
    gt_labels = data.fraud_labels

    if rankings:
        # Build prediction array from rankings
        student_to_prob = {r["student_id"]: r["fraud_probability"] for r in rankings}
        y_true = gt_labels
        y_pred_prob = np.array([student_to_prob.get(sid, 0.0) for sid in data.student_ids])
        y_pred = (y_pred_prob >= 0.5).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        accuracy = (tp + tn) / len(y_true)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        try:
            from sklearn.metrics import roc_auc_score
            auc_roc = roc_auc_score(y_true, y_pred_prob)
        except Exception:
            auc_roc = 0.0
    else:
        # Fallback: use engine flags
        flagged_ids = set()
        for r in engine_results.values():
            flagged_ids.update(r.flagged_student_ids)

        y_pred = np.array([1 if sid in flagged_ids else 0 for sid in data.student_ids])
        y_true = gt_labels

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        accuracy = (tp + tn) / len(y_true)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        auc_roc = 0.0

    total_time = time.time() - t0

    # ── Report ──
    print("\n" + "=" * 70)
    print("  BENCHMARK REPORT")
    print("=" * 70)
    print(f"""
  Dataset
  ├─ Students:     {len(data.student_ids):,}
  ├─ Questions:    {data.answers.shape[1]}
  ├─ Centers:      {len(data.center_ids)}
  ├─ Fraud Rate:   {gt_labels.mean()*100:.1f}%
  └─ Ground Truth: {int(gt_labels.sum())} fraudulent / {len(gt_labels)} total

  Confusion Matrix
  ├─ True Pos:     {tp}
  ├─ False Pos:    {fp}
  ├─ False Neg:    {fn}
  └─ True Neg:     {tn}

  Accuracy Metrics
  ├─ Accuracy:     {accuracy*100:.2f}%
  ├─ Precision:    {precision*100:.2f}%
  ├─ Recall:       {recall*100:.2f}%
  ├─ F1 Score:     {f1*100:.2f}%
  └─ AUC-ROC:      {auc_roc*100:.2f}%

  Timing
  ├─ Data Gen:     {gen_time:.2f}s
  ├─ CPU Engines:  {cpu_time:.2f}s
  ├─ GPU Engines:  {gpu_time:.2f}s
  ├─ Ensemble:     {ens_time:.2f}s
  └─ Total:        {total_time:.2f}s

  Engines ({len(engine_results)} total)""")

    for name, result in engine_results.items():
        dev = result.result_data.get("device", "cpu") if result.result_data else "cpu"
        print(f"  ├─ {name:25s}  {result.flagged_count:4d} flagged  [{result.status}] ({dev})")

    print(f"""
  Result: {'PASS ✓' if f1 > 0.5 else 'NEEDS IMPROVEMENT ✗'}
""")
    print("=" * 70)

    return f1 > 0.5


if __name__ == "__main__":
    success = asyncio.run(run_e2e())
    sys.exit(0 if success else 1)
