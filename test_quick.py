"""Quick test of the data generator."""
import sys
sys.path.insert(0, "backend")

from data.generator import generate_exam_data

print("Testing generator with 1000 students, 50 questions, 10 centers...")
d = generate_exam_data(n_students=1000, n_questions=50, n_centers=10)
print(f"  Students: {len(d.student_ids)}")
print(f"  Fraud labels: {d.fraud_labels.sum()}")
print(f"  Copy rings: {len(d.ground_truth['copy_rings'])}")
print(f"  Leaked students: {len(d.ground_truth['leaked_students'])}")
print(f"  Anomalous centers: {len(d.ground_truth['anomalous_centers'])}")
print(f"  Timing data shape: {d.timing_data.shape if d.timing_data is not None else 'None'}")
print(f"  Question texts: {len(d.question_texts) if d.question_texts else 0}")
print("Generator OK!")
