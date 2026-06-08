"""Engine 1: Copy Ring Detection.

Algorithms: MinHash LSH → Jaccard + Wrong Answer Agreement → Louvain Community Detection
Finds clusters of students who copied from each other.

Pipeline:
  1. Create MinHash signatures per student (num_perm=128)
  2. LSH index with threshold=0.7 → candidate similar pairs
  3. Exact Jaccard similarity on candidates
  4. Wrong Answer Agreement (WAA) — THE KEY METRIC
  5. Combined score: 0.4*Jaccard + 0.6*WAA
  6. Graph construction (edges where score > 0.75)
  7. Louvain community detection → clusters ≥ 3 members
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from datasketch import MinHash, MinHashLSH
import networkx as nx
from community import community_louvain

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)

# Constants from config
LSH_THRESHOLD = 0.7
LSH_NUM_PERM = 128
SIMILARITY_EDGE_THRESHOLD = 0.75
WAA_WEIGHT = 0.6
JACCARD_WEIGHT = 0.4
MIN_CLUSTER_SIZE = 3


class CopyRingEngine(BaseEngine):
    """Detect copy rings using MinHash LSH and Louvain clustering."""

    def __init__(self):
        super().__init__(engine_name="copy_ring", requires_gpu=False)

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

        # ── Step 1: Create MinHash signatures ──
        self.report_progress(10, "Creating MinHash signatures for all students")

        signatures: dict[int, MinHash] = {}
        for i in range(n_students):
            mh = MinHash(num_perm=LSH_NUM_PERM)
            for q_idx in range(n_questions):
                token = f"{q_idx}_{answers[i, q_idx]}"
                mh.update(token.encode("utf8"))
            signatures[i] = mh

            if i % 10000 == 0 and i > 0:
                pct = 10 + int(20 * i / n_students)
                self.report_progress(pct, f"MinHash signatures: {i:,}/{n_students:,}")

        # ── Step 2: LSH index → candidate pairs ──
        self.report_progress(30, "Building LSH index for approximate similarity search")

        lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)
        for i, mh in signatures.items():
            try:
                lsh.insert(str(i), mh)
            except ValueError:
                pass  # Duplicate key (shouldn't happen)

        self.report_progress(40, "Querying LSH for candidate similar pairs")

        candidate_pairs: set[tuple[int, int]] = set()
        for i in range(n_students):
            results = lsh.query(signatures[i])
            for r in results:
                j = int(r)
                if i < j:
                    candidate_pairs.add((i, j))

            if i % 10000 == 0 and i > 0:
                pct = 40 + int(15 * i / n_students)
                self.report_progress(pct, f"LSH query: {i:,}/{n_students:,} — {len(candidate_pairs):,} candidates")

        logger.info(f"LSH produced {len(candidate_pairs):,} candidate pairs")

        # ── Step 3 & 4: Exact Similarity on Candidates ──
        self.report_progress(55, f"Computing exact similarity on {len(candidate_pairs):,} candidate pairs")

        # Precompute answer sets and wrong answer sets
        correct_mask = None
        if answer_key is not None:
            correct_mask = answers == answer_key[np.newaxis, :]

        edges: list[dict[str, Any]] = []
        total_pairs = len(candidate_pairs)

        for idx, (i, j) in enumerate(candidate_pairs):
            a_i = answers[i]
            a_j = answers[j]

            # Jaccard similarity
            matches = (a_i == a_j)
            jaccard = matches.sum() / n_questions

            # Wrong Answer Agreement
            waa = 0.0
            if answer_key is not None:
                wrong_i = np.where(a_i != answer_key)[0]
                wrong_j = np.where(a_j != answer_key)[0]
                wrong_union = np.union1d(wrong_i, wrong_j)
                if len(wrong_union) > 0:
                    wrong_match = 0
                    for q in wrong_union:
                        if a_i[q] == a_j[q] and a_i[q] != answer_key[q]:
                            wrong_match += 1
                    waa = wrong_match / len(wrong_union)
            else:
                # Without answer key, use all-answer matching as proxy
                waa = jaccard

            # Combined score
            combined = JACCARD_WEIGHT * jaccard + WAA_WEIGHT * waa

            if combined > SIMILARITY_EDGE_THRESHOLD:
                edges.append({
                    "i": i, "j": j,
                    "jaccard": float(jaccard),
                    "waa": float(waa),
                    "combined": float(combined),
                })

            if idx % 5000 == 0 and idx > 0:
                pct = 55 + int(20 * idx / max(total_pairs, 1))
                self.report_progress(pct, f"Similarity: {idx:,}/{total_pairs:,} pairs")

        logger.info(f"Found {len(edges)} edges above threshold {SIMILARITY_EDGE_THRESHOLD}")

        # ── Step 5: Build graph ──
        self.report_progress(75, "Building similarity graph")

        G = nx.Graph()
        involved_students: set[int] = set()

        for edge in edges:
            i, j = edge["i"], edge["j"]
            G.add_edge(i, j, weight=edge["combined"])
            involved_students.add(i)
            involved_students.add(j)

        # ── Step 6: Louvain community detection ──
        self.report_progress(80, "Running Louvain community detection")

        clusters: list[dict[str, Any]] = []
        flagged_ids: list[str] = []

        if len(G.nodes()) > 0:
            partition = community_louvain.best_partition(G, weight="weight", random_state=42)

            # Group by community
            communities: dict[int, list[int]] = {}
            for node, comm_id in partition.items():
                communities.setdefault(comm_id, []).append(node)

            # Filter clusters with >= MIN_CLUSTER_SIZE
            cluster_id = 0
            for comm_id, members in communities.items():
                if len(members) >= MIN_CLUSTER_SIZE:
                    cluster_id += 1

                    # Compute cluster statistics
                    member_ids = [student_ids[m] for m in members]
                    member_centers = [center_ids[m] for m in members] if center_ids else []

                    # Internal similarity
                    internal_sims = []
                    internal_waas = []
                    matching_wrong = 0
                    for edge in edges:
                        if edge["i"] in members and edge["j"] in members:
                            internal_sims.append(edge["combined"])
                            internal_waas.append(edge["waa"])
                            if answer_key is not None:
                                a_i = answers[edge["i"]]
                                a_j = answers[edge["j"]]
                                for q in range(n_questions):
                                    if a_i[q] == a_j[q] and a_i[q] != answer_key[q]:
                                        matching_wrong += 1

                    avg_sim = float(np.mean(internal_sims)) if internal_sims else 0.0
                    avg_waa = float(np.mean(internal_waas)) if internal_waas else 0.0

                    # Check center overlap
                    unique_member_centers = list(set(member_centers)) if member_centers else []
                    same_center = len(unique_member_centers) <= 2

                    # Confidence based on cluster quality
                    confidence = min(0.99, avg_sim * 0.6 + avg_waa * 0.4)

                    # Compute p_chance (rough estimate)
                    p_chance = (0.25 ** max(1, matching_wrong // len(members))) if matching_wrong > 0 else 1.0

                    clusters.append({
                        "cluster_id": cluster_id,
                        "size": len(members),
                        "avg_similarity": round(avg_sim, 3),
                        "avg_waa": round(avg_waa, 3),
                        "center_ids": unique_member_centers,
                        "same_center": same_center,
                        "confidence": round(confidence, 3),
                        "students": member_ids,
                        "student_indices": members,
                        "evidence": {
                            "matching_wrong_answers": matching_wrong,
                            "total_questions": n_questions,
                            "p_chance": f"{p_chance:.2e}",
                        },
                    })
                    flagged_ids.extend(member_ids)

        # Sort clusters by size
        clusters.sort(key=lambda c: c["size"], reverse=True)

        # ── Build graph data for visualization ──
        self.report_progress(90, "Preparing graph visualization data")

        graph_nodes = []
        graph_edges = []

        for student_idx in involved_students:
            sid = student_ids[student_idx]
            cluster_assignment = None
            for c in clusters:
                if sid in c["students"]:
                    cluster_assignment = c["cluster_id"]
                    break

            graph_nodes.append({
                "id": sid,
                "label": sid,
                "cluster": cluster_assignment,
                "center_id": center_ids[student_idx] if center_ids else None,
                "score": int(answers[student_idx].sum()) if answer_key is None else int((answers[student_idx] == answer_key).sum()),
            })

        for edge in edges:
            graph_edges.append({
                "source": student_ids[edge["i"]],
                "target": student_ids[edge["j"]],
                "weight": round(edge["combined"], 3),
            })

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(set(flagged_ids)),
            flagged_student_ids=list(set(flagged_ids)),
            result_data={
                "clusters_found": len(clusters),
                "total_flagged_students": len(set(flagged_ids)),
                "clusters": clusters,
                "graph_data": {"nodes": graph_nodes, "edges": graph_edges},
                "total_candidate_pairs": len(candidate_pairs),
                "total_edges": len(edges),
            },
            summary={
                "clusters_found": len(clusters),
                "flagged": len(set(flagged_ids)),
                "largest_cluster": max((c["size"] for c in clusters), default=0),
                "avg_waa": round(float(np.mean([c["avg_waa"] for c in clusters])), 3) if clusters else 0,
            },
        )
