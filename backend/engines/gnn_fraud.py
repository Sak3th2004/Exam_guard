"""Engine 6: Graph Neural Network — Copy Ring Enhancer.

Architecture: GraphSAGE (2-layer) for node classification
Framework: PyTorch Geometric

Takes Engine 1's similarity graph, adds node features, and trains a GNN
to classify fraud vs clean nodes. Catches sophisticated fraud that
rule-based Louvain clustering misses.

Node features (6-dim):
  - total_correct, score_percentile
  - n_similar_neighbors, avg_waa
  - center_deviation, difficulty_gradient
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)


class GNNFraudEngine(BaseEngine):
    """GNN-based fraud detection using GraphSAGE."""

    def __init__(self):
        super().__init__(engine_name="gnn_copy_ring", requires_gpu=True)

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

        # Try importing PyTorch and PyG
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import SAGEConv
            from torch_geometric.data import Data
        except ImportError as e:
            logger.warning(f"PyTorch Geometric not available: {e}")
            return EngineOutput(
                engine_name=self.engine_name,
                status="failed",
                error_message=f"PyTorch Geometric not installed: {e}",
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"GNN engine using device: {device}")

        # Get copy ring graph data from kwargs
        copy_ring_data = kwargs.get("copy_ring_result", {})
        graph_data = copy_ring_data.get("graph_data", {"nodes": [], "edges": []})

        if not graph_data.get("edges"):
            logger.info("No E1 similarity graph — building k-NN graph from answer similarity")
            self.report_progress(10, "Building k-NN similarity graph from answer data")

            # Build a k-NN graph using answer similarity (top-k nearest neighbors)
            # Sample if too many students
            max_graph = min(n_students, 2000)
            if n_students > max_graph:
                sample_idx = np.random.choice(n_students, max_graph, replace=False)
            else:
                sample_idx = np.arange(n_students)

            sampled_answers = answers[sample_idx]
            # Compute pairwise Jaccard similarity using vectorized ops
            # Use binary comparison (same answer = 1)
            edges_src, edges_tgt = [], []
            n_sample = len(sample_idx)
            
            # Find top-k neighbors via batch dot product
            k_neighbors = min(10, n_sample - 1)
            for i in range(0, n_sample, 200):
                batch_end = min(i + 200, n_sample)
                batch = sampled_answers[i:batch_end]
                # Match matrix: how many questions have same answer
                matches = np.array([
                    (sampled_answers == batch[j:j+1]).sum(axis=1) / n_questions
                    for j in range(batch.shape[0])
                ])
                # For each student in batch, get top-k most similar
                for j in range(matches.shape[0]):
                    matches[j, i + j] = 0  # exclude self
                    top_k = np.argsort(-matches[j])[:k_neighbors]
                    for k_idx in top_k:
                        if matches[j, k_idx] > 0.5:
                            edges_src.extend([i + j, int(k_idx)])
                            edges_tgt.extend([int(k_idx), i + j])

            if not edges_src:
                return EngineOutput(
                    engine_name=self.engine_name,
                    status="complete",
                    result_data={"message": "No significant similarity found in answer data"},
                    summary={"model": "GraphSAGE", "flagged": 0, "device": str(device)},
                )

            # Build PyG-compatible graph data
            node_ids = [student_ids[idx] for idx in sample_idx]
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            n_graph_nodes = len(node_ids)
            edge_index_np = np.array([edges_src, edges_tgt])
            # Deduplicate
            unique_edges = np.unique(edge_index_np, axis=1)
            edge_index = torch.tensor(unique_edges, dtype=torch.long)
            graph_data = {"nodes": [{"id": nid} for nid in node_ids], "edges": []}
        else:
            self.report_progress(10, "Building PyG graph from E1 similarity data")

            # ── Build node mapping from E1 graph ──
            node_ids = list(set(
                [n["id"] for n in graph_data["nodes"]]
            ))
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            n_graph_nodes = len(node_ids)

            if n_graph_nodes < 10:
                return EngineOutput(
                    engine_name=self.engine_name,
                    status="complete",
                    result_data={"message": f"Graph too small ({n_graph_nodes} nodes)"},
                    summary={"model": "GraphSAGE", "flagged": 0, "device": str(device)},
                )

            # ── Build edge index ──
            edge_sources = []
            edge_targets = []
            for edge in graph_data["edges"]:
                if edge["source"] in id_to_idx and edge["target"] in id_to_idx:
                    s = id_to_idx[edge["source"]]
                    t = id_to_idx[edge["target"]]
                    edge_sources.extend([s, t])
                    edge_targets.extend([t, s])

            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # ── Build node features (6-dim) ──
        self.report_progress(20, "Computing 6-dimensional node features")

        if answer_key is not None:
            correct_matrix = (answers == answer_key[np.newaxis, :])
            scores = correct_matrix.sum(axis=1)
        else:
            scores = answers.sum(axis=1)

        national_mean = float(scores.mean())
        national_std = float(scores.std())
        percentiles = np.argsort(np.argsort(scores)) / n_students

        # Center means
        center_means = {}
        if center_ids:
            for i, cid in enumerate(center_ids):
                center_means.setdefault(cid, []).append(int(scores[i]))
            center_means = {k: np.mean(v) for k, v in center_means.items()}

        # Difficulty gradients (Q4 - Q1 accuracy)
        if answer_key is not None:
            q_difficulty = 1.0 - correct_matrix.mean(axis=0)
            quartile_bounds = np.percentile(q_difficulty, [25, 75])
            easy_qs = q_difficulty <= quartile_bounds[0]
            hard_qs = q_difficulty >= quartile_bounds[1]
            if easy_qs.sum() > 0 and hard_qs.sum() > 0:
                easy_acc = correct_matrix[:, easy_qs].mean(axis=1)
                hard_acc = correct_matrix[:, hard_qs].mean(axis=1)
                gradients = hard_acc - easy_acc
            else:
                gradients = np.zeros(n_students)
        else:
            gradients = np.zeros(n_students)

        # Build feature matrix for graph nodes
        features = np.zeros((n_graph_nodes, 6))
        student_id_to_idx = {sid: i for i, sid in enumerate(student_ids)}

        # Compute neighbor statistics from graph
        neighbor_counts = np.zeros(n_graph_nodes)
        neighbor_waa = np.zeros(n_graph_nodes)
        for node in graph_data["nodes"]:
            if node["id"] in id_to_idx:
                nidx = id_to_idx[node["id"]]
                # Count edges
                neighbor_counts[nidx] = sum(
                    1 for e in graph_data["edges"]
                    if e["source"] == node["id"] or e["target"] == node["id"]
                )
                # Average edge weight as WAA proxy
                weights = [
                    e["weight"] for e in graph_data["edges"]
                    if e["source"] == node["id"] or e["target"] == node["id"]
                ]
                neighbor_waa[nidx] = float(np.mean(weights)) if weights else 0

        for graph_idx, nid in enumerate(node_ids):
            if nid in student_id_to_idx:
                s_idx = student_id_to_idx[nid]
                features[graph_idx, 0] = scores[s_idx] / n_questions  # normalized score
                features[graph_idx, 1] = percentiles[s_idx]
                features[graph_idx, 2] = neighbor_counts[graph_idx] / max(neighbor_counts.max(), 1)
                features[graph_idx, 3] = neighbor_waa[graph_idx]
                if center_ids and s_idx < len(center_ids):
                    cid = center_ids[s_idx]
                    features[graph_idx, 4] = (scores[s_idx] - center_means.get(cid, national_mean)) / max(national_std, 1)
                features[graph_idx, 5] = gradients[s_idx]

        x = torch.tensor(features, dtype=torch.float32)

        # ── Labels (from ground truth) ──
        self.report_progress(30, "Setting up training labels from ground truth")

        labels = torch.zeros(n_graph_nodes, dtype=torch.long)
        if ground_truth:
            fraud_student_set = set()
            for ring in ground_truth.get("copy_rings", []):
                for s_idx in ring.get("student_indices", []):
                    if s_idx < len(student_ids):
                        fraud_student_set.add(student_ids[s_idx])
            for s_idx in ground_truth.get("leaked_students", []):
                if s_idx < len(student_ids):
                    fraud_student_set.add(student_ids[s_idx])

            for graph_idx, nid in enumerate(node_ids):
                if nid in fraud_student_set:
                    labels[graph_idx] = 1

        n_fraud = int((labels == 1).sum())
        n_clean = int((labels == 0).sum())
        logger.info(f"GNN labels: {n_fraud} fraud, {n_clean} clean in graph")

        if n_fraud == 0:
            # No ground truth labels — use unsupervised approach
            logger.info("No ground truth — using degree-based heuristic for labels")
            # High-degree nodes in similarity graph are likely fraud
            degree_threshold = np.percentile(neighbor_counts, 80)
            labels = torch.tensor(
                (neighbor_counts > degree_threshold).astype(int), dtype=torch.long
            )
            n_fraud = int((labels == 1).sum())
            n_clean = int((labels == 0).sum())

        # ── Train/val split ──
        n_nodes = n_graph_nodes
        perm = torch.randperm(n_nodes)
        train_size = int(0.7 * n_nodes)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[perm[:train_size]] = True
        val_mask[perm[train_size:]] = True

        # ── Build PyG Data ──
        data = Data(
            x=x,
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
        ).to(device)

        # ── Define GNN Model ──
        class FraudGNN(torch.nn.Module):
            def __init__(self, in_channels, hidden=64, out_channels=2):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden)
                self.conv2 = SAGEConv(hidden, hidden // 2)
                self.classifier = torch.nn.Linear(hidden // 2, out_channels)
                self.dropout = torch.nn.Dropout(0.3)

            def forward(self, x, edge_index):
                h = self.conv1(x, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
                h = self.conv2(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
                return F.log_softmax(self.classifier(h), dim=1)

        # ── Training ──
        self.report_progress(40, "Training GraphSAGE model on CUDA")

        model = FraudGNN(in_channels=6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        weight = torch.tensor([1.0, max(n_clean / max(n_fraud, 1), 1.0)]).to(device)
        criterion = torch.nn.NLLLoss(weight=weight)

        best_f1 = 0.0
        best_state = None
        patience = 15
        wait = 0
        final_epoch = 0

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_pred = pred[data.val_mask]
                val_true = data.y[data.val_mask]
                tp = ((val_pred == 1) & (val_true == 1)).sum().float()
                fp = ((val_pred == 1) & (val_true == 0)).sum().float()
                fn = ((val_pred == 0) & (val_true == 1)).sum().float()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        final_epoch = epoch
                        break

            if epoch % 10 == 0:
                pct = 40 + int(40 * epoch / 100)
                self.report_progress(pct, f"GNN training epoch {epoch}/100 — F1: {f1:.3f}")

            final_epoch = epoch

        # ── Inference ──
        self.report_progress(85, "Running GNN inference on all nodes")

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        model.eval()
        with torch.no_grad():
            probs = torch.exp(model(data.x, data.edge_index))
            fraud_probs = probs[:, 1].cpu().numpy()

        # Flag students with fraud probability > 0.5
        flagged_mask = fraud_probs > 0.5
        flagged_ids = [node_ids[i] for i in range(n_graph_nodes) if flagged_mask[i]]

        # Find novel detections (flagged by GNN but not in copy ring clusters)
        copy_ring_flagged = set()
        for cluster in copy_ring_data.get("clusters", []):
            copy_ring_flagged.update(cluster.get("students", []))
        novel_detections = [fid for fid in flagged_ids if fid not in copy_ring_flagged]

        # Build enhanced graph data
        enhanced_nodes = []
        for i, nid in enumerate(node_ids):
            enhanced_nodes.append({
                "id": nid,
                "label": nid,
                "fraud_prob": round(float(fraud_probs[i]), 3),
                "is_flagged": bool(flagged_mask[i]),
                "features": {
                    "score": round(float(features[i, 0]), 3),
                    "percentile": round(float(features[i, 1]), 3),
                    "neighbors": round(float(features[i, 2]), 3),
                    "avg_waa": round(float(features[i, 3]), 3),
                },
            })

        fraud_probs_dict = {
            node_ids[i]: round(float(fraud_probs[i]), 4)
            for i in range(n_graph_nodes)
        }

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={
                "model": f"GraphSAGE (2-layer, 64→32→2)",
                "training_epochs": final_epoch + 1,
                "validation_f1": round(best_f1, 3),
                "total_flagged": len(flagged_ids),
                "novel_detections": len(novel_detections),
                "fraud_probabilities": fraud_probs_dict,
                "enhanced_graph_data": {
                    "nodes": enhanced_nodes,
                    "edges": graph_data["edges"],
                },
                "device": str(device),
            },
            summary={
                "model": "GraphSAGE",
                "flagged": len(flagged_ids),
                "novel_detections": len(novel_detections),
                "validation_f1": round(best_f1, 3),
                "training_epochs": final_epoch + 1,
                "device": str(device),
            },
        )
