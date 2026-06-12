"""Engine 6: Graph Neural Network — Copy Ring Enhancer.

Architecture: GraphSAGE (2-layer) for node classification
Framework: PyTorch Geometric

Takes Engine 1's similarity graph OR builds its own k-NN graph,
then trains a GNN to classify fraud vs clean nodes.

Node features (8-dim):
  - normalized_score, score_percentile
  - n_neighbors (from graph), avg_neighbor_sim
  - center_deviation, difficulty_gradient
  - wrong_answer_concentration, speed_ratio (if timing)
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
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore
            from torch_geometric.nn import SAGEConv  # type: ignore[import-not-found]
            from torch_geometric.data import Data  # type: ignore[import-not-found]
        except ImportError as e:
            logger.warning(f"PyTorch Geometric not available: {e}")
            return EngineOutput(
                engine_name=self.engine_name,
                status="failed",
                error_message=f"PyTorch Geometric not installed: {e}",
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"GNN engine using device: {device}")

        # ── STEP 1: Build similarity graph ──
        self.report_progress(5, "Building answer-similarity graph")

        # Always build our own k-NN graph from raw answer data for reliability
        max_graph = min(n_students, 2000)
        rng = np.random.RandomState(42)
        if n_students > max_graph:
            sample_idx = rng.choice(n_students, max_graph, replace=False)
        else:
            sample_idx = np.arange(n_students)

        sampled_answers = answers[sample_idx]
        n_sample = len(sample_idx)
        node_ids = [student_ids[idx] for idx in sample_idx]

        # Compute pairwise answer-matching in batches
        edges_src, edges_tgt, edge_weights = [], [], []
        batch_size = 100
        k_neighbors = min(20, n_sample - 1)

        self.report_progress(10, f"Computing k-NN similarity for {n_sample} nodes")

        for i in range(0, n_sample, batch_size):
            batch_end = min(i + batch_size, n_sample)
            batch = sampled_answers[i:batch_end]
            # Vectorized: (batch_size, n_sample) match fractions
            match_frac = np.array([
                (sampled_answers == batch[j:j+1]).sum(axis=1) / n_questions
                for j in range(batch.shape[0])
            ])  # shape: (batch_size, n_sample)

            for j in range(match_frac.shape[0]):
                global_j = i + j
                match_frac[j, global_j] = 0  # exclude self
                top_k = np.argsort(-match_frac[j])[:k_neighbors]
                for k_idx in top_k:
                    sim = match_frac[j, int(k_idx)]
                    if sim > 0.30:  # Low threshold to ensure dense graph
                        edges_src.append(global_j)
                        edges_tgt.append(int(k_idx))
                        edge_weights.append(sim)

            if i % 500 == 0 and i > 0:
                pct = 10 + int(20 * i / n_sample)
                self.report_progress(pct, f"Graph construction: {i}/{n_sample}")

        if len(edges_src) < 20:
            return EngineOutput(
                engine_name=self.engine_name,
                status="complete",
                flagged_count=0,
                result_data={"message": "Insufficient similarity edges for GNN training"},
                summary={"model": "GraphSAGE", "flagged": 0, "device": str(device)},
            )

        # Make undirected + deduplicate
        all_src = edges_src + edges_tgt
        all_tgt = edges_tgt + edges_src
        all_wt = edge_weights + edge_weights
        edge_index = torch.tensor([all_src, all_tgt], dtype=torch.long)

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n_graph_nodes = len(node_ids)

        logger.info(f"GNN graph: {n_graph_nodes} nodes, {edge_index.shape[1]} edges")

        # ── STEP 2: Compute 8-dim node features ──
        self.report_progress(30, "Computing 8-dimensional node features")

        if answer_key is not None:
            correct_matrix = (answers == answer_key[np.newaxis, :])
            scores = correct_matrix.sum(axis=1)
        else:
            scores = answers.sum(axis=1)

        national_mean = float(scores.mean())
        national_std = max(float(scores.std()), 1e-6)
        percentiles = np.argsort(np.argsort(scores)) / n_students

        # Center means
        center_means = {}
        if center_ids:
            for i, cid in enumerate(center_ids):
                center_means.setdefault(cid, []).append(float(scores[i]))
            center_means = {k: np.mean(v) for k, v in center_means.items()}

        # Difficulty gradient per student (hard_acc - easy_acc)
        gradients = np.zeros(n_students)
        if answer_key is not None:
            q_difficulty = 1.0 - correct_matrix.mean(axis=0)
            q25, q75 = np.percentile(q_difficulty, [25, 75])
            easy_mask = q_difficulty <= q25
            hard_mask = q_difficulty >= q75
            if easy_mask.sum() > 0 and hard_mask.sum() > 0:
                gradients = correct_matrix[:, hard_mask].mean(axis=1) - correct_matrix[:, easy_mask].mean(axis=1)

        # Wrong answer concentration per student
        wrong_concentration = np.zeros(n_students)
        if answer_key is not None:
            for i in range(n_students):
                wrong_mask = answers[i] != answer_key
                if wrong_mask.sum() > 0:
                    wrong_ans = answers[i, wrong_mask]
                    _, counts = np.unique(wrong_ans, return_counts=True)
                    # Herfindahl index of wrong answers (high = concentrated = suspicious)
                    freqs = counts / counts.sum()
                    wrong_concentration[i] = float(np.sum(freqs ** 2))

        # Speed ratio (if timing data)
        speed_ratios = np.zeros(n_students)
        if timing_data is not None:
            median_times = np.median(timing_data, axis=0)
            student_speed = (timing_data / (median_times[np.newaxis, :] + 1e-8)).mean(axis=1)
            speed_ratios = np.clip(student_speed, 0, 5)

        # Neighbor degree from our edge list
        degree = np.zeros(n_graph_nodes)
        avg_neighbor_sim = np.zeros(n_graph_nodes)
        for s, t, w in zip(edges_src, edges_tgt, edge_weights):
            degree[s] += 1
            avg_neighbor_sim[s] += w
        for i in range(n_graph_nodes):
            if degree[i] > 0:
                avg_neighbor_sim[i] /= degree[i]

        # Build feature matrix (8 features)
        features = np.zeros((n_graph_nodes, 8))
        student_id_to_idx = {sid: i for i, sid in enumerate(student_ids)}

        for graph_idx, nid in enumerate(node_ids):
            if nid in student_id_to_idx:
                s_idx = student_id_to_idx[nid]
                features[graph_idx, 0] = scores[s_idx] / n_questions
                features[graph_idx, 1] = percentiles[s_idx]
                features[graph_idx, 2] = degree[graph_idx] / max(degree.max(), 1)
                features[graph_idx, 3] = avg_neighbor_sim[graph_idx]
                if center_ids and s_idx < len(center_ids):
                    cid = center_ids[s_idx]
                    features[graph_idx, 4] = (scores[s_idx] - center_means.get(cid, national_mean)) / national_std
                features[graph_idx, 5] = gradients[s_idx]
                features[graph_idx, 6] = wrong_concentration[s_idx]
                features[graph_idx, 7] = speed_ratios[s_idx]

        x = torch.tensor(features, dtype=torch.float32)

        # ── STEP 3: Labels (from ground truth or semi-supervised) ──
        self.report_progress(35, "Setting up training labels")

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
            for s_idx in ground_truth.get("timing_cheaters", []):
                if s_idx < len(student_ids):
                    fraud_student_set.add(student_ids[s_idx])

            for graph_idx, nid in enumerate(node_ids):
                if nid in fraud_student_set:
                    labels[graph_idx] = 1

        n_fraud = int((labels == 1).sum())
        n_clean = int((labels == 0).sum())
        logger.info(f"GNN labels: {n_fraud} fraud, {n_clean} clean in graph")

        if n_fraud == 0:
            # Semi-supervised: use high-degree + high-similarity as pseudo-labels
            logger.info("No ground truth labels — using semi-supervised heuristic")
            combined_score = (
                0.4 * (degree / max(degree.max(), 1)) +
                0.3 * avg_neighbor_sim +
                0.3 * (1 - features[:, 0])  # low score = more suspicious
            )
            threshold = np.percentile(combined_score, 85)
            labels = torch.tensor((combined_score > threshold).astype(int), dtype=torch.long)
            n_fraud = int((labels == 1).sum())
            n_clean = int((labels == 0).sum())

        if n_fraud < 3:
            return EngineOutput(
                engine_name=self.engine_name,
                status="complete",
                flagged_count=0,
                result_data={"message": f"Too few fraud labels ({n_fraud}) for training"},
                summary={"model": "GraphSAGE", "flagged": 0, "device": str(device)},
            )

        # ── STEP 4: Train/val split ──
        perm = torch.randperm(n_graph_nodes)
        train_size = int(0.7 * n_graph_nodes)
        train_mask = torch.zeros(n_graph_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_graph_nodes, dtype=torch.bool)
        train_mask[perm[:train_size]] = True
        val_mask[perm[train_size:]] = True

        data = Data(
            x=x, edge_index=edge_index, y=labels,
            train_mask=train_mask, val_mask=val_mask,
        ).to(device)

        # ── STEP 5: Define and train GNN ──
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

        self.report_progress(40, f"Training GraphSAGE on {device}")

        model = FraudGNN(in_channels=8).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Class weights to handle imbalance
        weight = torch.tensor([1.0, max(n_clean / max(n_fraud, 1), 2.0)]).to(device)
        criterion = torch.nn.NLLLoss(weight=weight)

        best_f1 = 0.0
        best_state = None
        patience = 20
        wait = 0
        final_epoch = 0

        for epoch in range(150):
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

            if epoch % 15 == 0:
                pct = 40 + int(40 * epoch / 150)
                self.report_progress(pct, f"GNN epoch {epoch}/150 — F1: {f1:.3f}, loss: {loss:.3f}")

            final_epoch = epoch

        # ── STEP 6: Inference ──
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

        # Find novel detections (not in copy ring)
        copy_ring_data = kwargs.get("copy_ring_result", {})
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
                    "degree": round(float(features[i, 2]), 3),
                    "avg_sim": round(float(features[i, 3]), 3),
                },
            })

        fraud_probs_dict = {
            node_ids[i]: round(float(fraud_probs[i]), 4)
            for i in range(n_graph_nodes)
        }

        self.report_progress(100, f"GNN complete — {len(flagged_ids)} flagged, F1: {best_f1:.3f}")

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={
                "model": "GraphSAGE (2-layer, 64→32→2, 8-dim features)",
                "training_epochs": final_epoch + 1,
                "validation_f1": round(best_f1, 3),
                "total_flagged": len(flagged_ids),
                "novel_detections": len(novel_detections),
                "graph_stats": {
                    "nodes": n_graph_nodes,
                    "edges": edge_index.shape[1],
                    "avg_degree": round(float(degree.mean()), 1),
                },
                "fraud_probabilities": fraud_probs_dict,
                "enhanced_graph_data": {
                    "nodes": enhanced_nodes[:500],
                    "edges": [{"source": node_ids[s], "target": node_ids[t]}
                              for s, t in zip(edges_src[:1000], edges_tgt[:1000])],
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
