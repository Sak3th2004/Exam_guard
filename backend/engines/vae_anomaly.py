"""Engine 7: Variational Autoencoder — Novel Anomaly Detector.

Architecture: VAE with 200×4→512→256→128→32 (latent) encoder
Framework: PyTorch

Catches UNKNOWN fraud patterns by learning what "normal" exam answers
look like. Students with unusual patterns → high reconstruction error.

Bonus: t-SNE on 32-dim latent space for visualization.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from engines.base_engine import BaseEngine, EngineOutput

logger = logging.getLogger(__name__)


class VAEAnomalyEngine(BaseEngine):
    """VAE-based anomaly detection for unknown fraud patterns."""

    def __init__(self):
        super().__init__(engine_name="vae_anomaly", requires_gpu=True)

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
        n_options = int(answers.max()) + 1

        if student_ids is None:
            student_ids = [f"STU_{i:06d}" for i in range(n_students)]

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            return EngineOutput(
                engine_name=self.engine_name,
                status="failed",
                error_message=f"PyTorch not installed: {e}",
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"VAE engine using device: {device}")

        # ── One-hot encode answers ──
        self.report_progress(5, "One-hot encoding answer data")

        # For large datasets, subsample for training but score all
        max_train = 50_000
        train_indices = np.random.choice(n_students, size=min(max_train, n_students), replace=False)

        onehot = np.zeros((n_students, n_questions * n_options), dtype=np.float32)
        for i in range(n_students):
            for q in range(n_questions):
                onehot[i, q * n_options + answers[i, q]] = 1.0

        input_dim = n_questions * n_options
        latent_dim = 32

        # ── Define VAE Model ──
        class ExamVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_questions = n_questions
                self.n_options = n_options

                # Encoder
                self.enc1 = nn.Linear(input_dim, 512)
                self.bn1 = nn.BatchNorm1d(512)
                self.enc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.enc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)
                self.fc_mu = nn.Linear(128, latent_dim)
                self.fc_logvar = nn.Linear(128, latent_dim)

                # Decoder
                self.dec1 = nn.Linear(latent_dim, 128)
                self.bn4 = nn.BatchNorm1d(128)
                self.dec2 = nn.Linear(128, 256)
                self.bn5 = nn.BatchNorm1d(256)
                self.dec3 = nn.Linear(256, 512)
                self.bn6 = nn.BatchNorm1d(512)
                self.dec_out = nn.Linear(512, input_dim)
                self.dropout = nn.Dropout(0.2)

            def encode(self, x):
                h = self.dropout(F.relu(self.bn1(self.enc1(x))))
                h = self.dropout(F.relu(self.bn2(self.enc2(h))))
                h = F.relu(self.bn3(self.enc3(h)))
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = self.dropout(F.relu(self.bn4(self.dec1(z))))
                h = self.dropout(F.relu(self.bn5(self.dec2(h))))
                h = F.relu(self.bn6(self.dec3(h)))
                return self.dec_out(h)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                return recon, mu, logvar

        # ── Training ──
        self.report_progress(15, "Training VAE on CUDA")

        model = ExamVAE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        beta = 0.5  # KL weight

        train_data = torch.tensor(onehot[train_indices], dtype=torch.float32)
        dataset = TensorDataset(train_data)
        loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

        model.train()
        for epoch in range(50):
            total_loss = 0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                recon, mu, logvar = model(batch)

                # Reconstruction loss
                recon_loss = F.binary_cross_entropy_with_logits(
                    recon, batch, reduction="sum"
                )
                # KL divergence
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if epoch % 5 == 0:
                pct = 15 + int(50 * epoch / 50)
                self.report_progress(pct, f"VAE epoch {epoch}/50 — loss: {avg_loss:.1f}")

        # ── Anomaly Scoring ──
        self.report_progress(70, "Computing anomaly scores for all students")

        model.eval()
        all_data = torch.tensor(onehot, dtype=torch.float32)
        all_dataset = TensorDataset(all_data)
        all_loader = DataLoader(all_dataset, batch_size=1024, shuffle=False)

        anomaly_scores = []
        latent_vectors = []

        with torch.no_grad():
            for (batch,) in all_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                # Per-student reconstruction error
                error = F.mse_loss(torch.sigmoid(recon), batch, reduction="none").sum(dim=1)
                anomaly_scores.extend(error.cpu().numpy())
                latent_vectors.append(mu.cpu().numpy())

        anomaly_scores = np.array(anomaly_scores)
        latent_vectors = np.concatenate(latent_vectors, axis=0)

        # Threshold: mean + 2.5 * std
        threshold = float(anomaly_scores.mean() + 2.5 * anomaly_scores.std())
        flagged_mask = anomaly_scores > threshold

        flagged_ids = [student_ids[i] for i in range(n_students) if flagged_mask[i]]

        # Novel detections (not flagged by classical engines)
        classical_flagged = set()
        for key in ["copy_ring_flagged", "leak_flagged"]:
            classical_flagged.update(kwargs.get(key, []))
        novel = [fid for fid in flagged_ids if fid not in classical_flagged]

        # ── t-SNE on latent space ──
        self.report_progress(80, "Computing t-SNE on VAE latent space (32-dim → 2D)")

        try:
            from sklearn.manifold import TSNE

            # Subsample for t-SNE (max 10K points)
            tsne_n = min(10_000, n_students)
            tsne_indices = np.random.choice(n_students, size=tsne_n, replace=False)
            # Ensure flagged students are included
            flagged_indices = np.where(flagged_mask)[0]
            if len(flagged_indices) > 0:
                tsne_indices = np.unique(np.concatenate([tsne_indices, flagged_indices]))[:tsne_n]

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_coords = tsne.fit_transform(latent_vectors[tsne_indices])

            latent_tsne_data = [
                {
                    "student_id": student_ids[idx],
                    "x": round(float(tsne_coords[i, 0]), 2),
                    "y": round(float(tsne_coords[i, 1]), 2),
                    "anomaly_score": round(float(anomaly_scores[idx]), 2),
                    "is_flagged": bool(flagged_mask[idx]),
                }
                for i, idx in enumerate(tsne_indices)
            ]
        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")
            latent_tsne_data = []

        anomaly_scores_dict = {
            student_ids[i]: round(float(anomaly_scores[i]), 2)
            for i in range(n_students)
        }

        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={
                "model": f"VAE ({input_dim}→512→256→128→{latent_dim} latent)",
                "training_epochs": 50,
                "reconstruction_loss_final": round(avg_loss, 1),
                "anomaly_threshold": round(threshold, 1),
                "total_flagged": len(flagged_ids),
                "novel_detections": len(novel),
                "anomaly_scores": anomaly_scores_dict,
                "latent_visualization": latent_tsne_data,
                "device": str(device),
                "score_stats": {
                    "mean": round(float(anomaly_scores.mean()), 2),
                    "std": round(float(anomaly_scores.std()), 2),
                    "min": round(float(anomaly_scores.min()), 2),
                    "max": round(float(anomaly_scores.max()), 2),
                    "threshold": round(threshold, 2),
                },
            },
            summary={
                "model": "VAE",
                "flagged": len(flagged_ids),
                "novel_detections": len(novel),
                "anomaly_threshold": round(threshold, 1),
                "device": str(device),
            },
        )
