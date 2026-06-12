# ExamGuard v2.0 — AI Forensic Intelligence Platform

> **4-Layer Hybrid AI System** for automated examination fraud detection with mathematical rigor and GPU-accelerated deep learning.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               FRONTEND (React + Vite + Bootstrap 5)           │
│  Upload │ Dashboard │ Network Graph │ Charts │ PDF Report     │
└──────────────────────────┬───────────────────────────────────┘
                           │ REST API + WebSocket
┌──────────────────────────┼───────────────────────────────────┐
│                     FASTAPI BACKEND                           │
│                                                               │
│  ═══ LAYER 1: CLASSICAL DETECTION (CPU) ════════════════      │
│  E1 Copy Ring    │ MinHash LSH + Louvain Community            │
│  E2 Stat Proof   │ Binomial + Bonferroni Correction           │
│  E3 Center       │ Isolation Forest + Z-Score                 │
│  E4 Leak         │ IRT 2PL + Difficulty Curve Inversion       │
│  E5 Timing       │ KDE + K-Means Clustering                  │
│                                                               │
│  ═══ LAYER 2: DEEP LEARNING (GPU — RTX 4060) ══════════      │
│  E6 GNN          │ GraphSAGE (PyTorch Geometric)              │
│  E7 VAE          │ Variational Autoencoder (PyTorch)          │
│  E8 NLP          │ Sentence Transformer (HuggingFace)         │
│                                                               │
│  ═══ LAYER 3: META-ENSEMBLE (GPU) ══════════════════════      │
│  XGBoost         │ Gradient Boosted Meta-Classifier           │
│                                                               │
│  ═══ LAYER 4: LLM NARRATOR (GPU) ══════════════════════      │
│  Mistral 7B      │ Local LLM via Ollama for report narration  │
└──────────────────────────────────────────────────────────────┘
```

## Performance Benchmarks

| Metric | Score |
|--------|-------|
| Accuracy | 98.6% |
| Precision | 81.8% |
| Recall | 85.7% |
| F1 Score | 83.7% |
| AUC-ROC | 97.8% |

*Evaluated on 1,000 students with 4.2% planted fraud rate (IRT 2PL synthetic data).*

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA 12.x (RTX 4060 or better)
- Ollama (optional, for LLM narrator)

### Backend
```bash
cd backend
pip install -r requirements.txt
# For PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse

python -m uvicorn main:app --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Run E2E Test
```bash
python tests/test_e2e.py
```

## Key Design Principles

1. **Separation of Concerns**: Detection (Layers 1-3) is pure math/ML. Narration (Layer 4) is LLM. The LLM never makes detection decisions.
2. **Graceful Degradation**: If GPU engines fail, classical engines still produce results. If LLM is unavailable, PDF uses template text.
3. **Parallel Execution**: CPU engines run concurrently in asyncio. GPU engines run sequentially (shared VRAM).
4. **Real-Time Streaming**: WebSocket pushes per-engine progress to frontend.

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, asyncio
- **Frontend**: React 19, Vite, Bootstrap 5, Chart.js
- **ML**: scikit-learn, XGBoost (GPU), PyTorch, PyTorch Geometric
- **NLP**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (Mistral 7B)
- **Data**: IRT 2PL synthetic generator (scipy, numpy)

## License

MIT
