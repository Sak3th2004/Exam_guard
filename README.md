# ExamGuard v2.0 — AI Forensic Intelligence for Examination Integrity

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+cu121-EE4C2C?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

**A 4-layer hybrid AI platform that detects exam fraud using 8 independent detection engines, GPU-accelerated deep learning, and LLM-narrated forensic reports.**

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ExamGuard v2.0                        │
├──────────┬──────────┬──────────────┬────────────────────┤
│ Layer 1  │ Layer 2  │   Layer 3    │     Layer 4        │
│Classical │  Deep    │  Ensemble    │    Forensic        │
│  (CPU)   │Learning  │   (GPU)      │    Narrator        │
│          │  (GPU)   │              │    (LLM)           │
├──────────┼──────────┼──────────────┼────────────────────┤
│ E1:Copy  │ E6:GNN   │  XGBoost    │  Ollama/Mistral    │
│ E2:Stats │ E7:VAE   │  Meta-      │  Report            │
│ E3:Center│ E8:NLP   │  Classifier │  Generation        │
│ E4:Leak  │          │             │                    │
│ E5:Time  │          │             │                    │
└──────────┴──────────┴──────────────┴────────────────────┘
```

### Detection Engines

| Engine | Name | Method | Layer |
|--------|------|--------|-------|
| E1 | Copy Ring Detection | MinHash LSH + Louvain Community Detection | Classical (CPU) |
| E2 | Statistical Impossibility Proof | Binomial Survival + Bonferroni Correction | Classical (CPU) |
| E3 | Center Anomaly Detection | 8-Feature Isolation Forest + Z-Score | Classical (CPU) |
| E4 | Leak Signature Detection | IRT 2PL Person-Fit + Difficulty Curve Inversion | Classical (CPU) |
| E5 | Response Time Analysis | KDE + K-Means Clustering | Classical (CPU) |
| E6 | Graph Neural Network | GraphSAGE (2-layer, 64→32→2) | Deep Learning (GPU) |
| E7 | Variational Autoencoder | VAE (input→512→256→128→32 latent) + t-SNE | Deep Learning (GPU) |
| E8 | Question Similarity | Sentence Transformer (all-MiniLM-L6-v2) | Deep Learning (GPU) |
| — | XGBoost Ensemble | Gradient Boosting Meta-Classifier (12 features) | Ensemble (GPU) |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA 12.1+ (for GPU engines)
- Ollama (optional, for LLM narration)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/Sak3th2004/Exam_guard.git
cd Exam_guard

# 2. Backend setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r backend/requirements.txt

# Install PyTorch with CUDA (for GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
docker-compose up -d
# Frontend: http://localhost
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 📊 Features

### Synthetic Data Generator
- IRT 2PL model for realistic exam responses
- 4 planted fraud patterns (copy rings, paper leak, center anomaly, timing fraud)
- Configurable: students, questions, centers, options
- Ground truth labels for supervised training

### Real-Time Analysis Dashboard
- WebSocket-based live engine progress streaming
- Per-engine status cards (CPU/GPU badges)
- Overall examination integrity score (0-100)
- Risk tier distribution (CRITICAL/HIGH/MEDIUM/LOW)

### Visualization Suite
- Network graph (copy ring clusters)
- Feature importance chart (XGBoost)
- Fraud probability rankings table
- Difficulty curve analysis
- t-SNE latent space scatter plot

### Forensic Reporting
- PDF report generation (ReportLab)
- LLM-narrated findings (Ollama/Mistral 7B)
- Template fallback when LLM unavailable
- Executive summary + per-engine sections

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health (GPU, Ollama status) |
| POST | `/api/v1/generate` | Generate synthetic data + auto-analyze |
| POST | `/api/v1/analyses` | Upload CSV and start analysis |
| GET | `/api/v1/analyses/{id}` | Get analysis status and summary |
| GET | `/api/v1/analyses/{id}/engines/{name}` | Get engine detail |
| GET | `/api/v1/analyses/{id}/flagged` | Paginated flagged entities |
| GET | `/api/v1/analyses/{id}/graph` | Network graph data |
| GET | `/api/v1/analyses/{id}/heatmap` | Geographic heatmap |
| GET | `/api/v1/analyses/{id}/difficulty-curve` | Difficulty curve data |
| GET | `/api/v1/analyses/{id}/latent-space` | VAE t-SNE coordinates |
| GET | `/api/v1/analyses/{id}/ensemble-rankings` | XGBoost fraud rankings |
| GET | `/api/v1/analyses/{id}/feature-importance` | Feature importance |
| POST | `/api/v1/analyses/{id}/compare` | Side-by-side student comparison |
| GET | `/api/v1/analyses/{id}/report` | Download PDF report |
| WS | `/ws/analyses/{id}` | Real-time progress stream |

## 🛠️ Tech Stack

### Backend
- **FastAPI** — Async REST API + WebSocket
- **SQLAlchemy** — Async ORM (SQLite/PostgreSQL)
- **PyTorch + PyG** — GNN and VAE engines
- **XGBoost** — GPU-accelerated ensemble
- **scikit-learn** — Isolation Forest, t-SNE
- **sentence-transformers** — NLP similarity
- **ReportLab** — PDF generation
- **Ollama** — Local LLM narration

### Frontend
- **React 18** — UI framework
- **Vite** — Build tool
- **Recharts** — Data visualization
- **React Router** — Client-side routing

### Infrastructure
- **Docker + Docker Compose** — Containerization
- **Nginx** — Reverse proxy + SPA serving
- **CUDA 12.1** — GPU acceleration

## 📁 Project Structure

```
ExamGuard/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analysis.py      # 15 REST endpoints
│   │   │   ├── generator.py     # Synthetic data generation
│   │   │   └── health.py        # GPU/Ollama health check
│   │   └── websocket.py         # Real-time progress streaming
│   ├── data/
│   │   ├── generator.py         # IRT 2PL data generator (770 lines)
│   │   ├── ingestion.py         # CSV parser with auto-detection
│   │   └── schemas.py           # 30+ Pydantic models
│   ├── engines/
│   │   ├── base_engine.py       # Abstract engine interface
│   │   ├── copy_ring.py         # E1: MinHash + Louvain
│   │   ├── stat_impossibility.py # E2: Binomial + Bonferroni
│   │   ├── center_anomaly.py    # E3: Isolation Forest
│   │   ├── leak_signature.py    # E4: IRT + Difficulty Curve
│   │   ├── response_time.py     # E5: KDE + K-Means
│   │   ├── gnn_fraud.py         # E6: GraphSAGE GNN
│   │   ├── vae_anomaly.py       # E7: VAE + t-SNE
│   │   ├── question_similarity.py # E8: Sentence Transformer
│   │   ├── xgboost_ensemble.py  # XGBoost meta-classifier
│   │   └── orchestrator.py      # Parallel engine execution
│   ├── models/
│   │   └── database.py          # SQLAlchemy async models
│   ├── services/
│   │   ├── comparison.py        # Student diff service
│   │   ├── llm_narrator.py      # Ollama + template fallback
│   │   └── report_generator.py  # PDF report builder
│   ├── config.py                # Pydantic settings
│   ├── main.py                  # FastAPI entry point
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # React SPA with routing
│   │   ├── api.js               # API client + WebSocket
│   │   ├── index.css            # Dark forensic design system
│   │   └── main.jsx             # React entry point
│   └── package.json
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
└── README.md
```

## 🔒 Design Principles

1. **Separation of Detection and Narration** — AI models detect patterns mathematically. LLM only narrates results for human readability. Never uses AI opinion for fraud determination.

2. **Graceful Degradation** — If GPU unavailable, classical CPU engines still work. If Ollama unavailable, template reports are generated. No single point of failure.

3. **Parallel Execution** — CPU engines run concurrently via asyncio. GPU engines run sequentially to share VRAM. Real-time progress via WebSocket.

4. **Evidence-Based Scoring** — Every flagged entity has mathematical evidence (p-values, similarity scores, anomaly distances). No black-box decisions.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
