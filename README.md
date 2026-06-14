# ExamGuard

Forensic analysis platform for detecting fraud patterns in computer-based examinations.

Built for **FAR AWAY 2026** — *Examinations* theme.

## Problem

Large-scale CBT exams (100k+ students) are vulnerable to:
- **Copy rings** — coordinated answer sharing between students
- **Paper leaks** — pre-exam access to questions/answers
- **Center-level fraud** — systematic manipulation at specific test centers
- **Timing anomalies** — impossibly fast responses indicating pre-knowledge

Manual detection is infeasible at scale. ExamGuard automates this with 9 detection engines across 3 analysis layers.

## Architecture

```
Layer 1: Classical Statistical Engines (CPU, parallel)
├── E1  Copy Ring Detection      — MinHash LSH + Louvain community detection
├── E2  Statistical Impossibility — Binomial test + Bonferroni correction
├── E3  Center Anomaly           — Isolation Forest + Z-score
├── E4  Leak Signature           — IRT 2PL difficulty gradient analysis
├── E5  Response Time            — KDE + K-Means clustering
└── E9  Benford's Law            — Chi-squared first-digit distribution test

Layer 2: Deep Learning Engines (GPU, sequential)
├── E6  GNN Fraud Detection      — 2-layer GraphSAGE node classification
├── E7  VAE Anomaly Detector     — Variational Autoencoder + t-SNE
└── E8  Question Similarity      — Sentence-BERT cosine similarity

Layer 3: Ensemble Meta-Classifier
└── XGBoost                      — Gradient boosted trees combining all engine outputs
```

## Features

- **9 detection engines** running in parallel with real-time WebSocket progress
- **IRT 2PL data generator** with planted fraud patterns for testing
- **CSV upload** for real exam data analysis
- **Interactive dashboard** with radar charts, risk distribution, and engine comparison
- **Student comparison view** with Jaccard similarity and statistical significance tests
- **PDF forensic report** with per-engine findings and evidence
- **XGBoost ensemble** ranking students by fraud probability with feature importance

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, FastAPI, SQLAlchemy |
| ML/Stats | NumPy, SciPy, scikit-learn, NetworkX, datasketch |
| Deep Learning | PyTorch, PyTorch Geometric, sentence-transformers |
| Ensemble | XGBoost (GPU) |
| Frontend | React 18, Vite, Chart.js |
| Reports | ReportLab (PDF generation) |

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA (optional, falls back to CPU)

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
python main.py
```

Backend runs at `http://127.0.0.1:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

## Sample Data

### Quick Test (included in repo)

The `sample_data/` directory contains pre-generated exam data:

| File | Contents |
|------|----------|
| `sample_exam_answers.csv` | 100 students × 30 questions with planted fraud |
| `sample_answer_key.csv` | Correct answers |
| `sample_timing_data.csv` | Response times per question |

**Planted fraud:** Copy ring (STU_0000–STU_0004), Paper leak (STU_0010–STU_0014)

### Scale Test (1,00,000 students)

To generate a full-scale dataset locally:
```bash
cd backend
python -c "from data.generator import generate_exam_data; generate_exam_data(100000, 200, 50)"
```

Or use the built-in simulator: set Students to `100000` in the UI and click Start Analysis.

## Usage

1. Open `http://localhost:5173`
2. Either:
   - Click **"Start Analysis"** to generate synthetic data and run all engines
   - Click **"Upload CSV"** to analyze your own exam data
3. View results on the dashboard:
   - **Dashboard** — overview with integrity score, charts
   - **Engine Detail** — per-engine findings (E1–E9)
   - **Fraud Rankings** — XGBoost ensemble student rankings
   - **Network Graph** — copy ring similarity clusters
   - **Compare Students** — side-by-side answer comparison
   - **Benchmarks** — accuracy, precision, recall metrics
4. Click **"Download Report"** for a PDF forensic report

## API

Full API documentation: [`docs/api-reference.md`](docs/api-reference.md)

Key endpoints:
```
POST   /api/v1/generate              Generate synthetic exam data + run analysis
GET    /api/v1/analyses/{id}         Analysis status and results
GET    /api/v1/analyses/{id}/report  Download PDF report
POST   /api/v1/analyses/{id}/compare Compare two students
GET    /health                       System status
```

## Documentation

- [`docs/algorithms.md`](docs/algorithms.md) — Mathematical foundations of each engine
- [`docs/architecture.md`](docs/architecture.md) — System architecture and data flow
- [`docs/api-reference.md`](docs/api-reference.md) — API endpoint reference

## Project Structure

```
ExamGuard/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings and thresholds
│   ├── engines/                # 9 detection engines + orchestrator
│   ├── data/                   # IRT data generator, CSV ingestion
│   ├── services/               # PDF reports, LLM narrator, comparison
│   ├── api/                    # REST routes + WebSocket
│   └── models/                 # Database models
├── frontend/
│   └── src/
│       ├── App.jsx             # Main application (7 pages)
│       ├── api.js              # Backend API client
│       └── index.css           # Design system
├── sample_data/                # Pre-generated test data
├── docs/                       # Technical documentation
└── README.md
```

## License

MIT
