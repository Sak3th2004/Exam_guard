# ExamGuard v2 — System Architecture

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Vite)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  Upload   │ │Dashboard │ │ Compare  │ │ Report   │          │
│  │  CSV/Gen  │ │ Chart.js │ │ Students │ │ Download │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       └────────────┴────────────┴─────────────┘                │
│                        │ HTTP + WebSocket                       │
└────────────────────────┼───────────────────────────────────────┘
                         │
┌────────────────────────┼───────────────────────────────────────┐
│                  BACKEND (FastAPI + uvicorn)                     │
│                        │                                        │
│  ┌─────────────────────┴────────────────────────────┐          │
│  │              ORCHESTRATOR                         │          │
│  │  Manages parallel engine execution + progress     │          │
│  └─────────────────────┬────────────────────────────┘          │
│                        │                                        │
│  ┌─────────────────────┴────────────────────────────┐          │
│  │  LAYER 1: Classical (CPU — asyncio parallel)      │          │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐      │          │
│  │  │ E1 │ │ E2 │ │ E3 │ │ E4 │ │ E5 │ │ E9 │      │          │
│  │  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘      │          │
│  └──────────────────────────────────────────────────┘          │
│  ┌──────────────────────────────────────────────────┐          │
│  │  LAYER 2: Deep Learning (GPU — sequential)        │          │
│  │  ┌────┐ ┌────┐ ┌────┐                             │          │
│  │  │ E6 │ │ E7 │ │ E8 │                             │          │
│  │  └────┘ └────┘ └────┘                             │          │
│  └──────────────────────────────────────────────────┘          │
│  ┌──────────────────────────────────────────────────┐          │
│  │  LAYER 3: XGBoost Ensemble (GPU)                  │          │
│  └──────────────────────────────────────────────────┘          │
│  ┌──────────────────────────────────────────────────┐          │
│  │  LAYER 4: LLM Narrator (Ollama + Fallback)       │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  SQLite DB   │  │  PDF Report  │  │  File Store  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Input**: CSV upload or IRT synthetic data generation
2. **Ingestion**: Parse CSV → numpy arrays (answers, timing, metadata)
3. **Layer 1**: 6 CPU engines run in parallel via `asyncio.gather()`
4. **Layer 2**: 3 GPU engines run sequentially (shared VRAM)
5. **Layer 3**: XGBoost combines 12 features from all engines
6. **Layer 4**: LLM writes human-readable report narratives
7. **Output**: Dashboard + PDF report + WebSocket progress

## Engine Dependencies

```
E1 (Copy Ring) ──→ E2 (Stat, uses E1 pairs)
                └──→ E6 (GNN, uses E1 graph)
E4 (Leak) ─────→ E7 (VAE, uses E4 labels for context)
All engines ────→ XGBoost Ensemble (aggregates all)
All results ────→ PDF Report (LLM narration)
```

## GPU Memory Management

- E6 GNN: ~200MB (GraphSAGE model + graph data)
- E7 VAE: ~400MB (800-dim encoder-decoder)
- E8 NLP: ~300MB (MiniLM model)
- XGBoost: ~100MB (gpu_hist training)
- Total: ~1GB (fits in 8GB RTX 4060 VRAM)
- Sequential execution prevents OOM

## Graceful Degradation

| Component | If Fails | Fallback |
|-----------|----------|----------|
| GPU | E6-E8 skip | Classical engines still produce results |
| torch_geometric | E6 skips | Other 8 engines run normally |
| Ollama | LLM skips | Template-based PDF text |
| Any single engine | Engine skipped | Other engines compensate |
| CSV parsing | 400 error | Clear validation error message |
