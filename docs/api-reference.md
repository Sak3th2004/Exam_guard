# ExamGuard v2 — API Reference

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
No authentication required (local deployment).

---

## Health Check

### `GET /health`
Check system status including GPU and Ollama availability.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4060",
  "gpu_memory_mb": 8188,
  "ollama_available": false,
  "timestamp": "2026-06-12T10:00:00Z"
}
```

---

## Data Generation

### `POST /api/v1/generate`
Generate IRT-based synthetic exam data with planted fraud patterns.

**Request Body:**
```json
{
  "n_students": 1000,
  "n_questions": 50,
  "n_centers": 10,
  "n_options": 4,
  "include_timing": true,
  "include_question_text": true,
  "exam_name": "NEET 2026 Forensic Simulation"
}
```

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "message": "Analysis started"
}
```

---

## Analysis

### `POST /api/v1/analyses`
Upload CSV file for analysis.

**Request:** `multipart/form-data`
- `file`: CSV file (required)
- `config`: JSON string with exam configuration (optional)

**Response:**
```json
{
  "id": "uuid-string",
  "status": "processing"
}
```

### `GET /api/v1/analyses/{id}`
Get analysis status and summary.

**Response:**
```json
{
  "id": "uuid-string",
  "status": "complete",
  "exam_name": "NEET 2026",
  "total_students": 1000,
  "total_questions": 50,
  "total_centers": 10,
  "total_flagged": 43,
  "overall_score": 95.7,
  "engine_summaries": {
    "copy_ring": { "status": "complete", "flagged_count": 3 },
    "stat_impossibility": { "status": "complete", "flagged_count": 6 }
  }
}
```

### `GET /api/v1/analyses/{id}/engines/{engine_name}`
Get detailed results from a specific engine.

**Engine names:** `copy_ring`, `stat_impossibility`, `center_anomaly`, `leak_signature`, `response_time`, `gnn_copy_ring`, `vae_anomaly`, `question_similarity`, `xgboost_ensemble`, `benford_law`

### `GET /api/v1/analyses/{id}/flagged`
Get paginated list of flagged entities.

**Query Parameters:**
- `limit` (int, default 50)
- `offset` (int, default 0)
- `engine` (string, optional filter)

### `GET /api/v1/analyses/{id}/graph`
Get network graph data for visualization.

### `GET /api/v1/analyses/{id}/heatmap`
Get geographic heatmap data.

### `GET /api/v1/analyses/{id}/difficulty-curve`
Get difficulty curve data from E4 leak signature engine.

### `GET /api/v1/analyses/{id}/latent-space`
Get VAE t-SNE latent space data for visualization.

### `GET /api/v1/analyses/{id}/ensemble-rankings`
Get XGBoost ensemble fraud probability rankings.

**Query Parameters:**
- `limit` (int, default 100)

### `GET /api/v1/analyses/{id}/feature-importance`
Get XGBoost feature importance values.

### `POST /api/v1/analyses/{id}/compare`
Compare two students side-by-side.

**Request Body:**
```json
{
  "student_a": "STU_000001",
  "student_b": "STU_000002"
}
```

### `GET /api/v1/analyses/{id}/report`
Download PDF forensic report.

---

## WebSocket

### `WS /ws/analyses/{id}`
Real-time progress streaming during analysis.

**Message Format:**
```json
{
  "engine": "copy_ring",
  "progress": 45,
  "message": "Computing Jaccard similarity",
  "status": "running"
}
```
