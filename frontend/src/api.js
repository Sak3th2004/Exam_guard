const API_BASE = 'http://localhost:8000/api/v1';
const WS_BASE = 'ws://localhost:8000';

export const api = {
  async health() {
    const res = await fetch('http://localhost:8000/health');
    return res.json();
  },

  async uploadCSV(file, config = {}) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('config', JSON.stringify(config));
    const res = await fetch(`${API_BASE}/analyses`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
    return res.json();
  },

  async generateData(config = {}) {
    const res = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        n_students: config.n_students || 100000,
        n_questions: config.n_questions || 200,
        n_centers: config.n_centers || 450,
        n_options: config.n_options || 4,
        include_timing: config.include_timing !== false,
        include_question_text: config.include_question_text !== false,
        exam_name: config.exam_name || 'NEET 2026 Forensic Simulation',
      }),
    });
    return res.json();
  },

  async getAnalysis(id) {
    const res = await fetch(`${API_BASE}/analyses/${id}`);
    return res.json();
  },

  async getEngineDetail(analysisId, engineName) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/engines/${engineName}`);
    return res.json();
  },

  async getFlagged(analysisId, params = {}) {
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/flagged?${qs}`);
    return res.json();
  },

  async getGraph(analysisId) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/graph`);
    return res.json();
  },

  async getHeatmap(analysisId) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/heatmap`);
    return res.json();
  },

  async getDifficultyCurve(analysisId) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/difficulty-curve`);
    return res.json();
  },

  async getLatentSpace(analysisId) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/latent-space`);
    return res.json();
  },

  async getRankings(analysisId, limit = 100) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/ensemble-rankings?limit=${limit}`);
    return res.json();
  },

  async getFeatureImportance(analysisId) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/feature-importance`);
    return res.json();
  },

  async compareStudents(analysisId, studentA, studentB) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_a: studentA, student_b: studentB }),
    });
    return res.json();
  },

  getReportUrl(analysisId) {
    return `${API_BASE}/analyses/${analysisId}/report`;
  },

  async getBenchmark(analysisId) {
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/benchmark`);
    return res.json();
  },

  connectWebSocket(analysisId, onMessage) {
    const ws = new WebSocket(`${WS_BASE}/ws/analyses/${analysisId}`);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };
    ws.onerror = (e) => console.error('WS error:', e);
    return ws;
  },
};
