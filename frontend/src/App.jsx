import { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, Link, useNavigate, useParams, useLocation } from 'react-router-dom';
import { api } from './api';
import './index.css';

// ═══════════════════════════════════════════════
// Sidebar Component
// ═══════════════════════════════════════════════

function Sidebar({ analysisId }) {
  const location = useLocation();
  const isActive = (path) => location.pathname.includes(path);

  return (
    <aside className="sidebar">
      <div className="sidebar__logo">
        <div className="sidebar__logo-icon">EG</div>
        <span className="sidebar__logo-text">ExamGuard</span>
      </div>
      <nav className="sidebar__nav">
        <Link to="/" className={`sidebar__link ${location.pathname === '/' ? 'sidebar__link--active' : ''}`}>
          <span className="sidebar__link-icon">🏠</span>Home
        </Link>
        {analysisId && (
          <>
            <Link to={`/analysis/${analysisId}`} className={`sidebar__link ${isActive(`/analysis/${analysisId}`) && !isActive('/engines') && !isActive('/rankings') && !isActive('/graph') ? 'sidebar__link--active' : ''}`}>
              <span className="sidebar__link-icon">📊</span>Dashboard
            </Link>
            <Link to={`/analysis/${analysisId}/engines`} className={`sidebar__link ${isActive('/engines') ? 'sidebar__link--active' : ''}`}>
              <span className="sidebar__link-icon">⚙️</span>Engines
            </Link>
            <Link to={`/analysis/${analysisId}/rankings`} className={`sidebar__link ${isActive('/rankings') ? 'sidebar__link--active' : ''}`}>
              <span className="sidebar__link-icon">🏆</span>Rankings
            </Link>
            <Link to={`/analysis/${analysisId}/graph`} className={`sidebar__link ${isActive('/graph') ? 'sidebar__link--active' : ''}`}>
              <span className="sidebar__link-icon">🔗</span>Network
            </Link>
          </>
        )}
      </nav>
      <div style={{ padding: '12px', borderTop: '1px solid var(--border)', marginTop: 'auto' }}>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>v2.0 · 4-Layer AI</p>
        <p style={{ fontSize: '10px', color: 'var(--text-muted)' }}>8 Engines · GPU</p>
      </div>
    </aside>
  );
}

// ═══════════════════════════════════════════════
// Home Page — Generate & Upload
// ═══════════════════════════════════════════════

function HomePage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState({
    n_students: 100000,
    n_questions: 200,
    n_centers: 450,
    exam_name: 'NEET 2026 Forensic Simulation',
  });

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
  }, []);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const result = await api.generateData(config);
      navigate(`/analysis/${result.analysis_id}`);
    } catch (e) {
      alert('Generation failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header" style={{ textAlign: 'center', padding: '60px 0 40px' }}>
        <h1 className="page-header__title" style={{ fontSize: '42px', marginBottom: '12px' }}>
          <span style={{ background: 'var(--gradient-cyan)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            ExamGuard
          </span>
        </h1>
        <p className="page-header__subtitle" style={{ fontSize: '18px', maxWidth: '600px', margin: '0 auto' }}>
          AI Forensic Intelligence for Examination Integrity
        </p>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '8px' }}>
          4-Layer Hybrid Detection · 8 Engines · GPU-Accelerated · LLM-Narrated Reports
        </p>
      </div>

      {/* System Status */}
      {health && (
        <div className="stats-grid" style={{ maxWidth: '800px', margin: '0 auto 40px' }}>
          <div className="stat-card stat-card--cyan">
            <div className="stat-card__icon">💻</div>
            <div className="stat-card__value">{health.gpu_available ? '✓' : '✗'}</div>
            <div className="stat-card__label">GPU ({health.gpu_name || 'N/A'})</div>
          </div>
          <div className="stat-card stat-card--green">
            <div className="stat-card__icon">🧠</div>
            <div className="stat-card__value">{health.gpu_memory_mb ? `${(health.gpu_memory_mb/1024).toFixed(1)}GB` : 'N/A'}</div>
            <div className="stat-card__label">VRAM Available</div>
          </div>
          <div className="stat-card stat-card--purple">
            <div className="stat-card__icon">🤖</div>
            <div className="stat-card__value">{health.ollama_available ? '✓' : '✗'}</div>
            <div className="stat-card__label">Ollama LLM</div>
          </div>
        </div>
      )}

      {/* Generate Card */}
      <div className="card" style={{ maxWidth: '800px', margin: '0 auto' }}>
        <div className="card__header">
          <div>
            <h2 className="card__title">Generate Forensic Simulation</h2>
            <p className="card__subtitle">Create IRT-based synthetic exam data with planted fraud patterns</p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
          <div>
            <label className="label">Students</label>
            <input className="input" type="number" value={config.n_students}
                   onChange={e => setConfig({...config, n_students: parseInt(e.target.value)})} />
          </div>
          <div>
            <label className="label">Questions</label>
            <input className="input" type="number" value={config.n_questions}
                   onChange={e => setConfig({...config, n_questions: parseInt(e.target.value)})} />
          </div>
          <div>
            <label className="label">Centers</label>
            <input className="input" type="number" value={config.n_centers}
                   onChange={e => setConfig({...config, n_centers: parseInt(e.target.value)})} />
          </div>
          <div>
            <label className="label">Exam Name</label>
            <input className="input" type="text" value={config.exam_name}
                   onChange={e => setConfig({...config, exam_name: e.target.value})} />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn btn--primary btn--lg" onClick={handleGenerate} disabled={loading} id="generate-btn">
            {loading ? <span className="spinner" /> : '🚀'} Generate & Analyze
          </button>
        </div>

        <div style={{ marginTop: '16px', fontSize: '12px', color: 'var(--text-muted)' }}>
          <p>⚡ Will generate {config.n_students.toLocaleString()} students × {config.n_questions} questions with 4 planted fraud patterns</p>
          <p>📊 All 8 engines will run automatically including GPU-accelerated GNN + VAE + XGBoost</p>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// Dashboard Page
// ═══════════════════════════════════════════════

function DashboardPage() {
  const { id } = useParams();
  const [analysis, setAnalysis] = useState(null);
  const [engineProgress, setEngineProgress] = useState({});
  const wsRef = useRef(null);

  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const data = await api.getAnalysis(id);
        setAnalysis(data);
        if (data.status === 'complete' || data.status === 'failed') {
          clearInterval(poll);
        }
      } catch (e) {
        console.error(e);
      }
    }, 2000);

    // WebSocket for real-time progress
    wsRef.current = api.connectWebSocket(id, (msg) => {
      setEngineProgress(prev => ({
        ...prev,
        [msg.engine]: { progress: msg.progress, message: msg.message, status: msg.status },
      }));
    });

    return () => {
      clearInterval(poll);
      if (wsRef.current) wsRef.current.close();
    };
  }, [id]);

  if (!analysis) {
    return (
      <div className="loading-screen">
        <div className="spinner" style={{ width: '48px', height: '48px', borderWidth: '4px' }} />
        <p className="loading-screen__text">Loading analysis...</p>
      </div>
    );
  }

  const score = analysis.overall_score || 0;
  const scoreColor = score < 50 ? 'var(--accent-red)' : score < 75 ? 'var(--accent-orange)' : 'var(--accent-green)';

  const engines = [
    { name: 'copy_ring', label: 'E1: Copy Ring', type: 'CPU', icon: '🔗' },
    { name: 'stat_impossibility', label: 'E2: Stat Proof', type: 'CPU', icon: '📐' },
    { name: 'center_anomaly', label: 'E3: Center Anomaly', type: 'CPU', icon: '🏫' },
    { name: 'leak_signature', label: 'E4: Leak Signature', type: 'CPU', icon: '🔓' },
    { name: 'response_time', label: 'E5: Response Time', type: 'CPU', icon: '⏱️' },
    { name: 'gnn_copy_ring', label: 'E6: GNN', type: 'GPU', icon: '🧠' },
    { name: 'vae_anomaly', label: 'E7: VAE', type: 'GPU', icon: '🔬' },
    { name: 'question_similarity', label: 'E8: NLP', type: 'GPU', icon: '📝' },
    { name: 'xgboost_ensemble', label: 'XGBoost Ensemble', type: 'GPU', icon: '🎯' },
  ];

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-header__title">{analysis.exam_name}</h1>
        <p className="page-header__subtitle">
          Analysis {analysis.id.substring(0, 8)} · {analysis.status.toUpperCase()}
        </p>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card stat-card--cyan">
          <div className="stat-card__icon">👨‍🎓</div>
          <div className="stat-card__value">{(analysis.total_students || 0).toLocaleString()}</div>
          <div className="stat-card__label">Students Analyzed</div>
        </div>
        <div className="stat-card stat-card--purple">
          <div className="stat-card__icon">📋</div>
          <div className="stat-card__value">{analysis.total_questions || 0}</div>
          <div className="stat-card__label">Questions</div>
        </div>
        <div className="stat-card stat-card--orange">
          <div className="stat-card__icon">🏫</div>
          <div className="stat-card__value">{analysis.total_centers || 0}</div>
          <div className="stat-card__label">Centers</div>
        </div>
        <div className="stat-card" style={{ '--stat-color': scoreColor }}>
          <div className="stat-card__icon">🛡️</div>
          <div className="stat-card__value" style={{ color: scoreColor }}>{score.toFixed(1)}</div>
          <div className="stat-card__label">Integrity Score</div>
        </div>
        <div className="stat-card stat-card--red">
          <div className="stat-card__icon">🚩</div>
          <div className="stat-card__value">{(analysis.total_flagged || 0).toLocaleString()}</div>
          <div className="stat-card__label">Flagged</div>
        </div>
      </div>

      {/* Engine Progress */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card__header">
          <h2 className="card__title">Detection Engines</h2>
          <span className={`badge badge--${analysis.status}`}>{analysis.status}</span>
        </div>
        <div className="engine-grid">
          {engines.map(eng => {
            const progress = engineProgress[eng.name];
            const summary = analysis.engine_summaries?.[eng.name];
            const status = progress?.status || summary?.status || 'pending';
            const pct = progress?.progress || (status === 'complete' ? 100 : 0);

            return (
              <div key={eng.name} className="engine-card">
                <div className="engine-card__header">
                  <span className="engine-card__name">{eng.icon} {eng.label}</span>
                  <span className={`engine-card__badge engine-card__badge--${eng.type.toLowerCase()}`}>{eng.type}</span>
                </div>
                <div className="engine-card__progress">
                  <div className="engine-card__progress-bar" style={{
                    width: `${pct}%`,
                    background: status === 'complete' ? 'var(--gradient-success)' :
                                status === 'failed' ? 'var(--gradient-danger)' : 'var(--gradient-cyan)',
                  }} />
                </div>
                <div className="engine-card__status">
                  {progress?.message || (status === 'complete' ? `${summary?.flagged_count || 0} flagged` : status)}
                </div>
                {summary?.flagged_count > 0 && (
                  <div className="engine-card__result">
                    <span className="badge badge--high">{summary.flagged_count} flagged</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Actions */}
      {analysis.status === 'complete' && (
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <Link to={`/analysis/${id}/engines`} className="btn btn--primary">⚙️ View Engine Details</Link>
          <Link to={`/analysis/${id}/rankings`} className="btn btn--secondary">🏆 View Rankings</Link>
          <a href={api.getReportUrl(id)} className="btn btn--secondary" target="_blank" rel="noopener">📄 Download PDF Report</a>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════
// Engines Detail Page
// ═══════════════════════════════════════════════

function EnginesPage() {
  const { id } = useParams();
  const [activeEngine, setActiveEngine] = useState('copy_ring');
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);

  const engines = [
    { name: 'copy_ring', label: 'E1: Copy Ring' },
    { name: 'stat_impossibility', label: 'E2: Stat Proof' },
    { name: 'center_anomaly', label: 'E3: Center Anomaly' },
    { name: 'leak_signature', label: 'E4: Leak Signature' },
    { name: 'response_time', label: 'E5: Response Time' },
    { name: 'gnn_copy_ring', label: 'E6: GNN' },
    { name: 'vae_anomaly', label: 'E7: VAE' },
    { name: 'question_similarity', label: 'E8: NLP' },
    { name: 'xgboost_ensemble', label: 'Ensemble' },
  ];

  useEffect(() => {
    setLoading(true);
    api.getEngineDetail(id, activeEngine).then(data => {
      setDetail(data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [id, activeEngine]);

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-header__title">Engine Results</h1>
      </div>

      {/* Engine Tabs */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '24px' }}>
        {engines.map(eng => (
          <button key={eng.name} className={`btn btn--sm ${activeEngine === eng.name ? 'btn--primary' : 'btn--secondary'}`}
                  onClick={() => setActiveEngine(eng.name)}>
            {eng.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading-screen"><div className="spinner" /></div>
      ) : detail ? (
        <div className="card">
          <div className="card__header">
            <div>
              <h2 className="card__title">{activeEngine.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</h2>
              <p className="card__subtitle">{detail.duration_ms ? `Completed in ${(detail.duration_ms/1000).toFixed(1)}s` : ''}</p>
            </div>
            <span className={`badge badge--${detail.status}`}>{detail.status}</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginBottom: '20px' }}>
            <div className="stat-card stat-card--red">
              <div className="stat-card__value">{detail.flagged_count || 0}</div>
              <div className="stat-card__label">Flagged</div>
            </div>
            <div className="stat-card stat-card--cyan">
              <div className="stat-card__value">{detail.duration_ms ? `${(detail.duration_ms/1000).toFixed(1)}s` : 'N/A'}</div>
              <div className="stat-card__label">Duration</div>
            </div>
            <div className="stat-card stat-card--purple">
              <div className="stat-card__value">{detail.result_data?.device || 'CPU'}</div>
              <div className="stat-card__label">Device</div>
            </div>
          </div>

          {/* Raw data display */}
          <details style={{ marginTop: '16px' }}>
            <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '13px' }}>
              📋 View Raw Result Data
            </summary>
            <pre style={{
              marginTop: '8px', padding: '16px', background: 'var(--bg-primary)',
              borderRadius: 'var(--radius-sm)', fontSize: '11px', fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)', overflow: 'auto', maxHeight: '400px',
            }}>
              {JSON.stringify(detail.result_data, null, 2)}
            </pre>
          </details>
        </div>
      ) : (
        <div className="card"><p style={{ color: 'var(--text-muted)' }}>No data available for this engine</p></div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════
// Rankings Page
// ═══════════════════════════════════════════════

function RankingsPage() {
  const { id } = useParams();
  const [rankings, setRankings] = useState([]);
  const [importance, setImportance] = useState([]);

  useEffect(() => {
    api.getRankings(id, 200).then(setRankings).catch(() => {});
    api.getFeatureImportance(id).then(setImportance).catch(() => {});
  }, [id]);

  const tierColor = (tier) => {
    const colors = { CRITICAL: 'var(--accent-red)', HIGH: 'var(--accent-orange)', MEDIUM: 'var(--accent-yellow)', LOW: 'var(--accent-green)' };
    return colors[tier] || 'var(--text-secondary)';
  };

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-header__title">Fraud Rankings</h1>
        <p className="page-header__subtitle">XGBoost ensemble — top {rankings.length} students by fraud probability</p>
      </div>

      {/* Feature Importance */}
      {importance.length > 0 && (
        <div className="card" style={{ marginBottom: '24px' }}>
          <h2 className="card__title" style={{ marginBottom: '16px' }}>Feature Importance</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {importance.slice(0, 8).map((item, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ width: '180px', fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                  {item.feature}
                </span>
                <div style={{ flex: 1, height: '8px', background: 'var(--bg-primary)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${item.importance * 100 / Math.max(...importance.map(x => x.importance))}%`, height: '100%', background: 'var(--gradient-cyan)', borderRadius: '4px' }} />
                </div>
                <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', width: '50px', textAlign: 'right' }}>
                  {(item.importance * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Rankings Table */}
      <div className="card">
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Student ID</th>
                <th>Fraud Probability</th>
                <th>Risk Tier</th>
                <th>Engines Flagged</th>
                <th>Center</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((r, i) => (
                <tr key={r.student_id}>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{i + 1}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{r.student_id}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ width: '80px', height: '6px', background: 'var(--bg-primary)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${r.fraud_probability * 100}%`, height: '100%', background: r.fraud_probability > 0.8 ? 'var(--accent-red)' : r.fraud_probability > 0.6 ? 'var(--accent-orange)' : 'var(--accent-yellow)', borderRadius: '3px' }} />
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{(r.fraud_probability * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td><span className={`badge badge--${r.risk_tier.toLowerCase()}`}>{r.risk_tier}</span></td>
                  <td style={{ fontSize: '11px' }}>{(r.engines_flagged || []).join(', ') || '—'}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>{r.center_id || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// Graph Page (placeholder)
// ═══════════════════════════════════════════════

function GraphPage() {
  const { id } = useParams();
  const [graph, setGraph] = useState(null);

  useEffect(() => {
    api.getGraph(id).then(setGraph).catch(() => {});
  }, [id]);

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-header__title">Network Graph</h1>
        <p className="page-header__subtitle">Similarity network from copy ring detection</p>
      </div>
      <div className="card">
        {graph ? (
          <div>
            <div className="stats-grid" style={{ marginBottom: '16px' }}>
              <div className="stat-card stat-card--cyan">
                <div className="stat-card__value">{graph.nodes?.length || 0}</div>
                <div className="stat-card__label">Nodes</div>
              </div>
              <div className="stat-card stat-card--purple">
                <div className="stat-card__value">{graph.edges?.length || 0}</div>
                <div className="stat-card__label">Edges</div>
              </div>
            </div>
            <div className="table-container" style={{ maxHeight: '500px', overflow: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr><th>Node ID</th><th>Fraud Prob</th><th>Cluster</th><th>Flagged</th></tr>
                </thead>
                <tbody>
                  {(graph.nodes || []).slice(0, 100).map(node => (
                    <tr key={node.id}>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{node.id}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: (node.fraud_prob || 0) > 0.5 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                        {((node.fraud_prob || 0) * 100).toFixed(1)}%
                      </td>
                      <td>{node.cluster || '—'}</td>
                      <td>{node.is_flagged ? <span className="badge badge--critical">Yes</span> : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="loading-screen"><div className="spinner" /></div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════
// App Root
// ═══════════════════════════════════════════════

function AppLayout() {
  const location = useLocation();
  const match = location.pathname.match(/\/analysis\/([^/]+)/);
  const analysisId = match ? match[1] : null;

  return (
    <div className="app-container">
      <Sidebar analysisId={analysisId} />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/analysis/:id" element={<DashboardPage />} />
          <Route path="/analysis/:id/engines" element={<EnginesPage />} />
          <Route path="/analysis/:id/rankings" element={<RankingsPage />} />
          <Route path="/analysis/:id/graph" element={<GraphPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
