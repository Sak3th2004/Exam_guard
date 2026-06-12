import { StrictMode, Component } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Error Boundary for graceful crash handling
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ExamGuard UI Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', minHeight: '100vh', padding: '40px',
          fontFamily: 'Inter, system-ui, sans-serif', background: '#f5f7fa',
        }}>
          <div style={{
            background: '#fff', padding: '40px', borderRadius: '16px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.07)', maxWidth: '480px',
            textAlign: 'center', border: '1px solid #e2e8f0',
          }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
            <h1 style={{ fontSize: '20px', fontWeight: 700, color: '#1e293b', marginBottom: '8px' }}>
              Something went wrong
            </h1>
            <p style={{ fontSize: '14px', color: '#64748b', marginBottom: '20px' }}>
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => { this.setState({ hasError: false }); window.location.href = '/'; }}
              style={{
                padding: '10px 24px', background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                color: 'white', border: 'none', borderRadius: '8px', fontSize: '14px',
                fontWeight: 600, cursor: 'pointer',
              }}
            >
              Return Home
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
