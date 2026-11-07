import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, Brain, LayoutDashboard, Scan, FileText, Settings,
  UploadCloud, ShieldCheck, Database, AlertCircle,
} from 'lucide-react'
import './style.css'

type StudyMetadata = {
  filename: string
  shape: number[]
  voxel_spacing_mm: number[]
  analysis_available: false
}

const items = [
  { name: 'Dashboard', icon: LayoutDashboard },
  { name: 'MRI Analysis', icon: Scan },
  { name: 'Studies', icon: Database },
  { name: 'Reports', icon: FileText },
  { name: 'Settings', icon: Settings },
]

function App() {
  const [page, setPage] = useState('Dashboard')
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<StudyMetadata | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function inspect() {
    if (!file) return
    setBusy(true)
    setError('')
    setResult(null)
    const data = new FormData()
    data.append('file', file)
    try {
      const response = await fetch('/api/studies/inspect', { method: 'POST', body: data })
      const body = await response.json()
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Inspection failed')
      setResult(body as StudyMetadata)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setBusy(false)
    }
  }

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon"><Brain size={23} /></div>
        <div><b>NeuroScope <span>AI</span></b><small>RESEARCH WORKSTATION</small></div>
      </div>
      <nav aria-label="Main navigation">{items.map(({ name, icon: Icon }) =>
        <button key={name} onClick={() => setPage(name)} className={page === name ? 'active' : ''}>
          <Icon size={19} />{name}
        </button>
      )}</nav>
      <div className="sidebar-bottom"><ShieldCheck size={17} /> Research use only<br />Not for clinical diagnosis</div>
    </aside>
    <main>
      <header>
        <div><div className="eyebrow">MEDICAL IMAGING / WORKSPACE</div>
          <h1>{page}</h1><p>AI-assisted brain MRI research platform</p></div>
        <div className="avatar" aria-label="Research user">R</div>
      </header>
      {page === 'Dashboard' ? <>
        <section className="notice"><AlertCircle size={21} /><div><b>Development environment</b>
          <div>AI inference is not connected yet. No diagnostic results are generated.</div></div></section>
        <div className="metrics">
          {([
            ['Studies processed', '0', Database],
            ['Completed analyses', '0', Activity],
            ['Models deployed', '0', Brain],
            ['Pending reviews', '0', FileText],
          ] as const).map(([label, value, Icon]) =>
            <article className="metric" key={label}>
              <div className="metric-icon"><Icon size={21} /></div>
              <span>{label}</span><strong>{value}</strong><small>Awaiting research data</small>
            </article>
          )}
        </div>
        <section className="panel"><h2>Start a new MRI study</h2>
          <p>Upload a NIfTI volume to inspect its dimensions and voxel spacing.</p>
          <button className="primary" onClick={() => setPage('MRI Analysis')}>Open MRI Workspace →</button>
        </section>
      </> : page === 'MRI Analysis' ?
        <section className="panel">
          <div className="panel-heading"><div><h2>MRI Study Inspection</h2>
            <p>Phase 1: metadata inspection. Segmentation and classification are not yet available.</p></div>
            <span className="tag">NIfTI</span></div>
          <label className="drop"><UploadCloud size={36} />
            <strong>{file ? file.name : 'Choose a brain MRI volume'}</strong>
            <span>Supported: .nii, .nii.gz · Up to 256 MiB</span>
            <input type="file" accept=".nii,.gz"
              onChange={e => { setFile(e.target.files?.[0] || null); setResult(null); setError('') }} />
          </label>
          <button className="primary" disabled={!file || busy} onClick={inspect}>
            {busy ? 'Inspecting…' : 'Inspect MRI Metadata'}
          </button>
          {error && <div role="alert" className="error">{error}</div>}
          {result && <div className="result"><h3>Image metadata</h3>
            <p>Dimensions: {result.shape.join(' × ')}</p>
            <p>Voxel spacing: {result.voxel_spacing_mm.map(v => v.toFixed(2)).join(' × ')} mm</p>
            <p>AI analysis: Not available in this phase</p>
          </div>}
        </section> :
        <section className="panel empty"><h2>{page}</h2>
          <p>This module is planned for the next development phase. No patient records or AI outputs are fabricated.</p>
          <button className="secondary" onClick={() => setPage('MRI Analysis')}>Go to MRI Analysis</button>
        </section>}
      <footer>NeuroScope AI · Graduation Research Project · Prototype v0.1</footer>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
