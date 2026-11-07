import React, { useState, useRef } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, Brain, LayoutDashboard, Scan, FileText, Settings,
  UploadCloud, ShieldCheck, Database, AlertCircle, ZoomIn, ZoomOut, RotateCcw,
} from 'lucide-react'
import './style.css'

type StudyMetadata = {
  filename: string
  shape: number[]
  voxel_spacing_mm: number[]
  analysis_available: false
  orientation: string
  previews: Record<'sagittal' | 'coronal' | 'axial', string>
  slice_indices: Record<'sagittal' | 'coronal' | 'axial', number>
}

type Plane = 'sagittal' | 'coronal' | 'axial'
type View = { zoom: number; x: number; y: number; brightness: number; contrast: number }
const defaultView = (): View => ({ zoom: 1, x: 0, y: 0, brightness: 100, contrast: 100 })

const items = [
  { name: 'Dashboard', icon: LayoutDashboard },
  { name: 'MRI Analysis', icon: Scan },
  { name: 'Studies', icon: Database },
  { name: 'Reports', icon: FileText },
  { name: 'Settings', icon: Settings },
]

async function parseApiResponse(response: Response, fallback: string) {
  const raw = await response.text()
  let body: any
  try {
    body = JSON.parse(raw)
  } catch {
    throw new Error(response.ok
      ? 'Server returned an invalid response.'
      : `Server error (HTTP ${response.status}). Check the FastAPI terminal for details.`)
  }
  if (!response.ok) {
    const detail = body?.detail
    throw new Error(typeof detail === 'string' ? detail : fallback)
  }
  return body
}

function App() {
  const [page, setPage] = useState('Dashboard')
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<StudyMetadata | null>(null)
  const [busy, setBusy] = useState(false)
  const [sliceBusy, setSliceBusy] = useState(false)
  const [error, setError] = useState('')
  const [views, setViews] = useState<Record<Plane, View>>({ sagittal: defaultView(), coronal: defaultView(), axial: defaultView() })
  const drag = useRef<{ plane: Plane; x: number; y: number } | null>(null)
  function updateView(plane: Plane, patch: Partial<View>) {
    setViews(previous => ({ ...previous, [plane]: { ...previous[plane], ...patch } }))
  }

  async function inspect() {
    if (!file) return
    setBusy(true)
    setError('')
    setResult(null)
    setViews({ sagittal: defaultView(), coronal: defaultView(), axial: defaultView() })
    const data = new FormData()
    data.append('file', file)
    try {
      const response = await fetch('/api/studies/inspect', { method: 'POST', body: data })
      const body = await parseApiResponse(response, 'MRI inspection failed')
      setResult(body as StudyMetadata)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setBusy(false)
    }
  }

  async function changeSlice(plane: 'sagittal' | 'coronal' | 'axial', index: number) {
    if (!file || !result || sliceBusy) return
    setSliceBusy(true)
    setError('')
    const next = { ...result.slice_indices, [plane]: index }
    const data = new FormData()
    data.append('file', file)
    try {
      const params = new URLSearchParams(Object.entries(next).map(([key, value]) => [key, String(value)]))
      const response = await fetch('/api/studies/slices?' + params.toString(), { method: 'POST', body: data })
      const body = await parseApiResponse(response, 'Unable to render MRI slice')
      setResult(previous => previous ? { ...previous, previews: body.previews, slice_indices: body.slice_indices } : null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Slice rendering failed')
    } finally {
      setSliceBusy(false)
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
            <p>Orientation: {result.orientation} (canonical)</p>
            <p>AI analysis: Not available in this phase</p>
            <div className="mri-grid">{(['axial','coronal','sagittal'] as const).map(plane =>
              <figure key={plane} className="mri-plane"><figcaption>{plane.toUpperCase()} · slice {result.slice_indices[plane] + 1}</figcaption>
                <div className="mri-viewport"
                  onPointerDown={e => { if (e.button !== 0) return; e.currentTarget.setPointerCapture(e.pointerId); drag.current = { plane, x: e.clientX, y: e.clientY } }}
                  onPointerMove={e => {
                    if (!drag.current || drag.current.plane !== plane) return
                    const { x, y } = drag.current
                    drag.current = { plane, x: e.clientX, y: e.clientY }
                    setViews(previous => ({ ...previous, [plane]: { ...previous[plane], x: previous[plane].x + e.clientX - x, y: previous[plane].y + e.clientY - y } }))
                  }}
                  onPointerUp={() => { drag.current = null }}
                  onPointerCancel={() => { drag.current = null }}
                  onWheel={e => { if (!e.ctrlKey) return; e.preventDefault(); updateView(plane, { zoom: Math.max(1, Math.min(5, views[plane].zoom + (e.deltaY < 0 ? .2 : -.2))) }) }}
                  aria-label={`${plane} MRI image; drag to pan`}>
                  <img draggable={false} src={result.previews[plane]} alt={`${plane} MRI slice preview`}
                    style={{ transform: `translate(${views[plane].x}px, ${views[plane].y}px) scale(${views[plane].zoom})`,
                      filter: `brightness(${views[plane].brightness}%) contrast(${views[plane].contrast}%)` }} />
                </div>
                <div className="viewer-tools">
                  <button type="button" aria-label={`Zoom in ${plane}`} onClick={() => updateView(plane, { zoom: Math.min(5, views[plane].zoom + .25) })}><ZoomIn size={17} /></button>
                  <span>{Math.round(views[plane].zoom * 100)}%</span>
                  <button type="button" aria-label={`Zoom out ${plane}`} onClick={() => updateView(plane, { zoom: Math.max(1, views[plane].zoom - .25) })}><ZoomOut size={17} /></button>
                  <button type="button" aria-label={`Reset ${plane} view`} onClick={() => updateView(plane, defaultView())}><RotateCcw size={17} /></button>
                </div>
                <div className="image-adjustments">
                  <label>Brightness <span>{views[plane].brightness}%</span><input type="range" min="50" max="200" value={views[plane].brightness}
                    onChange={e => updateView(plane, { brightness: Number(e.target.value) })} /></label>
                  <label>Contrast <span>{views[plane].contrast}%</span><input type="range" min="50" max="200" value={views[plane].contrast}
                    onChange={e => updateView(plane, { contrast: Number(e.target.value) })} /></label>
                </div>
                <label className="slice-control"><span>Slice {result.slice_indices[plane] + 1} / {result.shape[plane === 'sagittal' ? 0 : plane === 'coronal' ? 1 : 2]}</span>
                  <input type="range" min={0} max={result.shape[plane === 'sagittal' ? 0 : plane === 'coronal' ? 1 : 2] - 1}
                    value={result.slice_indices[plane]} disabled={sliceBusy}
                    onChange={e => changeSlice(plane, Number(e.target.value))} aria-label={`${plane} slice`} /></label>
              </figure>
            )}</div>
            <p className="preview-note">Interactive slice navigation, zoom, drag-to-pan, brightness and contrast. Slice changes still upload the volume; large files may respond slowly. No tumor segmentation or diagnostic inference.</p>
          </div>}
        </section> :
        <section className="panel empty"><h2>{page}</h2>
          <p>This module is planned for the next development phase. No patient records or AI outputs are fabricated.</p>
          <button className="secondary" onClick={() => setPage('MRI Analysis')}>Go to MRI Analysis</button>
        </section>}
      <footer>NeuroScope AI · Graduation Research Project · Prototype v0.4</footer>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
