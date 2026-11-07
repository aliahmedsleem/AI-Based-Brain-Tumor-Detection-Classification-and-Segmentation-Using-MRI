import React, { useState, useRef } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, Brain, LayoutDashboard, Scan, FileText, Settings,
  UploadCloud, ShieldCheck, Database, AlertCircle, ZoomIn, ZoomOut, RotateCcw,
} from 'lucide-react'
import './style.css'

type StudyMetadata = {
  study_id: string
  filename: string
  shape: number[]
  original_shape: number[]
  volume_count: number
  selected_volume: number
  notice: string | null
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
  const sliceRequests = useRef<Partial<Record<Plane, AbortController>>>({})
  const [error, setError] = useState('')
  const [studyLabel, setStudyLabel] = useState('')
  const [studyNote, setStudyNote] = useState('')
  const [reportData, setReportData] = useState<any>(null)
  const [reportBusy, setReportBusy] = useState(false)
  const [reportError, setReportError] = useState('')
  const [volumeIndex, setVolumeIndex] = useState(0)
  const [sessionMessage, setSessionMessage] = useState('')
  const [preprocessSummary, setPreprocessSummary] = useState<{nonzero_voxels:number; normalized_mean:number; normalized_std:number; clipping_percentiles:number[]} | null>(null)
  const [preprocessBusy, setPreprocessBusy] = useState(false)
  const [segmentBusy, setSegmentBusy] = useState(false)
  const [segmentation, setSegmentation] = useState<{segmented_voxels:number; estimated_mask_volume_ml:number; notice:string} | null>(null)
  const [confirmFlair, setConfirmFlair] = useState(false)
  const [showOverlay, setShowOverlay] = useState(true)
  const [overlayOpacity, setOverlayOpacity] = useState(55)
  const [views, setViews] = useState<Record<Plane, View>>({ sagittal: defaultView(), coronal: defaultView(), axial: defaultView() })
  const drag = useRef<{ plane: Plane; x: number; y: number } | null>(null)
  function updateView(plane: Plane, patch: Partial<View>) {
    setViews(previous => ({ ...previous, [plane]: { ...previous[plane], ...patch } }))
  }

  async function inspect() {
    if (!file) return
    Object.values(sliceRequests.current).forEach(controller => controller?.abort())
    setBusy(true)
    setError('')
    setResult(null)
    setSegmentation(null)
    setReportData(null)
    setShowOverlay(true)
    setViews({ sagittal: defaultView(), coronal: defaultView(), axial: defaultView() })
    const data = new FormData()
    data.append('file', file)
    try {
      const response = await fetch(`/api/studies/inspect?volume_index=${volumeIndex}`, { method: 'POST', body: data })
      const body = await parseApiResponse(response, 'MRI inspection failed')
      setResult(body as StudyMetadata)
      setPreprocessSummary(null)
      setStudyLabel(file.name)
      setStudyNote('')
      setSessionMessage('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setBusy(false)
    }
  }

  async function changeSlice(plane: Plane, index: number, overlay = showOverlay, opacity = overlayOpacity) {
    if (!result) return
    const studyId = result.study_id
    sliceRequests.current[plane]?.abort()
    const controller = new AbortController()
    sliceRequests.current[plane] = controller
    setError('')
    setResult(previous => previous ? { ...previous, slice_indices: { ...previous.slice_indices, [plane]: index } } : null)
    try {
      const url = `/api/studies/${encodeURIComponent(studyId)}/slice/${plane}?index=${index}&overlay=${overlay}&opacity=${opacity / 100}`
      const response = await fetch(url, { signal: controller.signal })
      const body = await parseApiResponse(response, 'Unable to render MRI slice')
      if (controller.signal.aborted) return
      setResult(previous => previous?.study_id === studyId
        ? { ...previous, previews: { ...previous.previews, [plane]: body.preview } }
        : previous)
    } catch (err) {
      if (controller.signal.aborted) return
      setError(err instanceof Error ? err.message : 'Slice rendering failed')
    }
  }

  async function runPreprocessing() {
    if (!result) return
    setPreprocessBusy(true)
    setError('')
    try {
      const response = await fetch(`/api/studies/${encodeURIComponent(result.study_id)}/preprocess/summary`, { method: 'POST' })
      const body = await parseApiResponse(response, 'Preprocessing failed')
      setPreprocessSummary(body)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Preprocessing failed')
    } finally {
      setPreprocessBusy(false)
    }
  }

  async function refreshOverlay(next: boolean, opacity = overlayOpacity) {
    if (!result || !segmentation) return
    setShowOverlay(next)
    setOverlayOpacity(opacity)
    const studyId = result.study_id
    Object.values(sliceRequests.current).forEach(controller => controller?.abort())
    try {
      const planes: Plane[] = ['axial', 'coronal', 'sagittal']
      const images = await Promise.all(planes.map(async plane => {
        const index = result.slice_indices[plane]
        const response = await fetch(`/api/studies/${encodeURIComponent(studyId)}/slice/${plane}?index=${index}&overlay=${next}&opacity=${opacity / 100}`)
        const body = await parseApiResponse(response, 'Unable to update overlay')
        return [plane, body.preview] as const
      }))
      setResult(previous => previous?.study_id === studyId ? {
        ...previous, previews: { ...previous.previews, ...Object.fromEntries(images) }
      } : previous)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to toggle overlay')
    }
  }

  async function loadResearchReport() {
    if (!result || !segmentation) return
    setReportBusy(true)
    setReportError('')
    try {
      const response = await fetch(`/api/studies/${encodeURIComponent(result.study_id)}/ai/report`)
      const body = await parseApiResponse(response, 'Unable to load research report')
      setReportData(body)
    } catch (err) {
      setReportError(err instanceof Error ? err.message : 'Report loading failed')
    } finally {
      setReportBusy(false)
    }
  }

  async function exportMask() {
    if (!result || !segmentation) return
    setError('')
    try {
      const response = await fetch(`/api/studies/${encodeURIComponent(result.study_id)}/ai/mask.nii.gz`)
      if (!response.ok) {
        await parseApiResponse(response, 'Unable to export NIfTI mask')
        return
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'neuroscope_whole_tumor_mask.nii.gz'
      link.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'NIfTI export failed')
    }
  }

  async function exportReport() {
    if (!result || !segmentation) return
    setError('')
    try {
      const response = await fetch(`/api/studies/${encodeURIComponent(result.study_id)}/ai/report`)
      const body = await parseApiResponse(response, 'Unable to export research report')
      const blob = new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'neuroscope_research_report.json'
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report export failed')
    }
  }

  async function runSegmentation() {
    if (!result || !confirmFlair) return
    const studyId = result.study_id
    setSegmentBusy(true)
    setError('')
    try {
      const response = await fetch(`/api/studies/${encodeURIComponent(studyId)}/ai/segment?modality=flair`, { method: 'POST' })
      const body = await parseApiResponse(response, 'Research segmentation failed')
      Object.values(sliceRequests.current).forEach(controller => controller?.abort())
      setResult(previous => previous?.study_id === studyId
        ? { ...previous, previews: body.previews, slice_indices: body.slice_indices } : previous)
      setSegmentation(body)
      setReportData(null)
      setShowOverlay(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Research segmentation failed')
    } finally {
      setSegmentBusy(false)
    }
  }

  async function releaseStudy() {
    if (!result) return
    const studyId = result.study_id
    try {
      const response = await fetch(`/api/studies/${encodeURIComponent(studyId)}`, { method: 'DELETE' })
      await parseApiResponse(response, 'Unable to release study')
      Object.values(sliceRequests.current).forEach(controller => controller?.abort())
      setResult(null)
      setPreprocessSummary(null)
      setSegmentation(null)
      setReportData(null)
      setFile(null)
      setStudyLabel('')
      setStudyNote('')
      setSessionMessage('Study released from server memory.')
    } catch (err) {
      setSessionMessage(err instanceof Error ? err.message : 'Unable to release study')
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
      {page === 'Reports' ? <section className="panel report-page">
        <div className="panel-heading"><div><h2>Research Segmentation Report</h2>
          <p>Session-based experimental MRI analysis. No clinical interpretation or persistent patient records.</p></div>
          <span className="tag">Research only</span></div>
        {!result || !segmentation ? <div className="study-empty">
          <FileText size={34}/><p>No completed segmentation in the active session. Run FLAIR research segmentation first.</p>
          <button className="primary" onClick={() => setPage('MRI Analysis')}>Open MRI Analysis</button>
        </div> : <>
          <div className="study-actions report-actions">
            <button className="primary" disabled={reportBusy} onClick={loadResearchReport}>
              {reportBusy ? 'Loading…' : reportData ? 'Refresh Report' : 'Generate Research Report'}
            </button>
            <button className="secondary" disabled={!reportData} onClick={() => window.print()}>Print / Save as PDF</button>
            <button className="secondary" onClick={exportReport}>Download JSON</button>
            <button className="secondary" onClick={exportMask}>Download NIfTI Mask</button>
          </div>
          {reportError && <p className="error" role="alert">{reportError}</p>}
          {reportData && <article className="report-document">
            <div className="report-topline"><Brain size={30}/><div><strong>NeuroScope AI</strong><span>Experimental MRI Research Report</span></div></div>
            <h3>Whole-Tumor Segmentation Summary</h3>
            <p className="report-warning"><b>RESEARCH USE ONLY.</b> Not for clinical diagnosis, treatment planning, or patient management.</p>
            <div className="report-stats">
              <div><span>Estimated mask volume</span><strong>{reportData.estimated_mask_volume_ml.toFixed(3)} mL</strong></div>
              <div><span>Segmented voxels</span><strong>{reportData.segmented_voxels.toLocaleString()}</strong></div>
              <div><span>Threshold</span><strong>{reportData.threshold.toFixed(2)}</strong></div>
            </div>
            <h4>Study & Model Details</h4>
            <dl className="report-details">
              <div><dt>Research study label</dt><dd>{studyLabel || 'Untitled session'}</dd></div>
              <div><dt>Input modality</dt><dd>{reportData.modality}</dd></div>
              <div><dt>Segmentation task</dt><dd>{reportData.task.replaceAll('_', ' ')}</dd></div>
              <div><dt>Volume dimensions</dt><dd>{reportData.image_shape.join(' × ')} voxels</dd></div>
              <div><dt>Voxel spacing</dt><dd>{reportData.voxel_spacing_mm.join(' × ')} mm</dd></div>
              <div><dt>Postprocessing</dt><dd>{reportData.postprocessing.replaceAll('_', ' ')}</dd></div>
            </dl>
            <h4>Selected MRI Views</h4>
            <div className="report-images">{(['axial','coronal','sagittal'] as const).map(plane =>
              <figure key={plane}><img src={result.previews[plane]} alt={`${plane} research MRI preview`}/><figcaption>{plane.toUpperCase()} · slice {result.slice_indices[plane] + 1}</figcaption></figure>
            )}</div>
            {studyNote && <><h4>Research Notes (local session)</h4><p className="report-notes">{studyNote}</p></>}
            <h4>Methodological Limitations</h4>
            <ul>{reportData.limitations.map((item:string) => <li key={item}>{item}</li>)}</ul>
            <p className="report-footnote">Generated from the current temporary study session. The displayed mask is an experimental model prediction, not a verified tumor boundary.</p>
          </article>}
        </>}
      </section> : page === 'Studies' ? <section className="panel">
        <div className="panel-heading"><div><h2>Active MRI Study</h2>
          <p>Manage the current temporary research session. No patient registry or permanent storage is enabled.</p></div>
          <span className="tag">Session only</span></div>
        {result ? <div className="study-manager">
          <div className="study-fields">
            <label>Study label<input maxLength={100} value={studyLabel} onChange={e => setStudyLabel(e.target.value)} placeholder="Research study label" /></label>
            <label>Research notes (local to this page)<textarea maxLength={1000} rows={4} value={studyNote} onChange={e => setStudyNote(e.target.value)} placeholder="Optional notes; avoid patient identifiers" /></label>
          </div>
          <div className="study-summary">
            <p><b>Original file:</b> {result.filename}</p>
            <p><b>Volume:</b> {result.shape.join(' × ')} voxels</p>
            <p><b>Orientation:</b> {result.orientation}</p>
            <p><b>AI analysis:</b> {segmentation ? `Experimental mask: ${segmentation.estimated_mask_volume_ml} mL` : "Not run"}</p>
            <p><b>Retention:</b> Temporary in-memory session, expires after 10 minutes idle; one active study per server.</p>
          </div>
          <div className="study-actions">
            <button className="primary" onClick={() => setPage('MRI Analysis')}>Open MRI Viewer</button>
            <button className="danger-button" onClick={releaseStudy}>Release Study</button>
          </div>
        </div> : <div className="study-empty"><Database size={34} /><p>No active study. Inspect a NIfTI volume to start a temporary session.</p>
          <button className="primary" onClick={() => setPage('MRI Analysis')}>Upload MRI</button></div>}
        {sessionMessage && <p role="status">{sessionMessage}</p>}
      </section> : page === 'Dashboard' ? <>
        <section className="notice"><AlertCircle size={21} /><div><b>Development environment</b>
          <div>Experimental local AI segmentation is available when a compatible checkpoint is configured. No diagnostic results are generated.</div></div></section>
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
            <p>MRI inspection and optional research-only FLAIR whole-tumor segmentation. Classification is not available.</p></div>
            <span className="tag">NIfTI</span></div>
          <label className="drop"><UploadCloud size={36} />
            <strong>{file ? file.name : 'Choose a brain MRI volume'}</strong>
            <span>Supported: .nii, .nii.gz · Up to 256 MiB</span>
            <input type="file" accept=".nii,.gz"
              onChange={e => { Object.values(sliceRequests.current).forEach(controller => controller?.abort()); setFile(e.target.files?.[0] || null); setResult(null); setSegmentation(null); setConfirmFlair(false); setVolumeIndex(0); setError('') }} />
          </label>
          <label className="volume-selector">4D volume index (0-based; use 0 for ordinary 3D MRI)
            <input type="number" min={0} max={999} step={1} value={volumeIndex} disabled={busy}
              onChange={e => setVolumeIndex(Math.max(0, Math.floor(Number(e.target.value) || 0)))} />
          </label>
          <button className="primary" disabled={!file || busy} onClick={inspect}>
            {busy ? 'Inspecting…' : 'Inspect MRI Metadata'}
          </button>
          {error && <div role="alert" className="error">{error}</div>}
          {result && <div className="result"><h3>Image metadata</h3>
            {result.notice && <p role="status" className="volume-notice">{result.notice}</p>}
            <p>Source dimensions: {result.original_shape.join(' × ')}</p>
            <p>Selected volume: {result.selected_volume + 1} of {result.volume_count}</p>
            <p>Dimensions: {result.shape.join(' × ')}</p>
            <p>Voxel spacing: {result.voxel_spacing_mm.map(v => v.toFixed(2)).join(' × ')} mm</p>
            <p>Orientation: {result.orientation} (canonical)</p>
            <div className="preprocess-panel">
              <h3>Experimental AI segmentation</h3>
              <p>Research-only binary whole-tumor mask. Requires a locally configured trained model and a genuine FLAIR MRI volume. Not for diagnosis.</p>
              <label><input type="checkbox" checked={confirmFlair} onChange={e => setConfirmFlair(e.target.checked)} /> I confirm this volume is FLAIR MRI and contains no patient identifiers.</label>
              <div><button className="primary" disabled={!confirmFlair || segmentBusy} onClick={runSegmentation}>
                {segmentBusy ? 'Segmenting on GPU…' : 'Run Research Segmentation'}
              </button></div>
              {segmentation && <div role="status">
                <p><b>Estimated mask volume:</b> {segmentation.estimated_mask_volume_ml} mL</p>
                <p><b>Segmented voxels:</b> {segmentation.segmented_voxels.toLocaleString()}</p>
                <p><b>Threshold:</b> 0.90 · Largest connected component</p>
                <p>{segmentation.notice}</p>
                <label><input type="checkbox" checked={showOverlay} onChange={e => refreshOverlay(e.target.checked)} /> Show red segmentation overlay</label>
                <label>Overlay opacity: {overlayOpacity}%
                  <input type="range" min={0} max={100} step={5} value={overlayOpacity}
                    disabled={!showOverlay} onChange={e => setOverlayOpacity(Number(e.target.value))}
                    onPointerUp={() => refreshOverlay(showOverlay, overlayOpacity)}
                    onKeyUp={() => refreshOverlay(showOverlay, overlayOpacity)} />
                </label>
                <div className="study-actions">
                  <button className="secondary" onClick={exportReport}>Export Research Report (JSON)</button>
                  <button className="secondary" onClick={exportMask}>Export Segmentation Mask (.nii.gz)</button>
                </div>
              </div>}
            </div>
            <div className="preprocess-panel">
              <h3>Research preprocessing</h3>
              <p>Inspect intensity normalization statistics. This does not detect, classify or segment tumors.</p>
              <button className="secondary" disabled={preprocessBusy} onClick={runPreprocessing}>
                {preprocessBusy ? 'Processing…' : 'Run Preprocessing Check'}
              </button>
              {preprocessSummary && <div className="preprocess-stats">
                <p>Nonzero voxels: {preprocessSummary.nonzero_voxels.toLocaleString()}</p>
                <p>Normalized mean: {preprocessSummary.normalized_mean.toFixed(4)}</p>
                <p>Normalized standard deviation: {preprocessSummary.normalized_std.toFixed(4)}</p>
                <p>Clipping percentiles: {preprocessSummary.clipping_percentiles.join(' – ')}</p>
              </div>}
            </div>
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
                    value={result.slice_indices[plane]} disabled={busy}
                    onChange={e => changeSlice(plane, Number(e.target.value))} aria-label={`${plane} slice`} /></label>
              </figure>
            )}</div>
            <p className="preview-note">Interactive slice navigation, zoom, drag-to-pan, brightness and contrast. Slices are rendered from a temporary in-memory session without re-uploading the volume. Sessions expire after 10 minutes of inactivity. Red overlay indicates experimental segmentation when enabled; not a diagnosis.</p>
          </div>}
        </section> :
        <section className="panel empty"><h2>{page}</h2>
          <p>This module is planned for the next development phase. No patient records or AI outputs are fabricated.</p>
          <button className="secondary" onClick={() => setPage('MRI Analysis')}>Go to MRI Analysis</button>
        </section>}
      <footer>NeuroScope AI · Graduation Research Project · Prototype v0.12</footer>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
