/**
 * QRC control-plane store — Pinia.
 *
 * Talks to the ``/api/qrc`` blueprint (Coder C, backend) per
 * ``docs/QRC/QRC_StageA_Contracts.md`` §3. Read-only endpoints
 * (``GET /runs``, ``.../progress``, ``.../fid``, ``.../results``,
 * ``.../status``) are unauthenticated; the control endpoints
 * (``POST /runs``, ``POST /runs/<id>/stop``) require the session cookie,
 * exactly like ``qml``'s ``startTraining`` — ``axios.create({ withCredentials
 * : true })`` ships the cookie automatically, no bearer token needed.
 *
 * Dev-time fallback: while Coder C's blueprint is being built in parallel
 * (or the dev-box tunnel is down), any *network*-level failure (no HTTP
 * response at all — connection refused, DNS, CORS preflight failure) on a
 * read-only GET falls back to local mock fixtures that match the §3 shapes
 * exactly, so the UI always has something to render. A real HTTP error
 * (4xx/5xx with a response) is NOT masked — that's a genuine backend
 * answer (e.g. "run not found") and must surface to the user as-is.
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import axios, { AxiosError } from 'axios'

const api = axios.create({ withCredentials: true })

/** True only for "the request never reached a server" failures. */
function isNetworkError(e: unknown): boolean {
  const err = e as AxiosError
  return !!err?.isAxiosError && !err.response
}

// ---------------------------------------------------------------------
// Types — mirror §3 JSON shapes exactly.
// ---------------------------------------------------------------------

export type QrcTask = 'trace-gen' | 'narma' | 'weather' | 'phase1'

// Must match the backend launcher vocabulary exactly (app/qrc/launcher.py):
// a run is 'running', then 'done' | 'stopped' | 'error'.
export type QrcRunStatus =
  | 'running'
  | 'done'
  | 'stopped'
  | 'error'

export interface QrcRunSummary {
  id: string
  task: QrcTask
  status: QrcRunStatus
  progress_pct: number
  eta_s: number | null
  created_utc: string
}

export interface QrcProgressEvent {
  t: string
  elapsed_s: number
  kind: string
  message: string
}

export interface QrcProgress {
  phase: string
  step: number
  total: number
  eta_s: number | null
  status: QrcRunStatus
  events: QrcProgressEvent[]
  results?: Record<string, any> | null
}

export interface QrcFidData {
  step: number
  n_steps: number
  t: number[]
  real: number[]
  imag: number[]
  mag: number[]
  freq_hz: number[]
  spectrum_mag: number[]
  peaks_hz: number[]
  /** Provenance: the exact saved waveform file this FID/spectrum came from. */
  trace_name?: string
  trace_path?: string
  live?: boolean
  /** Sweep runs expose one saved trace per encoding, for a UI picker. */
  available_traces?: { fn: string; name: string }[]
}

/** Loosely typed — §3 says "the results JSON" without pinning every key.
 *  Known fields (from §2's ``evaluate_features`` protocol dict) are typed;
 *  anything else rides along under the index signature so the UI never
 *  chokes on an unrecognized shape. */
export interface QrcResults {
  experiment?: string
  feature_count?: number
  weather_r2?: Record<string, number>
  narma10_nmse?: number
  wall_clock_s?: number
  delta_vs_baseline?: Record<string, number | string>
  esn_comparison?: Record<string, any>
  [key: string]: any
}

export interface QrcRunStatusInfo {
  status: QrcRunStatus
  exit_code: number | null
}

/** 2-D feature-space projection from ``GET /runs/<id>/embedding``.
 *  ``points[i] = [x, y]`` is step ``i``'s reservoir feature vector projected
 *  to 2-D; ``color[i]`` is that step's target (for a continuous colour map)
 *  and ``split[i]`` its washout/train/test membership. */
export interface QrcEmbedding {
  method: 'pca' | 'umap'
  feature: 'magnitude653' | 'phase' | 'multimodal'
  n_features_in: number
  n_points: number
  points: number[][]
  color: number[]
  color_label: string
  split: string[]
  /** Provenance: the saved waveform file this embedding was built from. */
  trace_name?: string
  trace_path?: string
}

/** One row of a phase1 ``summary.json`` (the sweep the Feature-Lab charts
 *  render): a feature-set experiment or a selection/reducer point. */
export interface QrcSummaryRow {
  experiment: string
  kind: 'feature' | 'selection'
  feature_count: number | null
  metric: number | null
  reducer?: string
  reducer_k?: number | null
}

const RUNNING: QrcRunStatus[] = ['running']

// ---------------------------------------------------------------------
// Mock fixtures — same shapes as §3, used only when the API is
// unreachable (see isNetworkError above).
// ---------------------------------------------------------------------

const MOCK_RUNS: QrcRunSummary[] = [
  {
    id: 'mock-weather-001',
    task: 'weather',
    status: 'running',
    progress_pct: 62,
    eta_s: 1840,
    created_utc: new Date(Date.now() - 3 * 3600_000).toISOString(),
  },
  {
    id: 'mock-narma-001',
    task: 'narma',
    status: 'done',
    progress_pct: 100,
    eta_s: 0,
    created_utc: new Date(Date.now() - 26 * 3600_000).toISOString(),
  },
  {
    id: 'mock-phase1-001',
    task: 'phase1',
    status: 'error',
    progress_pct: 34,
    eta_s: null,
    created_utc: new Date(Date.now() - 5 * 3600_000).toISOString(),
  },
]

function mockProgress(id: string): QrcProgress {
  const run = MOCK_RUNS.find((r) => r.id === id) ?? MOCK_RUNS[0]
  const events: QrcProgressEvent[] = [
    { t: new Date(Date.now() - 600_000).toISOString(), elapsed_s: 0, kind: 'start', message: `Launched ${run.task} run` },
    { t: new Date(Date.now() - 480_000).toISOString(), elapsed_s: 120, kind: 'phase', message: 'Loading cached trace' },
    { t: new Date(Date.now() - 300_000).toISOString(), elapsed_s: 300, kind: 'step', message: 'Evolving reservoir — step 320/900' },
    { t: new Date(Date.now() - 60_000).toISOString(), elapsed_s: 540, kind: 'step', message: 'Evolving reservoir — step 610/900' },
  ]
  return {
    phase: run.status === 'done' ? 'done' : run.status === 'error' ? 'failed' : 'evolving',
    step: Math.round((run.progress_pct / 100) * 900),
    total: 900,
    eta_s: run.eta_s,
    status: run.status,
    events,
    results: run.status === 'done' ? mockResults(id) : null,
  }
}

const FID_POINTS = 2048
const FID_DWELL = 3e-4
const MOCK_N_STEPS = 50

/** Deterministic synthetic FID: a couple of decaying sinusoids + a
 *  small amount of seeded pseudo-noise, so every step looks plausible
 *  but distinct (mirrors a T2*-decaying NMR free-induction decay). */
function mockFid(step: number): QrcFidData {
  const n = FID_POINTS
  const dwell = FID_DWELL
  const seedFreqs = [800 + step * 6, 1500 - step * 3, 2300 + step * 2]
  const t: number[] = new Array(n)
  const real: number[] = new Array(n)
  const imag: number[] = new Array(n)
  const mag: number[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const time = i * dwell
    t[i] = time
    let re = 0
    let im = 0
    for (const f of seedFreqs) {
      const decay = Math.exp(-time / 0.03)
      const phase = 2 * Math.PI * f * time
      re += decay * Math.cos(phase)
      im += decay * Math.sin(phase)
    }
    // tiny deterministic jitter so it doesn't look perfectly analytic
    const jitter = Math.sin(i * 12.9898 + step * 78.233) * 0.02
    re += jitter
    im += jitter * 0.5
    real[i] = re
    imag[i] = im
    mag[i] = Math.sqrt(re * re + im * im)
  }
  // Naive DFT magnitude at 653 selected bins spanning 0..1/(2*dwell).
  const nyquist = 1 / (2 * dwell)
  const nBins = 653
  const freq_hz: number[] = new Array(nBins)
  const spectrum_mag: number[] = new Array(nBins)
  for (let k = 0; k < nBins; k++) {
    const f = (k / (nBins - 1)) * nyquist
    freq_hz[k] = f
    let sre = 0
    let sim = 0
    for (let i = 0; i < n; i += 4) {
      // subsample the DFT sum for mock-perf; good enough for a fake plot
      const ang = -2 * Math.PI * f * t[i]
      sre += real[i] * Math.cos(ang) - imag[i] * Math.sin(ang)
      sim += real[i] * Math.sin(ang) + imag[i] * Math.cos(ang)
    }
    spectrum_mag[k] = Math.sqrt(sre * sre + sim * sim)
  }
  const peaks_hz = seedFreqs.filter((f) => f >= 0 && f <= nyquist)
  return {
    step, n_steps: MOCK_N_STEPS, t, real, imag, mag, freq_hz, spectrum_mag, peaks_hz,
    trace_name: 'mock:synthetic-fid (backend unreachable)',
  }
}

function mockResults(_id: string): QrcResults {
  return {
    experiment: 'mock_baseline_magnitude653',
    feature_count: 653,
    weather_r2: { h1: 0.95, h10: 0.888, h20: 0.86, h30: 0.812, h45: 0.786 },
    narma10_nmse: 2.46e-5,
    wall_clock_s: 5400,
    delta_vs_baseline: { h1: '+0.000', h10: '+0.000', h20: '+0.000', h30: '+0.000', h45: '+0.000' },
    esn_comparison: {
      h1: { qrc: 0.95, qrc_rbf: 0.953, best_esn: 0.956 },
      h10: { qrc: 0.888, qrc_rbf: 0.866, best_esn: 0.849 },
      h20: { qrc: 0.86, qrc_rbf: 0.815, best_esn: 0.82 },
      h30: { qrc: 0.812, qrc_rbf: 0.834, best_esn: 0.752 },
      h45: { qrc: 0.786, qrc_rbf: 0.778, best_esn: 0.697 },
    },
  }
}

/** Deterministic synthetic embedding: two temperature-graded gaussian
 *  blobs so the scatter + colour map render when the API is unreachable. */
function mockEmbedding(method: 'pca' | 'umap', feature: QrcEmbedding['feature']): QrcEmbedding {
  const n = 180
  const points: number[][] = []
  const color: number[] = []
  const split: string[] = []
  for (let i = 0; i < n; i++) {
    const s = Math.sin(i * 12.9898) * 43758.5453
    const r1 = s - Math.floor(s)
    const s2 = Math.sin(i * 78.233) * 12543.1234
    const r2 = s2 - Math.floor(s2)
    const blob = i % 2
    const cx = blob ? 2.4 : -2.4
    points.push([cx + (r1 - 0.5) * 3, (r2 - 0.5) * 3])
    color.push(blob ? 0.3 + r1 * 0.4 : 0.6 + r2 * 0.4)
    split.push(i < 30 ? 'washout' : i < 130 ? 'train' : 'test')
  }
  return {
    method,
    feature,
    n_features_in: feature === 'magnitude653' ? 653 : feature === 'phase' ? 1959 : 1977,
    n_points: n,
    points,
    color,
    color_label: 'temperature (norm)',
    split,
    trace_name: 'mock:synthetic-embedding (backend unreachable)',
  }
}

// ---------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------

export const useQrcStore = defineStore('qrc', () => {
  const runs = ref<QrcRunSummary[]>([])
  const usingMockRuns = ref(false)

  const currentProgress = ref<QrcProgress | null>(null)
  const usingMockProgress = ref(false)

  const currentFid = ref<QrcFidData | null>(null)
  const usingMockFid = ref(false)

  const currentResults = ref<QrcResults | null>(null)
  const usingMockResults = ref(false)

  const currentEmbedding = ref<QrcEmbedding | null>(null)
  const usingMockEmbedding = ref(false)

  const error = ref<string | null>(null)

  let runsPollTimer: ReturnType<typeof setInterval> | null = null
  let progressPollTimer: ReturnType<typeof setInterval> | null = null

  const anyRunActive = computed(() =>
    runs.value.some((r) => RUNNING.includes(r.status)),
  )

  async function loadRuns() {
    try {
      const r = await api.get<QrcRunSummary[]>('/api/qrc/runs')
      runs.value = r.data
      usingMockRuns.value = false
    } catch (e) {
      if (isNetworkError(e)) {
        runs.value = MOCK_RUNS
        usingMockRuns.value = true
      } else {
        throw e
      }
    }
  }

  function startRunsPolling(intervalMs = 5000) {
    stopRunsPolling()
    runsPollTimer = setInterval(() => {
      loadRuns().catch(() => {})
    }, intervalMs)
  }

  function stopRunsPolling() {
    if (runsPollTimer) {
      clearInterval(runsPollTimer)
      runsPollTimer = null
    }
  }

  async function loadProgress(id: string) {
    try {
      const r = await api.get<QrcProgress>(`/api/qrc/runs/${id}/progress`)
      currentProgress.value = r.data
      usingMockProgress.value = false
    } catch (e) {
      if (isNetworkError(e)) {
        currentProgress.value = mockProgress(id)
        usingMockProgress.value = true
      } else {
        throw e
      }
    }
  }

  /** Poll progress every ~3s while the run is active (§0.D spec). Stops
   *  itself once the run reaches a terminal status. */
  function startProgressPolling(id: string, intervalMs = 3000) {
    stopProgressPolling()
    progressPollTimer = setInterval(async () => {
      await loadProgress(id).catch(() => {})
      const s = currentProgress.value?.status
      if (s && !RUNNING.includes(s)) {
        stopProgressPolling()
      }
    }, intervalMs)
  }

  function stopProgressPolling() {
    if (progressPollTimer) {
      clearInterval(progressPollTimer)
      progressPollTimer = null
    }
  }

  async function loadFid(id: string, step: number, trace?: string) {
    try {
      const params: Record<string, unknown> = { step }
      if (trace) params.trace = trace
      const r = await api.get<QrcFidData>(`/api/qrc/runs/${id}/fid`, { params })
      currentFid.value = r.data
      usingMockFid.value = false
    } catch (e) {
      if (isNetworkError(e)) {
        currentFid.value = mockFid(step)
        usingMockFid.value = true
      } else {
        throw e
      }
    }
  }

  async function loadResults(id: string) {
    try {
      const r = await api.get<QrcResults>(`/api/qrc/runs/${id}/results`)
      currentResults.value = r.data
      usingMockResults.value = false
    } catch (e) {
      if (isNetworkError(e)) {
        currentResults.value = mockResults(id)
        usingMockResults.value = true
      } else {
        throw e
      }
    }
  }

  /** GET /api/qrc/runs/<id>/embedding — 2-D projection of the run's feature
   *  vectors. ``method`` pca|umap, ``feature`` magnitude653|phase|multimodal.
   *  UMAP can take a few seconds server-side (it's computed on demand from the
   *  trace); the caller drives loading state. A real HTTP error (e.g. UMAP
   *  extra missing → 503) surfaces to the caller; only a network drop mocks. */
  async function loadEmbedding(
    id: string,
    method: 'pca' | 'umap' = 'pca',
    feature: QrcEmbedding['feature'] = 'multimodal',
  ): Promise<QrcEmbedding> {
    try {
      const r = await api.get<QrcEmbedding>(`/api/qrc/runs/${id}/embedding`, {
        params: { method, feature },
      })
      currentEmbedding.value = r.data
      usingMockEmbedding.value = false
      return r.data
    } catch (e) {
      if (isNetworkError(e)) {
        const m = mockEmbedding(method, feature)
        currentEmbedding.value = m
        usingMockEmbedding.value = true
        return m
      }
      throw e
    }
  }

  async function loadStatus(id: string): Promise<QrcRunStatusInfo> {
    const r = await api.get<QrcRunStatusInfo>(`/api/qrc/runs/${id}/status`)
    return r.data
  }

  /** POST /api/qrc/runs — auth-gated. Enforces (server-side) a single
   *  active heavy job; a 409 response is surfaced with a clear message
   *  rather than the raw axios error. */
  async function createRun(task: QrcTask, config: Record<string, any>): Promise<{ id: string }> {
    error.value = null
    try {
      const r = await api.post<{ id: string }>('/api/qrc/runs', { task, config })
      await loadRuns().catch(() => {})
      return r.data
    } catch (e: any) {
      if (e?.response?.status === 409) {
        const msg = e.response.data?.error
          || 'Another QRC run is already active — only one heavy job may run at a time.'
        error.value = msg
        throw new Error(msg)
      }
      const msg = e?.response?.data?.error || e?.message || 'Failed to launch run'
      error.value = msg
      throw new Error(msg)
    }
  }

  /** POST /api/qrc/runs/<id>/stop — auth-gated. */
  async function stopRun(id: string): Promise<void> {
    error.value = null
    try {
      await api.post(`/api/qrc/runs/${id}/stop`)
      await loadRuns().catch(() => {})
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Failed to stop run'
      error.value = msg
      throw new Error(msg)
    }
  }

  function reset() {
    stopRunsPolling()
    stopProgressPolling()
    currentProgress.value = null
    currentFid.value = null
    currentResults.value = null
    currentEmbedding.value = null
    usingMockProgress.value = false
    usingMockFid.value = false
    usingMockResults.value = false
    usingMockEmbedding.value = false
    error.value = null
  }

  return {
    runs,
    usingMockRuns,
    currentProgress,
    usingMockProgress,
    currentFid,
    usingMockFid,
    currentResults,
    usingMockResults,
    currentEmbedding,
    usingMockEmbedding,
    error,
    anyRunActive,
    loadRuns,
    startRunsPolling,
    stopRunsPolling,
    loadProgress,
    startProgressPolling,
    stopProgressPolling,
    loadFid,
    loadResults,
    loadEmbedding,
    loadStatus,
    createRun,
    stopRun,
    reset,
  }
})
