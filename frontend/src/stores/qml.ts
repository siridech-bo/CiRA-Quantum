/**
 * QML training store — Pinia.
 *
 * Mirrors the optimization-side ``useSolveStore`` for the QML pipeline:
 * - submit training jobs (POST /api/qml/train)
 * - open the SSE channel and translate events into reactive state
 * - load past jobs by id (resting state after the stream closes)
 *
 * The shape of a "job" here is intentionally separate from the
 * optimization Job — different tables, different lifecycle, different
 * payload fields — so the two stores stay clean.
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import axios from 'axios'

const api = axios.create({ withCredentials: true })

export type QmlJobStatus =
  | 'queued'
  | 'loading'
  | 'training'
  | 'baselines_training'
  | 'complete'
  | 'error'

export type QmlQpuRunStatus =
  | 'queued'
  | 'submitting'
  | 'submitted'
  | 'running'
  | 'complete'
  | 'error'

export interface QmlQpuRunMetrics {
  test_accuracy: number
  confusion_matrix: number[][]
  predictions: number[]
  probabilities: number[]
  backend_name: string
  shots: number
  is_real_hardware: boolean
}

export interface QmlQpuRun {
  id: string
  qml_job_id: string
  user_id: number
  provider: 'ibmq' | 'originqc'
  backend_name: string | null
  shots: number
  status: QmlQpuRunStatus
  cloud_job_id: string | null
  queue_position: number | null
  live_status: string | null
  metrics: string | null
  error: string | null
  created_at: string
  completed_at: string | null
  wall_time_ms: number | null
  submission_context: string | null
}

export interface QmlEpochPoint {
  epoch: number
  loss: number
  train_accuracy: number
  test_accuracy: number
}

export interface QmlCircuitInfo {
  backend_name: string
  backend_kind: 'statevector' | 'shot_simulator' | 'qpu'
  is_real_hardware: boolean
  n_qubits: number
  n_layers: number
  n_trainable_params: number
  encoding: string
  entangler: string
  measurement: string
  shots: number | null
}

export interface QmlDecisionGrid {
  resolution: number
  x_min: number
  x_max: number
  y_min: number
  y_max: number
  /** Row-major: probabilities[row * resolution + col] is the
   *  predicted P(class=1) at (xs[col], ys[row]). */
  probabilities: number[]
}

export interface QmlScatterPoint {
  x: number
  y: number
  label: number
  split: 'train' | 'test'
}

export interface QmlBaseline {
  name: string
  title: string
  library: string
  version: string
  family: 'linear' | 'kernel' | 'ensemble' | 'neural'
  train_accuracy: number
  test_accuracy: number
  train_time_ms: number
  confusion_matrix: number[][]
  notes: string
}

export interface QmlMetrics {
  final_train_accuracy: number
  final_test_accuracy: number
  final_loss: number
  confusion_matrix: number[][]
  weights: number[][]
  bias: number
  n_qubits: number
  pca_applied: boolean
  classes: string[]
  feature_names: string[]
  notes: string[]
  train_time_ms: number
  circuit_info?: QmlCircuitInfo
  baselines?: QmlBaseline[]
  decision_grid?: QmlDecisionGrid | null
  scatter_points?: QmlScatterPoint[] | null
}

export interface QmlJob {
  id: string
  user_id: number
  dataset_id: string
  model: string
  status: QmlJobStatus
  hyperparameters?: string | null
  training_history?: string | null
  metrics?: string | null
  error?: string | null
  created_at: string
  completed_at?: string | null
  train_time_ms?: number | null
}

export interface QmlDataset {
  id: string
  title: string
  category: 'synthetic' | 'real'
  difficulty: 'easy' | 'medium' | 'hard'
  n_features: number
  n_classes: number
  n_samples: number
  summary: string
  source: string
}

export interface QmlCapabilities {
  pennylane: boolean
  sklearn: boolean
  qiskit_ibm_runtime: boolean
}

export interface QmlHyperparameters {
  n_qubits?: number
  n_layers?: number
  n_epochs?: number
  batch_size?: number
  learning_rate?: number
  seed?: number
}

const TERMINAL: QmlJobStatus[] = ['complete', 'error']

export const useQmlStore = defineStore('qml', () => {
  const datasets = ref<QmlDataset[]>([])
  const capabilities = ref<QmlCapabilities | null>(null)
  const currentJob = ref<QmlJob | null>(null)
  /** Live epoch points pushed by the SSE stream. Components render
   * this as a chart; on page reload it's hydrated from the DB row. */
  const liveHistory = ref<QmlEpochPoint[]>([])
  /** Most-recent decision-boundary grid (pushed every 3 epochs for
   *  2-qubit jobs). The frontend renders this as a heatmap behind the
   *  scatter plot so students watch the boundary curve form. */
  const liveDecisionGrid = ref<QmlDecisionGrid | null>(null)
  /** Latest boundary epoch — useful for the "boundary @ epoch N" caption. */
  const liveDecisionGridEpoch = ref<number | null>(null)
  /** QML-5: real-QPU inference runs against the currently-loaded job. */
  const qpuRuns = ref<QmlQpuRun[]>([])
  let qpuPollTimer: ReturnType<typeof setInterval> | null = null
  /** Hyperparameters echoed back from the trainer once it boots up
   * (with PCA + sample-count info that the user form didn't have). */
  const trainingMeta = ref<Record<string, any> | null>(null)
  const error = ref<string | null>(null)
  let eventSource: EventSource | null = null

  const isTraining = computed(() => {
    const s = currentJob.value?.status
    return s !== undefined && !TERMINAL.includes(s)
  })

  const finalMetrics = computed<QmlMetrics | null>(() => {
    if (!currentJob.value?.metrics) return null
    try {
      return JSON.parse(currentJob.value.metrics) as QmlMetrics
    } catch {
      return null
    }
  })

  function closeStream() {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
  }

  async function loadDatasets() {
    const r = await api.get('/api/qml/datasets')
    datasets.value = r.data.datasets
  }

  async function loadCapabilities() {
    const r = await api.get('/api/qml/health')
    capabilities.value = r.data.capabilities
  }

  async function loadJob(jobId: string) {
    const r = await api.get(`/api/qml/jobs/${jobId}`)
    currentJob.value = r.data
    // Hydrate live state from the persisted row so a page refresh
    // mid-training shows the same chart shape it had before.
    if (r.data.training_history) {
      try {
        liveHistory.value = JSON.parse(r.data.training_history)
      } catch {
        liveHistory.value = []
      }
    } else {
      liveHistory.value = []
    }
    // Reset the live grid on every load — if the job already finished,
    // finalMetrics.decision_grid is the source of truth.
    liveDecisionGrid.value = null
    liveDecisionGridEpoch.value = null
  }

  function subscribeStream(jobId: string) {
    closeStream()
    error.value = null
    eventSource = new EventSource(`/api/qml/jobs/${jobId}/stream`, {
      withCredentials: true,
    })
    eventSource.addEventListener('status', (raw) => {
      try {
        const event = JSON.parse((raw as MessageEvent).data)
        applyEvent(event)
      } catch (e) {
        console.warn('QML SSE parse error', e)
      }
    })
    eventSource.addEventListener('error', () => {
      // SSE auto-reconnects unless we close it. If the job is already
      // terminal, just close — there's nothing more to receive.
      if (currentJob.value && TERMINAL.includes(currentJob.value.status)) {
        closeStream()
      }
    })
  }

  function applyEvent(event: any) {
    const status = event.status as QmlJobStatus | 'epoch'

    if (status === 'epoch') {
      // Append to the live curve, dedupe on epoch number in case the
      // bus replays history.
      const point: QmlEpochPoint = {
        epoch: event.epoch,
        loss: event.loss,
        train_accuracy: event.train_accuracy,
        test_accuracy: event.test_accuracy,
      }
      const existing = liveHistory.value.find((p) => p.epoch === point.epoch)
      if (!existing) {
        liveHistory.value.push(point)
      }
      return
    }

    if (status === 'decision_grid') {
      liveDecisionGrid.value = event.grid as QmlDecisionGrid
      liveDecisionGridEpoch.value = event.epoch
      return
    }

    // status transitions — update the job row directly.
    if (currentJob.value) {
      currentJob.value.status = status
    }
    if (status === 'training') {
      trainingMeta.value = {
        n_qubits: event.n_qubits,
        n_layers: event.n_layers,
        n_epochs: event.n_epochs,
        n_samples_train: event.n_samples_train,
        n_samples_test: event.n_samples_test,
        pca_applied: event.pca_applied,
        notes: event.notes,
        feature_names: event.feature_names,
        classes: event.classes,
        circuit_info: event.circuit_info,
      }
    }
    if (status === 'complete' && currentJob.value) {
      // Stuff the full metrics blob into the job row so finalMetrics
      // can read it without a follow-up GET.
      currentJob.value.metrics = JSON.stringify({
        final_train_accuracy: event.final_train_accuracy,
        final_test_accuracy: event.final_test_accuracy,
        final_loss: event.final_loss,
        confusion_matrix: event.confusion_matrix,
        weights: event.weights,
        bias: event.bias,
        n_qubits: event.n_qubits,
        pca_applied: event.pca_applied,
        classes: event.classes,
        feature_names: event.feature_names,
        notes: event.notes,
        train_time_ms: event.train_time_ms,
        circuit_info: event.circuit_info,
        baselines: event.baselines,
        decision_grid: event.decision_grid,
        scatter_points: event.scatter_points,
      })
      currentJob.value.train_time_ms = event.train_time_ms
      closeStream()
    }
    if (status === 'error' && currentJob.value) {
      currentJob.value.error = event.message || 'Training failed.'
      closeStream()
    }
  }

  async function startTraining(
    datasetId: string,
    hyperparameters: QmlHyperparameters,
  ): Promise<QmlJob> {
    error.value = null
    liveHistory.value = []
    liveDecisionGrid.value = null
    liveDecisionGridEpoch.value = null
    trainingMeta.value = null
    const r = await api.post('/api/qml/train', {
      dataset_id: datasetId,
      model: 'vqc',
      hyperparameters,
    })
    const job = r.data.job as QmlJob
    currentJob.value = job
    subscribeStream(job.id)
    return job
  }

  function reset() {
    closeStream()
    stopQpuPolling()
    currentJob.value = null
    liveHistory.value = []
    liveDecisionGrid.value = null
    liveDecisionGridEpoch.value = null
    trainingMeta.value = null
    error.value = null
    qpuRuns.value = []
  }

  // ----- QML-5 real-QPU inference -----

  async function loadQpuRuns(jobId: string) {
    const r = await api.get(`/api/qml/jobs/${jobId}/qpu`)
    qpuRuns.value = r.data.qpu_runs
  }

  async function submitQpuRun(
    jobId: string,
    opts: {
      provider?: 'ibmq' | 'originqc'
      shots?: number
      backend_name?: string | null
      sample_index?: number | null
    },
  ): Promise<QmlQpuRun> {
    const provider = opts.provider ?? 'ibmq'
    const path = provider === 'originqc'
      ? `/api/qml/jobs/${jobId}/qpu/originqc`
      : `/api/qml/jobs/${jobId}/qpu/ibmq`
    const body: Record<string, any> = {
      shots: opts.shots ?? (provider === 'originqc' ? 2048 : 1024),
      backend_name: opts.backend_name ?? null,
    }
    if (provider === 'originqc' && opts.sample_index != null) {
      body.sample_index = opts.sample_index
    }
    const r = await api.post(path, body)
    const run = r.data.qpu_run as QmlQpuRun
    qpuRuns.value = [run, ...qpuRuns.value.filter((x) => x.id !== run.id)]
    return run
  }

  async function refreshQpuRun(runId: string): Promise<QmlQpuRun> {
    const r = await api.post(`/api/qml/qpu-runs/${runId}/refresh`)
    const updated = r.data.qpu_run as QmlQpuRun
    qpuRuns.value = qpuRuns.value.map((x) => (x.id === runId ? updated : x))
    return updated
  }

  /** Poll every 5 s while any tracked run is non-terminal. Stops on its
   *  own once everything settles. */
  function startQpuPolling() {
    stopQpuPolling()
    qpuPollTimer = setInterval(async () => {
      const pending = qpuRuns.value.filter(
        (r) => r.status !== 'complete' && r.status !== 'error',
      )
      if (pending.length === 0) {
        stopQpuPolling()
        return
      }
      await Promise.all(pending.map((r) => refreshQpuRun(r.id).catch(() => null)))
    }, 5000)
  }

  function stopQpuPolling() {
    if (qpuPollTimer) {
      clearInterval(qpuPollTimer)
      qpuPollTimer = null
    }
  }

  return {
    datasets,
    capabilities,
    currentJob,
    liveHistory,
    liveDecisionGrid,
    liveDecisionGridEpoch,
    trainingMeta,
    error,
    isTraining,
    finalMetrics,
    qpuRuns,
    loadDatasets,
    loadCapabilities,
    loadJob,
    subscribeStream,
    startTraining,
    closeStream,
    reset,
    loadQpuRuns,
    submitQpuRun,
    refreshQpuRun,
    startQpuPolling,
    stopQpuPolling,
  }
})
