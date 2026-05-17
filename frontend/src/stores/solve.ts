/**
 * Solve store — drives the entire solve UI.
 *
 * State the components read:
 *   currentJob   — the live solve in progress, or the most recently
 *                  inspected past job
 *   history      — paginated list of the user's jobs
 *   keys         — list of BYOK provider entries (no values)
 *   error        — last user-facing error (cleared on next submit)
 *
 * Methods the components call:
 *   submitProblem(...)  — POST /api/solve and open SSE
 *   subscribeToJob(id)  — open SSE on an existing job (e.g. on detail page)
 *   loadJob(id)         — fetch one job's full detail
 *   loadHistory(page?)  — paginated list
 *   deleteJob(id)
 *   loadKeys() / putKey(provider, key) / deleteKey(provider)
 *
 * Why SSE is wired in the store (not in components):
 *   The store owns the EventSource lifetime — switching pages,
 *   submitting a new solve, or logging out cleanly tears it down. A
 *   component that owned its own EventSource would leak when unmounted
 *   in the middle of a stream.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

const api = axios.create({ withCredentials: true })

export type JobStatus =
  | 'queued'
  | 'formulating'
  | 'compiling'
  | 'validating'
  | 'solving'
  | 'complete'
  | 'error'

export interface Job {
  id: string
  user_id: number
  problem_statement: string
  provider: 'claude' | 'openai' | 'local'
  status: JobStatus
  cqm_json?: Record<string, any> | null
  variable_registry?: Record<string, string> | null
  validation_report?: Record<string, any> | null
  solution?: Record<string, number> | null
  interpreted_solution?: string | null
  error?: string | null
  num_variables?: number | null
  num_constraints?: number | null
  solve_time_ms?: number | null
  template_id?: string | null
  expected_optimum?: number | null
  solvers_requested?: string[] | null
  solver_results?: {
    solvers: Record<string, SolverResult>
    primary: string | null
    sense?: 'minimize' | 'maximize'
    /** True while the orchestrator is still iterating through solvers
     * — incremental snapshots get this flag set; the final write at
     * stage-4 completion does not. */
    in_progress?: boolean
  } | null
  created_at: string
  completed_at?: string | null
}

export interface SolverResult {
  status: 'pending' | 'running' | 'queued' | 'complete' | 'error'
  energy?: number
  raw_energy?: number
  feasible?: boolean
  elapsed_ms?: number
  tier_source?: string
  version?: string
  hardware?: string | null
  error?: string
  qaoa_extras?: QaoaExtras
  /** Set on rows that submitted asynchronously to a real-QPU cloud
   * (Phase 11). Polling materializes the row into ``status='complete'``
   * when the cloud job finishes. */
  cloud_job_id?: string
  backend_name?: string
}

export interface QaoaExtras {
  layer: number | null
  trained_gammas: (number | null)[]
  trained_betas: (number | null)[]
  num_qubits: number
  /** Number of logical variables in the original CQM. The difference
   * (num_qubits − num_logical_vars) is the count of slack qubits dimod
   * added when lowering ≤/≥ inequality constraints. Null on legacy
   * runs that pre-date the slack-qubit annotation. */
  num_logical_vars: number | null
  /** Per-qubit linear bias [(qubit_index, coefficient), ...]. Drives
   * the RZ gates in the gate-level circuit view. */
  linear_terms: [number, number][]
  /** Pairwise coupling [(qubit_i, qubit_j, coefficient), ...]. Drives
   * the CNOT–RZ–CNOT decompositions in the gate-level view. */
  quadratic_terms: [number, number, number][]
  top_bitstrings: string[]
  top_probabilities: (number | null)[]
  top_energies: (number | null)[]
  train_loss: number | null
  train_optimizer: string | null
  backend_name: string | null
  is_real_hardware: boolean
  job_id: string | null
}

export interface SolverInfo {
  name: string
  version: string
  source: string
  hardware: string | null
  tier: string
  tier_label: string
  tier_color: string
  recommended_default: boolean
  warning: string | null
  requires_key: string | null
}

export interface StoredKey {
  provider: string
  created_at: string
}

const TERMINAL_STATUSES: JobStatus[] = ['complete', 'error']

function hasQueuedRows(job: Job | null): boolean {
  if (!job?.solver_results?.solvers) return false
  return Object.values(job.solver_results.solvers).some(
    (r) => r.status === 'queued',
  )
}

export const useSolveStore = defineStore('solve', () => {
  const currentJob = ref<Job | null>(null)
  const history = ref<Job[]>([])
  const historyTotal = ref(0)
  const historyPage = ref(1)
  const keys = ref<StoredKey[]>([])
  const solvers = ref<SolverInfo[]>([])
  const error = ref<string | null>(null)
  let eventSource: EventSource | null = null

  const isRunning = computed(() => {
    const s = currentJob.value?.status
    return s !== undefined && !TERMINAL_STATUSES.includes(s)
  })

  function closeStream() {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
  }

  function streamStatus(jobId: string) {
    closeStream()
    eventSource = new EventSource(`/api/jobs/${jobId}/stream`, {
      withCredentials: true,
    })
    eventSource.addEventListener('status', (raw) => {
      try {
        const data = JSON.parse((raw as MessageEvent).data)
        if (currentJob.value?.id !== jobId) return
        // Per-solver progress: the orchestrator emits this every time a
        // solver in the fan-out starts, completes, or errors. The status
        // field is "solver_progress" (not a JobStatus); we re-fetch the
        // full job so the live solver_results dict is in sync.
        if (data.status === 'solver_progress') {
          void loadJob(jobId)
          return
        }
        // Patch the live job with whatever the event carried.
        currentJob.value = {
          ...currentJob.value,
          status: data.status as JobStatus,
          ...(data.num_variables !== undefined && { num_variables: data.num_variables }),
          ...(data.num_constraints !== undefined && { num_constraints: data.num_constraints }),
          ...(data.solve_time_ms !== undefined && { solve_time_ms: data.solve_time_ms }),
          ...(data.error !== undefined && { error: data.error }),
        }
        if (TERMINAL_STATUSES.includes(data.status)) {
          // Refresh the full job detail so result/cqm/validation tabs populate.
          void loadJob(jobId).then(() => {
            // Phase 11: keep the SSE alive while any queued cloud rows
            // are still waiting on async materialization. The poller
            // will emit solver_progress when each one resolves.
            if (!hasQueuedRows(currentJob.value)) {
              closeStream()
            }
          })
        }
      } catch (e) {
        // Malformed SSE chunk — ignore (the next event will arrive cleanly).
        console.warn('failed to parse SSE event', e)
      }
    })
    eventSource.onerror = () => {
      // Browser auto-reconnects; if the stream closed because the job
      // ended, the addEventListener('status') already handled cleanup.
      // Otherwise we leave the stream alive for retry.
    }
  }

  async function submitProblem(payload: {
    problem_statement: string
    provider: 'claude' | 'openai' | 'local'
    api_key?: string
    use_stored_key?: boolean
    solvers?: string[]
  }): Promise<Job> {
    error.value = null
    const r = await api.post<{ success: boolean; job: Job }>('/api/solve', payload)
    currentJob.value = r.data.job
    streamStatus(r.data.job.id)
    return r.data.job
  }

  async function loadSolvers(): Promise<SolverInfo[]> {
    if (solvers.value.length) return solvers.value
    const r = await api.get<{ solvers: SolverInfo[] }>('/api/solvers')
    solvers.value = r.data.solvers
    return solvers.value
  }

  async function subscribeToJob(jobId: string): Promise<void> {
    // Used by JobDetailPage for a job that may still be in-flight when
    // the user navigates to its URL directly.
    await loadJob(jobId)
    if (currentJob.value && !TERMINAL_STATUSES.includes(currentJob.value.status)) {
      streamStatus(jobId)
    } else if (hasQueuedRows(currentJob.value)) {
      // Phase 11: job is "complete" but real-QPU rows are still queued
      // in cloud. Attach the SSE so we see the materialize events when
      // they fire.
      streamStatus(jobId)
    }
  }

  async function loadJob(jobId: string): Promise<Job> {
    const r = await api.get<{ job: Job }>(`/api/jobs/${jobId}`)
    currentJob.value = r.data.job
    return r.data.job
  }

  async function loadHistory(page = 1, pageSize = 20): Promise<void> {
    const r = await api.get<{ jobs: Job[]; page: number; total: number; page_size: number }>(
      `/api/jobs?page=${page}&page_size=${pageSize}`,
    )
    history.value = r.data.jobs
    historyTotal.value = r.data.total
    historyPage.value = r.data.page
  }

  async function deleteJob(jobId: string): Promise<void> {
    await api.delete(`/api/jobs/${jobId}`)
    // Local-only optimistic update so the table doesn't blink.
    history.value = history.value.filter((j) => j.id !== jobId)
    historyTotal.value = Math.max(0, historyTotal.value - 1)
    if (currentJob.value?.id === jobId) {
      currentJob.value = null
    }
  }

  async function loadKeys(): Promise<void> {
    const r = await api.get<{ keys: StoredKey[] }>('/api/keys')
    keys.value = r.data.keys
  }

  async function putKey(provider: string, key: string): Promise<void> {
    await api.put(`/api/keys/${provider}`, { key })
    await loadKeys()
  }

  async function deleteKey(provider: string): Promise<void> {
    await api.delete(`/api/keys/${provider}`)
    await loadKeys()
  }

  /** Phase 11 — kick the server-side pending-jobs poller, then reload
   * the current job. Hitting ``/api/benchmarks/cloud-jobs/pending``
   * triggers materialization as a side-effect for any cloud jobs that
   * have reached terminal state, including any queued rows on the
   * currently-displayed solve job. Returns the number of pending
   * entries remaining (useful for the auto-poll loop to decide
   * whether to stop). */
  async function refreshCloudPoll(): Promise<number> {
    try {
      const r = await api.get<{ pending: unknown[] }>(
        '/api/benchmarks/cloud-jobs/pending',
      )
      const pendingCount = (r.data.pending ?? []).length
      if (currentJob.value) {
        await loadJob(currentJob.value.id)
      }
      return pendingCount
    } catch {
      // Network error / 503 — leave the row unchanged. The next poll
      // will retry. We don't surface this in the panel.
      return -1
    }
  }

  async function testKey(provider: string): Promise<{
    ok: boolean
    message?: string
    error?: string
    elapsed_ms?: number
  }> {
    try {
      const r = await api.post<{
        ok: boolean
        message?: string
        error?: string
        elapsed_ms?: number
      }>(`/api/keys/${provider}/test`)
      return r.data
    } catch (e: any) {
      return {
        ok: false,
        error: e?.response?.data?.error || e?.message || 'Test request failed',
      }
    }
  }

  function reset(): void {
    closeStream()
    currentJob.value = null
    history.value = []
    keys.value = []
    solvers.value = []
    error.value = null
  }

  return {
    // state
    currentJob,
    history,
    historyTotal,
    historyPage,
    keys,
    solvers,
    error,
    // computed
    isRunning,
    // actions
    submitProblem,
    subscribeToJob,
    loadJob,
    loadHistory,
    deleteJob,
    loadKeys,
    putKey,
    deleteKey,
    testKey,
    loadSolvers,
    refreshCloudPoll,
    closeStream,
    reset,
  }
})
