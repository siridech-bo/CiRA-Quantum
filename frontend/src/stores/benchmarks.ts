/**
 * Benchmark dashboard store.
 *
 * Reads from the public, read-only `/api/benchmarks/*` endpoints. No
 * auth required — the dashboard's whole point is that it's a public
 * scoreboard. We still use `withCredentials: true` so users who happen
 * to be logged in see consistent behavior with the rest of the app,
 * but the routes work for anonymous visitors as well.
 *
 * Cached state shapes:
 *   suites             — listing for the dashboard landing page
 *   suiteDetail        — keyed by suite_id, fetched lazily on view
 *   solverDetail       — keyed by solver_name
 *   instanceDetail     — keyed by instance_id
 *   recordDetail       — keyed by record_id
 *
 * We cache everything because the archive is append-only: an already-
 * fetched record never changes. The cache survives navigation, only
 * busted by an explicit `refresh()`.
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

const api = axios.create({ withCredentials: true })

export interface SuiteSummary {
  suite_id: string
  num_records: number
  solvers_seen: string[]
}

export interface InstanceMeta {
  instance_id: string
  problem_class: string
  expected_optimum: number | null
  expected_optimum_kind: string
  tags: string[]
}

export interface RecordSummary {
  record_id: string
  solver_name: string
  solver_version: string
  instance_id: string
  hardware_id: string
  started_at: string
  best_user_energy: number | null
  best_energy: number | null
  num_feasible: number | null
  num_samples: number | null
  elapsed_ms: number | null
  converged_to_expected: boolean | null
  gap_to_expected: number | null
  expected_optimum: number | null
}

export interface SuiteDetail {
  suite_id: string
  instances: InstanceMeta[]
  records: RecordSummary[]
}

export interface SolverIdentity {
  name: string
  version?: string
  source?: string
  hardware?: string | null
  parameter_schema?: Record<string, any>
}

export interface SolverDetail {
  solver: SolverIdentity
  is_currently_registered: boolean
  records_by_suite: Record<string, RecordSummary[]>
}

export interface InstanceDetail {
  instance_id: string
  leaderboard: RecordSummary[]
}

export interface FullRunRecord {
  record_id: string
  code_version: string
  solver: SolverIdentity
  parameters: Record<string, any>
  instance_id: string
  hardware_id: string
  started_at: string
  completed_at: string
  repro_hash: string
  results: Record<string, any>
  sample_set_path: string | null
}

export const useBenchmarksStore = defineStore('benchmarks', () => {
  const suites = ref<SuiteSummary[]>([])
  const suiteDetail = ref<Record<string, SuiteDetail>>({})
  const solverDetail = ref<Record<string, SolverDetail>>({})
  const instanceDetail = ref<Record<string, InstanceDetail>>({})
  const recordDetail = ref<Record<string, FullRunRecord>>({})
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function loadSuites(force = false): Promise<void> {
    if (!force && suites.value.length) return
    loading.value = true
    error.value = null
    try {
      const r = await api.get<{ suites: SuiteSummary[] }>('/api/benchmarks/suites')
      suites.value = r.data.suites
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || 'Failed to load suites'
    } finally {
      loading.value = false
    }
  }

  async function loadSuite(suiteId: string, force = false): Promise<SuiteDetail | null> {
    if (!force && suiteDetail.value[suiteId]) return suiteDetail.value[suiteId]
    loading.value = true
    error.value = null
    try {
      const r = await api.get<SuiteDetail>(`/api/benchmarks/suites/${suiteId}`)
      suiteDetail.value = { ...suiteDetail.value, [suiteId]: r.data }
      return r.data
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || `Failed to load suite ${suiteId}`
      return null
    } finally {
      loading.value = false
    }
  }

  async function loadSolver(solverName: string, force = false): Promise<SolverDetail | null> {
    if (!force && solverDetail.value[solverName]) return solverDetail.value[solverName]
    loading.value = true
    error.value = null
    try {
      const r = await api.get<SolverDetail>(`/api/benchmarks/solvers/${solverName}`)
      solverDetail.value = { ...solverDetail.value, [solverName]: r.data }
      return r.data
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || `Failed to load solver ${solverName}`
      return null
    } finally {
      loading.value = false
    }
  }

  async function loadInstance(instanceId: string, force = false): Promise<InstanceDetail | null> {
    if (!force && instanceDetail.value[instanceId]) return instanceDetail.value[instanceId]
    loading.value = true
    error.value = null
    try {
      const r = await api.get<InstanceDetail>(`/api/benchmarks/instances/${instanceId}`)
      instanceDetail.value = { ...instanceDetail.value, [instanceId]: r.data }
      return r.data
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || `Failed to load instance ${instanceId}`
      return null
    } finally {
      loading.value = false
    }
  }

  async function loadRecord(recordId: string, force = false): Promise<FullRunRecord | null> {
    if (!force && recordDetail.value[recordId]) return recordDetail.value[recordId]
    loading.value = true
    error.value = null
    try {
      const r = await api.get<{ record: FullRunRecord }>(`/api/benchmarks/records/${recordId}`)
      recordDetail.value = { ...recordDetail.value, [recordId]: r.data.record }
      return r.data.record
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || `Failed to load record ${recordId}`
      return null
    } finally {
      loading.value = false
    }
  }

  async function fetchCitation(recordId: string, kind: 'bibtex' | 'string' = 'bibtex'): Promise<string | null> {
    try {
      const r = await api.get<{ citation: string }>(
        `/api/benchmarks/records/${recordId}/cite`,
        { params: { kind } },
      )
      return r.data.citation
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || 'Failed to load citation'
      return null
    }
  }

  function refresh(): void {
    suites.value = []
    suiteDetail.value = {}
    solverDetail.value = {}
    instanceDetail.value = {}
    recordDetail.value = {}
  }

  return {
    // state
    suites,
    suiteDetail,
    solverDetail,
    instanceDetail,
    recordDetail,
    loading,
    error,
    // actions
    loadSuites,
    loadSuite,
    loadSolver,
    loadInstance,
    loadRecord,
    fetchCitation,
    refresh,
  }
})
