/**
 * qLDPC code-family store — Pinia.
 *
 * Sprint 0 is read-only: just loads the code-family registry and the
 * capability flags. Sprint 1 will add matrix generation + Tanner graph
 * loading; Sprint 3 will add async benchmark jobs + SSE streaming;
 * Sprint 4 will add QPU run submission. The store shape is intentionally
 * kept minimal here so adding those slices is purely additive.
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

const api = axios.create({ withCredentials: true })

export type QldpcCategory = 'css_classical' | 'css_product' | 'topological'
export type QldpcRegime = 'zero-rate' | 'finite-rate'

export interface QldpcCodeFamily {
  id: string
  title: string
  category: QldpcCategory
  regime: QldpcRegime
  /** Number of physical qubits in the representative instance. */
  n: number
  /** Number of logical qubits encoded. */
  k: number
  /** Code distance — minimum-weight non-trivial Pauli on a logical operator. */
  d: number | null
  /** ``"upper_bound"`` (Monte Carlo) or ``"exact"``. Only set when live. */
  d_mode?: 'upper_bound' | 'exact'
  summary: string
  discovered_by: string
  key_property: string
  use_case: string
  /** Best-known fault-tolerance threshold under standard depolarizing noise. */
  best_known_threshold_pct: number
  /** Sprint 1: True when ``n, k, d`` came from the live ``qldpc`` lib. */
  live?: boolean
  /** Sprint 1: construction parameter hints surfaced to the detail page. */
  params?: Record<string, unknown>
}

export interface QldpcCapabilities {
  /** ``qldpc`` PyPI package (Sprint 1 — matrix generation). */
  qldpc_lib: boolean
  /** ``stim`` C++ stabilizer simulator (Sprint 3 — Monte Carlo). */
  stim: boolean
  /** ``qiskit-qec`` (Sprint 4 — syndrome extraction circuits). */
  qiskit_qec: boolean
  /** ``networkx`` (Sprint 2 — Tanner graph layout). */
  networkx: boolean
}

// ---- Sprint 1: matrix / CSS check / distance / Tanner-graph payloads ----

export interface QldpcMatrix {
  matrix_x: number[][]
  matrix_z: number[][]
  n: number
  k: number
  num_checks_x: number
  num_checks_z: number
  nonzeros_x: number
  nonzeros_z: number
}

export interface QldpcCssCheck {
  commutes: boolean
  residual_nonzero_count: number
  product_shape: number[]
}

export interface QldpcMatrixPayload {
  family_id: string
  matrix: QldpcMatrix
  css_check: QldpcCssCheck
}

export interface QldpcDistance {
  family_id: string
  distance: number | null
  mode: 'upper_bound' | 'exact'
  time_ms: number
}

export interface QldpcTannerNode {
  id: string
  type: 'data' | 'check'
}

export interface QldpcTannerEdge {
  source: string
  target: string
}

export interface QldpcTannerSubgraph {
  nodes: QldpcTannerNode[]
  edges: QldpcTannerEdge[]
}

export type QldpcLayoutStrategy = 'bipartite' | 'kamada_kawai' | 'spring' | 'circular'

export interface QldpcRoutingMetrics {
  num_nodes: number
  num_edges: number
  avg_edge_length: number
  min_edge_length: number
  max_edge_length: number
  p95_edge_length: number
  edge_crossings: number
}

/** Map of stringified node id → [x, y] normalized to [0, 1]². */
export type QldpcPositions = Record<string, [number, number]>

export interface QldpcTannerGraph {
  family_id: string
  graph_x: QldpcTannerSubgraph
  graph_z: QldpcTannerSubgraph
  node_count_x: number
  node_count_z: number
  /** Available layout strategies the backend can compute. */
  available_strategies: QldpcLayoutStrategy[]
  /** Sprint 2 — only present when ``strategy`` query param was passed. */
  strategy?: QldpcLayoutStrategy
  positions_x?: QldpcPositions
  positions_z?: QldpcPositions
  metrics_x?: QldpcRoutingMetrics
  metrics_z?: QldpcRoutingMetrics
}

export const useQldpcStore = defineStore('qldpc', () => {
  const codeFamilies = ref<QldpcCodeFamily[]>([])
  const currentCodeFamily = ref<QldpcCodeFamily | null>(null)
  const capabilities = ref<QldpcCapabilities | null>(null)
  const error = ref<string | null>(null)

  // Sprint 1 — per-detail-page payloads. All keyed by family_id so
  // navigating between families clears stale state via load actions.
  const currentMatrix = ref<QldpcMatrixPayload | null>(null)
  const currentDistance = ref<QldpcDistance | null>(null)
  const currentTannerGraph = ref<QldpcTannerGraph | null>(null)

  async function loadCapabilities() {
    const r = await api.get('/api/qldpc/health')
    capabilities.value = r.data.capabilities
  }

  async function loadCodeFamilies() {
    const r = await api.get('/api/qldpc/code-families')
    codeFamilies.value = r.data.code_families
  }

  async function loadCodeFamily(familyId: string) {
    const r = await api.get(`/api/qldpc/code-families/${familyId}`)
    currentCodeFamily.value = r.data
    // Clear stale matrix/distance/Tanner state when switching families.
    if (currentMatrix.value?.family_id !== familyId) currentMatrix.value = null
    if (currentDistance.value?.family_id !== familyId) currentDistance.value = null
    if (currentTannerGraph.value?.family_id !== familyId) currentTannerGraph.value = null
  }

  async function loadMatrix(familyId: string): Promise<QldpcMatrixPayload> {
    const r = await api.get(`/api/qldpc/code-families/${familyId}/matrix`)
    currentMatrix.value = r.data
    return r.data
  }

  async function loadDistance(
    familyId: string,
    opts: { exact?: boolean } = {},
  ): Promise<QldpcDistance> {
    const params = opts.exact ? { exact: 'true' } : {}
    const r = await api.get(`/api/qldpc/code-families/${familyId}/distance`, { params })
    currentDistance.value = r.data
    return r.data
  }

  async function loadTannerGraph(
    familyId: string,
    opts: { strategy?: QldpcLayoutStrategy } = {},
  ): Promise<QldpcTannerGraph> {
    const params = opts.strategy ? { strategy: opts.strategy } : {}
    const r = await api.get(`/api/qldpc/code-families/${familyId}/tanner-graph`, { params })
    currentTannerGraph.value = r.data
    return r.data
  }

  function clearDetailState() {
    currentMatrix.value = null
    currentDistance.value = null
    currentTannerGraph.value = null
  }

  return {
    codeFamilies,
    currentCodeFamily,
    capabilities,
    error,
    currentMatrix,
    currentDistance,
    currentTannerGraph,
    loadCapabilities,
    loadCodeFamilies,
    loadCodeFamily,
    loadMatrix,
    loadDistance,
    loadTannerGraph,
    clearDetailState,
  }
})
