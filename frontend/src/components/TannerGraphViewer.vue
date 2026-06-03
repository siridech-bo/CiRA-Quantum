<script setup lang="ts">
/**
 * Interactive Tanner-graph viewer (Sprint 2).
 *
 * Renders the bipartite Tanner graph of a CSS code with positions
 * computed server-side via networkx. Data qubits are circles, check
 * operators are squares. Hovering a node dims non-incident edges +
 * non-neighbor nodes. The strategy selector triggers a refetch +
 * re-render; the Pauli toggle controls which subgraphs are shown.
 *
 * Server contract is documented in
 * [backend/app/qldpc/layouts.py](backend/app/qldpc/layouts.py).
 */
import { computed, onMounted, ref, watch } from 'vue'
import {
  useQldpcStore,
  type QldpcLayoutStrategy,
  type QldpcPositions,
  type QldpcTannerSubgraph,
  type QldpcRoutingMetrics,
} from '@/stores/qldpc'

interface Props {
  familyId: string
}

const props = defineProps<Props>()
const qldpc = useQldpcStore()

const loading = ref(false)
const error = ref<string | null>(null)

const strategy = ref<QldpcLayoutStrategy>('bipartite')
type PauliView = 'X' | 'Z' | 'BOTH'
const pauliView = ref<PauliView>('X')

const STRATEGY_OPTIONS: { value: QldpcLayoutStrategy; label: string; hint: string }[] = [
  { value: 'bipartite', label: 'Bipartite (data left, checks right)', hint: 'Cleanest view of Tanner structure' },
  { value: 'kamada_kawai', label: 'Kamada–Kawai (energy minimization)', hint: 'Reveals natural code geometry' },
  { value: 'spring', label: 'Spring (force-directed)', hint: 'Fruchterman-Reingold; deterministic seed' },
  { value: 'circular', label: 'Circular', hint: 'All nodes on a ring; useful for finite-rate codes' },
]

const PAULI_OPTIONS: { value: PauliView; label: string }[] = [
  { value: 'X', label: 'X-checks only' },
  { value: 'Z', label: 'Z-checks only' },
  { value: 'BOTH', label: 'Both overlaid' },
]

async function fetchLayout() {
  loading.value = true
  error.value = null
  try {
    await qldpc.loadTannerGraph(props.familyId, { strategy: strategy.value })
  } catch (e: any) {
    error.value =
      e?.response?.data?.error || e?.message || 'Failed to compute Tanner-graph layout'
  } finally {
    loading.value = false
  }
}

onMounted(fetchLayout)
watch(() => props.familyId, fetchLayout)
watch(strategy, fetchLayout)

// ---- Rendering ------------------------------------------------------------

const VIEW_W = 640
const VIEW_H = 420

const tanner = computed(() => qldpc.currentTannerGraph)

interface RenderedNode {
  id: string
  type: 'data' | 'check'
  pauli: 'X' | 'Z'
  x: number  // pixel coordinates
  y: number
}

interface RenderedEdge {
  source: string
  target: string
  pauli: 'X' | 'Z'
  x1: number; y1: number
  x2: number; y2: number
}

function scale(positions: QldpcPositions | undefined, key: string): [number, number] | null {
  if (!positions) return null
  const p = positions[key]
  if (!p) return null
  return [p[0] * VIEW_W, p[1] * VIEW_H]
}

function buildScene(
  subgraph: QldpcTannerSubgraph | undefined,
  positions: QldpcPositions | undefined,
  pauli: 'X' | 'Z',
): { nodes: RenderedNode[]; edges: RenderedEdge[] } {
  if (!subgraph || !positions) return { nodes: [], edges: [] }
  const nodes: RenderedNode[] = []
  const nodeIndex: Record<string, RenderedNode> = {}
  for (const n of subgraph.nodes) {
    const coord = scale(positions, n.id)
    if (!coord) continue
    const rn: RenderedNode = { id: n.id, type: n.type, pauli, x: coord[0], y: coord[1] }
    nodes.push(rn)
    nodeIndex[n.id] = rn
  }
  const edges: RenderedEdge[] = []
  for (const e of subgraph.edges) {
    const s = nodeIndex[e.source]
    const t = nodeIndex[e.target]
    if (!s || !t) continue
    edges.push({ source: e.source, target: e.target, pauli, x1: s.x, y1: s.y, x2: t.x, y2: t.y })
  }
  return { nodes, edges }
}

const renderedX = computed(() =>
  buildScene(tanner.value?.graph_x, tanner.value?.positions_x, 'X'),
)
const renderedZ = computed(() =>
  buildScene(tanner.value?.graph_z, tanner.value?.positions_z, 'Z'),
)

const visibleEdges = computed<RenderedEdge[]>(() => {
  const view = pauliView.value
  if (view === 'X') return renderedX.value.edges
  if (view === 'Z') return renderedZ.value.edges
  return [...renderedX.value.edges, ...renderedZ.value.edges]
})

const visibleNodes = computed<RenderedNode[]>(() => {
  const view = pauliView.value
  if (view === 'X') return renderedX.value.nodes
  if (view === 'Z') return renderedZ.value.nodes
  // BOTH: merge by id, keep X's coord (X is the primary subgraph the
  // bipartite layout was anchored on).
  const merged: Record<string, RenderedNode> = {}
  for (const n of renderedX.value.nodes) merged[n.id] = n
  for (const n of renderedZ.value.nodes) {
    if (!merged[n.id]) merged[n.id] = n
  }
  return Object.values(merged)
})

// Hover + pin state.
const hoveredNode = ref<string | null>(null)
const pinnedNode = ref<string | null>(null)
const focusNode = computed(() => pinnedNode.value || hoveredNode.value)

const neighborSet = computed<Set<string>>(() => {
  const target = focusNode.value
  if (!target) return new Set()
  const out = new Set<string>([target])
  for (const e of visibleEdges.value) {
    if (e.source === target) out.add(e.target)
    if (e.target === target) out.add(e.source)
  }
  return out
})

function isDimmed(id: string): boolean {
  if (!focusNode.value) return false
  return !neighborSet.value.has(id)
}

function edgeDimmed(e: RenderedEdge): boolean {
  if (!focusNode.value) return false
  return !(e.source === focusNode.value || e.target === focusNode.value)
}

function onNodeHover(id: string | null) {
  hoveredNode.value = id
}

function onNodeClick(id: string) {
  pinnedNode.value = pinnedNode.value === id ? null : id
}

function onBackgroundClick() {
  pinnedNode.value = null
}

// ---- Metrics --------------------------------------------------------------

interface MetricSummary {
  pauli: 'X' | 'Z'
  metrics: QldpcRoutingMetrics
}

const metricsSummaries = computed<MetricSummary[]>(() => {
  const out: MetricSummary[] = []
  if (tanner.value?.metrics_x && (pauliView.value === 'X' || pauliView.value === 'BOTH')) {
    out.push({ pauli: 'X', metrics: tanner.value.metrics_x })
  }
  if (tanner.value?.metrics_z && (pauliView.value === 'Z' || pauliView.value === 'BOTH')) {
    out.push({ pauli: 'Z', metrics: tanner.value.metrics_z })
  }
  return out
})

function fmt(v: number): string {
  return v.toFixed(3)
}

function pauliColor(p: 'X' | 'Z'): string {
  return p === 'X' ? '#42a5f5' : '#ef5350'
}
</script>

<template>
  <div class="tanner-viewer">
    <!-- Controls -->
    <div class="d-flex align-center flex-wrap ga-3 mb-3">
      <v-select
        v-model="strategy"
        :items="STRATEGY_OPTIONS"
        item-title="label"
        item-value="value"
        label="Layout strategy"
        density="compact"
        hide-details
        style="min-width: 280px; max-width: 360px"
        :disabled="loading"
      />
      <v-btn-toggle
        v-model="pauliView"
        mandatory
        density="compact"
        color="primary"
      >
        <v-btn
          v-for="opt in PAULI_OPTIONS"
          :key="opt.value"
          :value="opt.value"
          size="small"
        >
          {{ opt.label }}
        </v-btn>
      </v-btn-toggle>
      <v-spacer />
      <v-chip
        v-if="tanner?.strategy"
        size="small"
        color="info"
        variant="tonal"
      >
        active: {{ tanner.strategy }}
      </v-chip>
    </div>

    <!-- Status -->
    <v-progress-linear v-if="loading" indeterminate color="primary" class="mb-2" />
    <v-alert v-if="error" type="error" variant="tonal" density="compact" class="mb-2">
      {{ error }}
    </v-alert>

    <!-- SVG canvas -->
    <div class="canvas-wrap" :style="{ width: `${VIEW_W}px`, height: `${VIEW_H}px` }">
      <svg
        :width="VIEW_W"
        :height="VIEW_H"
        :viewBox="`0 0 ${VIEW_W} ${VIEW_H}`"
        xmlns="http://www.w3.org/2000/svg"
        @click="onBackgroundClick"
        role="img"
        aria-label="Tanner graph viewer"
      >
        <!-- Edges first so nodes paint over them. -->
        <line
          v-for="(e, i) in visibleEdges"
          :key="`e${i}`"
          :x1="e.x1"
          :y1="e.y1"
          :x2="e.x2"
          :y2="e.y2"
          :stroke="pauliColor(e.pauli)"
          :stroke-width="1"
          :opacity="edgeDimmed(e) ? 0.08 : 0.5"
        />
        <!-- Nodes -->
        <template v-for="n in visibleNodes" :key="`n${n.pauli}${n.id}`">
          <circle
            v-if="n.type === 'data'"
            :cx="n.x"
            :cy="n.y"
            :r="focusNode === n.id ? 7 : 4.5"
            :fill="focusNode === n.id ? '#ffd54f' : '#90caf9'"
            stroke="#1565c0"
            stroke-width="0.8"
            :opacity="isDimmed(n.id) ? 0.15 : 1"
            class="tanner-node"
            @mouseenter="onNodeHover(n.id)"
            @mouseleave="onNodeHover(null)"
            @click.stop="onNodeClick(n.id)"
          >
            <title>{{ n.id }} · data</title>
          </circle>
          <rect
            v-else
            :x="n.x - 4"
            :y="n.y - 4"
            :width="focusNode === n.id ? 10 : 8"
            :height="focusNode === n.id ? 10 : 8"
            :fill="focusNode === n.id ? '#ffd54f' : pauliColor(n.pauli)"
            stroke="#0d47a1"
            stroke-width="0.6"
            :opacity="isDimmed(n.id) ? 0.15 : 1"
            class="tanner-node"
            @mouseenter="onNodeHover(n.id)"
            @mouseleave="onNodeHover(null)"
            @click.stop="onNodeClick(n.id)"
          >
            <title>{{ n.id }} · {{ n.pauli }}-check</title>
          </rect>
        </template>
      </svg>
    </div>

    <!-- Focused-node banner -->
    <div v-if="focusNode" class="d-flex align-center mt-2 mb-1 ga-2 flex-wrap">
      <v-chip size="x-small" color="amber-darken-3" variant="tonal">
        Focused: <strong class="ml-1">{{ focusNode }}</strong>
      </v-chip>
      <span class="text-caption text-medium-emphasis">
        Neighbors highlighted; non-incident edges dimmed.
        Click again or click background to clear.
      </span>
    </div>

    <!-- Routing metrics -->
    <v-row dense class="mt-2">
      <v-col
        v-for="s in metricsSummaries"
        :key="s.pauli"
        cols="12"
        :md="metricsSummaries.length > 1 ? 6 : 12"
      >
        <v-card variant="outlined" class="pa-3">
          <div class="d-flex align-center mb-1 ga-2">
            <v-chip
              size="x-small"
              variant="flat"
              :color="s.pauli === 'X' ? 'blue' : 'red'"
            >
              {{ s.pauli }}-subgraph
            </v-chip>
            <span class="text-caption text-medium-emphasis">
              {{ s.metrics.num_nodes }} nodes · {{ s.metrics.num_edges }} edges
            </span>
          </div>
          <div class="d-flex flex-wrap ga-2">
            <v-chip size="x-small" variant="tonal">
              avg edge {{ fmt(s.metrics.avg_edge_length) }}
            </v-chip>
            <v-chip size="x-small" variant="tonal">
              p95 {{ fmt(s.metrics.p95_edge_length) }}
            </v-chip>
            <v-chip
              size="x-small"
              variant="tonal"
              :color="s.metrics.max_edge_length > 0.5 ? 'warning' : undefined"
            >
              max {{ fmt(s.metrics.max_edge_length) }}
            </v-chip>
            <v-chip
              size="x-small"
              variant="tonal"
              :color="s.metrics.edge_crossings > 200 ? 'warning' : undefined"
            >
              crossings {{ s.metrics.edge_crossings.toLocaleString() }}
            </v-chip>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <div v-if="tanner" class="text-caption text-medium-emphasis mt-2">
      Distances in normalized [0, 1]² space. Lower is better — shorter
      max edge = shorter physical wire; fewer crossings = less routing
      collision risk.
    </div>
  </div>
</template>

<style scoped>
.tanner-viewer {
  font-family: inherit;
}
.canvas-wrap {
  background: rgba(0, 0, 0, 0.25);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 4px;
  overflow: hidden;
  max-width: 100%;
}
.canvas-wrap svg {
  display: block;
}
.tanner-node {
  cursor: pointer;
  transition: opacity 0.1s ease, r 0.1s ease;
}
.tanner-node:hover {
  filter: brightness(1.15);
}
</style>
