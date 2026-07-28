<script setup lang="ts">
/**
 * QrcFeatureLabPanel — Phase-1 Feature-Lab view (Exp 1.1 / 1.2 / 1.3).
 *
 * Three things, from a phase1 run's cached trace + ``summary.json``:
 *  1. Feature-space embedding — 2-D PCA/UMAP scatter of the reservoir feature
 *     vectors (via ``/embedding``), coloured by target. The "UMAP visualization".
 *  2. Feature-method comparison (Exp 1.1/1.2) — magnitude653 vs phase vs
 *     multimodal, metric + feature count, as bars.
 *  3. Selection sweep (Exp 1.3) — metric vs #features, one curve per reducer
 *     (pca/kpca/umap/lasso/mi/random), the interactive twin of the static
 *     ``phase1_selection.png`` figure.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { useQrcStore, type QrcSummaryRow } from '@/stores/qrc'
import QrcEmbeddingScatter from './QrcEmbeddingScatter.vue'

const props = defineProps<{
  runId: string
  results: Record<string, any> | null
  task?: string
}>()

const qrc = useQrcStore()

const FEATURES = [
  { value: 'magnitude653', title: 'magnitude653' },
  { value: 'phase', title: 'phase' },
  { value: 'multimodal', title: 'multimodal' },
] as const
const feature = ref<'magnitude653' | 'phase' | 'multimodal'>('multimodal')
const method = ref<'pca' | 'umap'>('pca')
const embLoading = ref(false)
const embError = ref<string | null>(null)

async function computeEmbedding() {
  embLoading.value = true
  embError.value = null
  try {
    await qrc.loadEmbedding(props.runId, method.value, feature.value)
  } catch (e: any) {
    embError.value = e?.response?.data?.error || e?.message || 'Failed to compute embedding'
  } finally {
    embLoading.value = false
  }
}

// ---- Summary rows (from phase1 summary.json) ----
const rows = computed<QrcSummaryRow[]>(() => (props.results?.rows as QrcSummaryRow[]) ?? [])
const hasSweep = computed(() => rows.value.length > 0)
const metricName = computed<string>(() => (props.results?.metric_name as string) ?? 'metric')
const higherBetter = computed(() => !metricName.value.includes('nmse'))
const featureRows = computed(() => rows.value.filter((r) => r.kind === 'feature'))
const selectionRows = computed(() =>
  rows.value.filter((r) => r.kind === 'selection' && r.metric != null && r.feature_count != null),
)

function fmtMetric(v: number | null | undefined): string {
  if (v == null) return '—'
  return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toFixed(4)
}

// Method-comparison bar widths: normalize metric across the feature rows so a
// bar's length is comparable even when values are negative (tiny-sample R²).
const metricSpan = computed(() => {
  const vals = featureRows.value.map((r) => r.metric ?? 0)
  return { min: Math.min(...vals, 0), max: Math.max(...vals, 0) }
})
function barPct(r: QrcSummaryRow): number {
  const { min, max } = metricSpan.value
  const span = max - min || 1
  return 8 + (((r.metric ?? 0) - min) / span) * 92
}

// ---- Selection-sweep uPlot ----
const REDUCER_COLORS: Record<string, string> = {
  none: '#9aa7b4', pca: '#4da3ff', kpca: '#a78bfa', umap: '#f43f5e',
  lasso: '#f59e0b', mi: '#22c55e', random: '#94a3b8',
}
const selHost = ref<HTMLDivElement | null>(null)
let selPlot: uPlot | null = null
let selResize: ResizeObserver | null = null

function buildSel(): { data: uPlot.AlignedData; series: uPlot.Series[] } {
  const pts = selectionRows.value
  const reducers = [...new Set(pts.map((r) => r.reducer || 'none'))]
  const xset = [...new Set(pts.map((r) => r.feature_count as number))].sort((a, b) => a - b)
  const data: any[] = [Float64Array.from(xset)]
  const series: uPlot.Series[] = [{}]
  for (const red of reducers) {
    const col = xset.map((x) => {
      const m = pts.find((p) => (p.reducer || 'none') === red && p.feature_count === x)
      return m ? (m.metric as number) : null
    })
    data.push(col)
    series.push({
      label: red,
      stroke: REDUCER_COLORS[red] || '#ccc',
      width: 1.75,
      spanGaps: true,
      points: { show: true, size: 6 },
    })
  }
  return { data: data as uPlot.AlignedData, series }
}

function makeSelOpts(width: number, series: uPlot.Series[]): uPlot.Options {
  return {
    width,
    height: 300,
    padding: [10, 12, 0, 6],
    scales: { x: { time: false }, y: higherBetter.value ? {} : { distr: 3 } },
    axes: [
      { label: 'number of features', stroke: '#ccc', grid: { stroke: 'rgba(255,255,255,0.08)' } },
      { label: metricName.value, stroke: '#ccc', grid: { stroke: 'rgba(255,255,255,0.08)' } },
    ],
    series,
    legend: { show: true },
  }
}

function renderSel() {
  if (!selHost.value || !selectionRows.value.length) return
  const width = selHost.value.clientWidth || 640
  const { data, series } = buildSel()
  selPlot?.destroy()
  selPlot = new uPlot(makeSelOpts(width, series), data, selHost.value)
}

onMounted(() => {
  // Auto-compute the fast PCA embedding on open; UMAP stays on-demand.
  if (props.task === 'phase1' || props.task === 'weather' || props.task === 'narma' || props.task === 'trace-gen') {
    computeEmbedding()
  }
  renderSel()
  if (selHost.value) {
    selResize = new ResizeObserver(() => {
      if (selPlot && selHost.value) selPlot.setSize({ width: selHost.value.clientWidth || 640, height: 300 })
    })
    selResize.observe(selHost.value)
  }
})
watch(selectionRows, () => renderSel())
onBeforeUnmount(() => {
  selResize?.disconnect()
  selPlot?.destroy()
  selPlot = null
})
</script>

<template>
  <v-card class="pa-4 my-4">
    <div class="d-flex align-center mb-1 flex-wrap ga-2">
      <v-icon icon="mdi-scatter-plot" class="mr-1" />
      <div class="text-h6 flex-grow-1">Feature Lab</div>
      <span class="text-caption text-medium-emphasis">Phase-1 · Exp 1.1 / 1.2 / 1.3</span>
    </div>

    <!-- 1. Embedding -->
    <div class="text-subtitle-1 mt-2 mb-1">Feature-space embedding</div>
    <div class="d-flex align-center flex-wrap ga-3 mb-2">
      <v-btn-toggle v-model="feature" density="compact" variant="outlined" mandatory>
        <v-btn v-for="f in FEATURES" :key="f.value" :value="f.value" size="small">{{ f.title }}</v-btn>
      </v-btn-toggle>
      <v-btn-toggle v-model="method" density="compact" variant="outlined" mandatory>
        <v-btn value="pca" size="small">PCA</v-btn>
        <v-btn value="umap" size="small">UMAP</v-btn>
      </v-btn-toggle>
      <v-btn
        color="primary"
        size="small"
        variant="tonal"
        prepend-icon="mdi-play"
        :loading="embLoading"
        @click="computeEmbedding"
      >Compute</v-btn>
      <span v-if="method === 'umap'" class="text-caption text-medium-emphasis">
        UMAP is computed on demand — a few seconds.
      </span>
    </div>

    <v-alert
      v-if="qrc.usingMockEmbedding"
      type="warning" variant="tonal" density="compact" class="mb-2" icon="mdi-cloud-off-outline"
    >Backend unreachable — showing a synthetic embedding so the layout renders.</v-alert>
    <v-alert v-if="embError" type="error" variant="tonal" density="compact" class="mb-2">{{ embError }}</v-alert>

    <QrcEmbeddingScatter :embedding="qrc.currentEmbedding" />

    <v-divider class="my-4" />

    <!-- 2. Feature-method comparison -->
    <div class="text-subtitle-1 mb-1">Feature-method comparison
      <span class="text-caption text-medium-emphasis">(Exp 1.1 / 1.2)</span>
    </div>
    <div v-if="!featureRows.length" class="text-body-2 text-medium-emphasis mb-2">
      No sweep results yet — run a <code>phase1</code> task on this trace.
    </div>
    <div v-else class="mb-2">
      <div v-for="r in featureRows" :key="r.experiment" class="method-row">
        <div class="method-name text-body-2">{{ r.experiment.replace(/^1\.\d+_/, '') }}</div>
        <div class="method-bar-track">
          <div class="method-bar" :style="{ width: barPct(r) + '%' }"></div>
        </div>
        <div class="method-val text-caption">
          {{ fmtMetric(r.metric) }}
          <span class="text-medium-emphasis">· {{ r.feature_count }} feat</span>
        </div>
      </div>
    </div>

    <v-divider class="my-4" />

    <!-- 3. Selection sweep -->
    <div class="text-subtitle-1 mb-1">Selection sweep
      <span class="text-caption text-medium-emphasis">(Exp 1.3 · {{ metricName }} vs #features)</span>
    </div>
    <div v-if="!selectionRows.length" class="text-body-2 text-medium-emphasis">
      No selection sweep yet — run a <code>phase1</code> task on this trace.
    </div>
    <div v-show="selectionRows.length" ref="selHost" class="sel-host"></div>

    <div v-if="hasSweep" class="text-caption text-medium-emphasis mt-2">
      {{ higherBetter ? 'Higher is better (R²).' : 'Lower is better (NMSE, log axis).' }}
      Negative R² just means the train/test split is too small to fit — expected
      on a quick preview trace; the full run produces the real curves.
    </div>
  </v-card>
</template>

<style scoped>
.method-row {
  display: grid;
  grid-template-columns: 130px 1fr 150px;
  align-items: center;
  gap: 10px;
  padding: 3px 0;
}
.method-bar-track {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 4px;
  height: 14px;
  overflow: hidden;
}
.method-bar {
  height: 100%;
  background: linear-gradient(90deg, #4da3ff, #22c55e);
  border-radius: 4px;
}
.method-val {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.sel-host {
  width: 100%;
}
:deep(.u-legend) {
  font-size: 0.75rem;
}
</style>
