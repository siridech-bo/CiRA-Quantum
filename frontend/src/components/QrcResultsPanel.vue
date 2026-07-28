<script setup lang="ts">
/**
 * QrcResultsPanel — renders the NARMA / weather-vs-ESN results tables
 * from ``GET /runs/<id>/results``.
 *
 * §3 only pins this endpoint's shape as "the results JSON" (it points at
 * §2's ``evaluate_features`` protocol dict for the field names actually
 * produced). We render the known fields (weather_r2, narma10_nmse,
 * delta_vs_baseline, esn_comparison, feature_count, wall_clock_s)
 * defensively — each block only appears if present — and fall back to a
 * raw JSON dump for anything else so no data is silently dropped if
 * Coder C's backend emits extra fields.
 */
import { computed } from 'vue'
import type { QrcResults } from '@/stores/qrc'

const props = defineProps<{
  results: QrcResults | null
}>()

const HORIZON_ORDER = ['h1', 'h10', 'h20', 'h30', 'h45']

const weatherHorizons = computed(() => {
  const r2 = props.results?.weather_r2
  if (!r2) return []
  const keys = Object.keys(r2)
  keys.sort((a, b) => {
    const ia = HORIZON_ORDER.indexOf(a)
    const ib = HORIZON_ORDER.indexOf(b)
    if (ia === -1 && ib === -1) return a.localeCompare(b)
    if (ia === -1) return 1
    if (ib === -1) return -1
    return ia - ib
  })
  return keys
})

const esn = computed(() => props.results?.esn_comparison || null)

const knownKeys = [
  'experiment', 'feature_count', 'weather_r2', 'narma10_nmse',
  'wall_clock_s', 'delta_vs_baseline', 'esn_comparison',
]
const extraFields = computed(() => {
  if (!props.results) return null
  const extra = Object.fromEntries(
    Object.entries(props.results).filter(([k]) => !knownKeys.includes(k)),
  )
  return Object.keys(extra).length ? extra : null
})

function fmtR2(v: number | undefined): string {
  return v === undefined ? '—' : v.toFixed(3)
}
</script>

<template>
  <v-card v-if="results" variant="outlined" class="pa-4">
    <div class="d-flex align-center flex-wrap ga-2 mb-3">
      <v-icon icon="mdi-table-large" class="mr-1" />
      <div class="text-subtitle-1 flex-grow-1">Results</div>
      <v-chip v-if="results.experiment" size="x-small" variant="tonal">
        {{ results.experiment }}
      </v-chip>
      <v-chip v-if="results.feature_count != null" size="x-small" variant="tonal">
        {{ results.feature_count }} features
      </v-chip>
      <v-chip v-if="results.wall_clock_s != null" size="x-small" variant="tonal">
        {{ (results.wall_clock_s / 60).toFixed(1) }} min
      </v-chip>
    </div>

    <!-- Weather R² (+ ESN comparison if present) -->
    <template v-if="weatherHorizons.length">
      <div class="text-subtitle-2 mb-2">Weather temperature R² by horizon</div>
      <div class="table-scroll mb-4">
        <table class="qrc-table">
          <thead>
            <tr>
              <th>horizon</th>
              <th>QRC</th>
              <th v-if="esn">QRC (this run)</th>
              <th v-if="esn">QRC+RBF</th>
              <th v-if="esn">best ESN</th>
              <th v-if="results?.delta_vs_baseline">Δ vs baseline</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="h in weatherHorizons" :key="h">
              <th>{{ h.replace('h', '') }}</th>
              <td class="num">{{ fmtR2(results?.weather_r2?.[h]) }}</td>
              <template v-if="esn">
                <td class="num">{{ fmtR2(esn[h]?.qrc) }}</td>
                <td class="num">{{ fmtR2(esn[h]?.qrc_rbf) }}</td>
                <td class="num">{{ fmtR2(esn[h]?.best_esn) }}</td>
              </template>
              <td v-if="results?.delta_vs_baseline" class="num">
                {{ results.delta_vs_baseline[h] ?? '—' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <!-- NARMA-10 NMSE -->
    <div v-if="results.narma10_nmse != null" class="mb-4">
      <div class="text-subtitle-2 mb-1">NARMA-10 NMSE</div>
      <div class="text-h6">{{ results.narma10_nmse.toExponential(2) }}</div>
    </div>

    <!-- Unrecognized fields, shown transparently rather than dropped -->
    <template v-if="extraFields">
      <v-divider class="my-3" />
      <div class="text-caption text-medium-emphasis mb-1">
        Additional fields returned by the backend
      </div>
      <pre class="extra-json">{{ JSON.stringify(extraFields, null, 2) }}</pre>
    </template>

    <div
      v-if="!weatherHorizons.length && results.narma10_nmse == null && !extraFields"
      class="text-body-2 text-medium-emphasis"
    >
      No results yet.
    </div>
  </v-card>
  <v-card v-else variant="outlined" class="pa-4">
    <div class="text-body-2 text-medium-emphasis text-center">
      Results are not available for this run yet.
    </div>
  </v-card>
</template>

<style scoped>
.table-scroll {
  overflow-x: auto;
}
.qrc-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.qrc-table th,
.qrc-table td {
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 6px 10px;
  text-align: center;
  white-space: nowrap;
}
.qrc-table th {
  font-weight: 500;
  background: rgba(255, 255, 255, 0.04);
}
.qrc-table td.num {
  font-variant-numeric: tabular-nums;
}
.extra-json {
  font-size: 0.75rem;
  background: rgba(255, 255, 255, 0.04);
  padding: 8px;
  border-radius: 4px;
  overflow-x: auto;
  max-height: 240px;
}
</style>
