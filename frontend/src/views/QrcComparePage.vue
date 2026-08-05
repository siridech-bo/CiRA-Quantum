<script setup lang="ts">
/**
 * QrcComparePage — /qrc/compare. Select N runs (+ benchmarks/tiers) and see them
 * side by side: a badge per item (task / readout / D / regime), the comparability
 * guard (SOP §0 — same task/readout/D or it warns), a config-diff table showing
 * exactly which parameters differ, and a metric table with bar comparison.
 *
 * Deep-linkable: /qrc/compare?runs=id1,id2 preselects runs.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useQrcStore, type QrcCompareResult } from '@/stores/qrc'
import CiraLogo from '@/components/CiraLogo.vue'

const route = useRoute()
const router = useRouter()
const qrc = useQrcStore()

const loading = ref(true)
const fatal = ref<string | null>(null)
const selectedRuns = ref<string[]>([])
const selectedRefs = ref<string[]>([])
const result = ref<QrcCompareResult | null>(null)
const comparing = ref(false)

const runItems = computed(() =>
  [...qrc.runs]
    .sort((a, b) => new Date(b.created_utc).getTime() - new Date(a.created_utc).getTime())
    .map((r) => ({ title: `${r.id}  ·  ${r.task}  ·  ${r.status}`, value: r.id })),
)
const refItems = computed(() =>
  qrc.references.map((r) => ({ title: `${r.label}  ·  ${r.kind}`, value: r.id })),
)

async function runCompare() {
  if (!selectedRuns.value.length) { result.value = null; return }
  comparing.value = true
  try {
    result.value = await qrc.compareRuns(selectedRuns.value, selectedRefs.value)
    fatal.value = null
  } catch (e: any) {
    fatal.value = e?.response?.data?.error || e?.message || 'Compare failed'
  } finally {
    comparing.value = false
  }
}

watch([selectedRuns, selectedRefs], runCompare, { deep: true })

const badgeColor: Record<string, string> = {
  memory: 'teal', learnable: 'purple', narma: 'blue', weather: 'cyan',
}

/** For a metric column, the max |value| across items (for bar scaling). */
function metricMax(key: string): number {
  const xs = (result.value?.items ?? [])
    .map((it) => it.metrics?.[key])
    .filter((v): v is number => typeof v === 'number')
    .map((v) => Math.abs(v))
  return xs.length ? Math.max(...xs) : 1
}
function barPct(key: string, v: number | undefined): number {
  if (typeof v !== 'number') return 0
  const m = metricMax(key)
  return m > 0 ? Math.min(100, (Math.abs(v) / m) * 100) : 0
}
function fmt(v: any): string {
  if (typeof v !== 'number') return v == null ? '—' : String(v)
  if (v !== 0 && Math.abs(v) < 1e-2) return v.toExponential(2)
  return v.toFixed(4)
}

onMounted(async () => {
  loading.value = true
  try {
    await Promise.all([qrc.loadRuns().catch(() => {}), qrc.loadReferences().catch(() => {})])
    const q = route.query.runs
    if (typeof q === 'string' && q) selectedRuns.value = q.split(',').filter(Boolean)
    if (selectedRuns.value.length) await runCompare()
  } catch (e: any) {
    fatal.value = e?.message || 'Failed to load'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QRC compare app bar">
    <div class="d-flex align-center logo-link ml-3" role="button" tabindex="0"
         @click="router.push('/qrc')" @keydown.enter="router.push('/qrc')">
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QRC / Compare</span>
    </div>
    <v-spacer />
    <v-btn variant="text" prepend-icon="mdi-format-list-bulleted" @click="router.push('/qrc')">Runs</v-btn>
    <v-btn variant="text" prepend-icon="mdi-plus" @click="router.push('/qrc/new')">New</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <div class="pt-8 pb-4">
        <div class="text-overline text-medium-emphasis">Run comparison</div>
        <div class="text-h4 font-weight-bold mb-2">Compare runs vs benchmarks &amp; tiers</div>
        <div class="text-body-1 text-medium-emphasis" style="max-width: 760px">
          Pick runs and references to see them side by side. Comparisons are only
          valid within the same task and readout/D — mixing them is flagged, never
          silently averaged (SOP §0).
        </div>
      </div>

      <v-alert v-if="fatal" type="error" variant="tonal" class="mb-4">{{ fatal }}</v-alert>
      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />

      <template v-else>
        <v-card variant="outlined" class="pa-4 mb-4">
          <v-row dense>
            <v-col cols="12" md="7">
              <v-select v-model="selectedRuns" :items="runItems" multiple chips closable-chips
                        label="Runs" variant="outlined" density="comfortable" hide-details />
            </v-col>
            <v-col cols="12" md="5">
              <v-select v-model="selectedRefs" :items="refItems" multiple chips closable-chips
                        label="Benchmarks / tiers" variant="outlined" density="comfortable" hide-details />
            </v-col>
          </v-row>
        </v-card>

        <v-progress-linear v-if="comparing" indeterminate class="mb-4" />

        <template v-if="result">
          <!-- comparability guard -->
          <v-alert v-for="(w, i) in result.warnings" :key="i" type="warning" variant="tonal"
                   density="compact" class="mb-2" icon="mdi-alert">{{ w }}</v-alert>
          <v-alert v-if="result.comparable && !result.warnings.length" type="success" variant="tonal"
                   density="compact" class="mb-4" icon="mdi-check-circle">
            Comparable — same task and readout/D.
          </v-alert>

          <!-- item badges -->
          <div class="d-flex flex-wrap ga-3 mb-4">
            <v-card v-for="it in result.items" :key="it.id" variant="outlined" class="pa-3" min-width="220">
              <div class="d-flex align-center ga-2 mb-1">
                <v-icon size="16" :icon="it.kind === 'reference' ? 'mdi-bookmark' : 'mdi-flask'" />
                <span class="text-body-2 font-weight-medium text-truncate" style="max-width: 220px">
                  {{ it.label }}
                </span>
              </div>
              <div v-if="it.missing" class="text-caption text-error">not found</div>
              <template v-else>
                <div class="d-flex flex-wrap ga-1">
                  <v-chip size="x-small" :color="badgeColor[it.task || ''] || 'grey'" variant="tonal">
                    {{ it.task }}
                  </v-chip>
                  <v-chip v-if="it.readout" size="x-small" variant="outlined">{{ it.readout }}</v-chip>
                  <v-chip v-if="it.D" size="x-small" variant="outlined">D={{ it.D }}</v-chip>
                </div>
                <div v-if="it.regime" class="text-caption text-medium-emphasis mt-1">{{ it.regime }}</div>
                <div v-if="it.note" class="text-caption text-medium-emphasis mt-1">{{ it.note }}</div>
              </template>
            </v-card>
          </div>

          <!-- metric table with bars -->
          <v-card variant="outlined" class="pa-4 mb-4">
            <div class="text-overline text-medium-emphasis mb-2">Metrics</div>
            <div v-if="!result.metric_keys.length" class="text-body-2 text-medium-emphasis">
              No scalar metrics yet (runs may still be in progress).
            </div>
            <div v-else class="table-scroll">
              <v-table density="comfortable">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th v-for="it in result.items" :key="it.id" class="text-right">
                      {{ it.label }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="key in result.metric_keys" :key="key">
                    <td class="text-medium-emphasis">{{ key }}</td>
                    <td v-for="it in result.items" :key="it.id" class="text-right" style="min-width: 140px">
                      <div class="d-flex align-center justify-end ga-2">
                        <span class="metric-val">{{ fmt(it.metrics?.[key]) }}</span>
                        <div class="bar-track">
                          <div class="bar-fill"
                               :class="it.kind === 'reference' ? 'ref' : 'run'"
                               :style="{ width: barPct(key, it.metrics?.[key]) + '%' }" />
                        </div>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </v-table>
            </div>
          </v-card>

          <!-- config diff -->
          <v-card variant="outlined" class="pa-4">
            <div class="text-overline text-medium-emphasis mb-2">Config differences</div>
            <div v-if="!Object.keys(result.config_diff).length" class="text-body-2 text-medium-emphasis">
              Selected runs share the same config (no differing parameters).
            </div>
            <div v-else class="table-scroll">
              <v-table density="comfortable">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th v-for="it in result.items.filter((x) => x.kind === 'run' && !x.missing)"
                        :key="it.id" class="text-right">{{ it.label }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(vals, key) in result.config_diff" :key="key">
                    <td class="text-medium-emphasis">{{ key }}</td>
                    <td v-for="it in result.items.filter((x) => x.kind === 'run' && !x.missing)"
                        :key="it.id" class="text-right">
                      <code class="text-caption">{{ fmt(vals[it.id]) }}</code>
                    </td>
                  </tr>
                </tbody>
              </v-table>
            </div>
          </v-card>
        </template>

        <v-alert v-else-if="!selectedRuns.length" type="info" variant="tonal" density="compact">
          Select at least one run above to compare.
        </v-alert>
      </template>
    </v-container>
  </v-main>
</template>

<style scoped>
.logo-link { cursor: pointer; transition: opacity 0.15s ease-in-out; }
.logo-link:hover { opacity: 0.8; }
.table-scroll { overflow-x: auto; }
.metric-val { font-variant-numeric: tabular-nums; min-width: 72px; display: inline-block; text-align: right; }
.bar-track { width: 48px; height: 8px; background: rgba(255, 255, 255, 0.08); border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-fill.run { background: #4da3ff; }
.bar-fill.ref { background: #e07b3c; }
</style>
