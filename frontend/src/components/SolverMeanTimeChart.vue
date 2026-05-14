<script setup lang="ts">
/**
 * SolverMeanTimeChart — horizontal bar chart of mean solve time per
 * solver, aggregated across every instance in the findings cells.
 *
 * Uses a logarithmic scale because per-solver wall times span four
 * orders of magnitude across the registry (exact_cqm at hundreds of ms
 * vs qaoa_originqc at minutes).
 */
import { computed } from 'vue'
import type { FindingsCell } from '@/stores/benchmarks'

const props = defineProps<{ cells: FindingsCell[] }>()

const TIER_COLORS: Record<string, string> = {
  exact_cqm: 'success',
  cpsat: 'success',
  highs: 'success',
  gpu_sa: 'primary',
  cpu_sa_neal: 'primary',
  parallel_tempering: 'info',
  simulated_bifurcation: 'info',
  qaoa_sim: 'warning',
  qaoa_originqc: 'error',
}

const rows = computed(() => {
  // Aggregate mean_elapsed_ms per solver. Cells without timing data
  // (n_runs=0, never attempted) are skipped.
  const byBolver: Record<string, { sum: number; count: number }> = {}
  for (const c of props.cells) {
    if (c.mean_elapsed_ms == null) continue
    if (!byBolver[c.solver_name]) {
      byBolver[c.solver_name] = { sum: 0, count: 0 }
    }
    byBolver[c.solver_name].sum += c.mean_elapsed_ms
    byBolver[c.solver_name].count += 1
  }
  const list = Object.entries(byBolver).map(([name, agg]) => ({
    name,
    avg_ms: agg.sum / Math.max(1, agg.count),
    color: TIER_COLORS[name] || 'default',
    n_instances: agg.count,
  }))
  return list.sort((a, b) => a.avg_ms - b.avg_ms)
})

const maxLog = computed(() => {
  if (!rows.value.length) return 1
  return Math.log10(Math.max(...rows.value.map((r) => r.avg_ms)))
})

const minLog = computed(() => {
  if (!rows.value.length) return 0
  return Math.log10(Math.max(0.5, Math.min(...rows.value.map((r) => r.avg_ms))))
})

function widthForMs(ms: number): number {
  // Map [minLog, maxLog] → [8, 100] %  (8% floor so the smallest bar is still visible)
  const lo = minLog.value
  const hi = maxLog.value
  if (hi <= lo) return 100
  const t = (Math.log10(ms) - lo) / (hi - lo)
  return 8 + Math.max(0, Math.min(1, t)) * 92
}

function fmtTime(ms: number): string {
  if (ms < 10) return `${ms.toFixed(1)} ms`
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`
  return `${(ms / 60000).toFixed(1)} min`
}
</script>

<template>
  <v-card class="pa-4">
    <div class="d-flex align-center mb-3">
      <v-icon icon="mdi-timer-outline" class="mr-2" />
      <div class="flex-grow-1">
        <div class="text-subtitle-1 font-weight-medium">Mean solve time by solver</div>
        <div class="text-caption text-medium-emphasis">
          Average wall time across attempted instances. Log-scaled — bar widths span four orders of magnitude.
        </div>
      </div>
    </div>

    <div v-if="!rows.length" class="text-medium-emphasis text-body-2">
      No timing data on file.
    </div>

    <div v-else class="chart-rows">
      <div v-for="r in rows" :key="r.name" class="chart-row">
        <div class="chart-label">
          <code>{{ r.name }}</code>
        </div>
        <div class="chart-track">
          <div
            class="chart-fill"
            :style="{
              width: `${widthForMs(r.avg_ms)}%`,
              background: `rgb(var(--v-theme-${r.color}))`,
            }"
          />
          <span class="chart-value">{{ fmtTime(r.avg_ms) }}</span>
        </div>
        <div class="chart-meta">
          {{ r.n_instances }} inst.
        </div>
      </div>
    </div>
  </v-card>
</template>

<style scoped>
.chart-rows {
  display: grid;
  gap: 0.4rem;
}
.chart-row {
  display: grid;
  grid-template-columns: 160px 1fr 80px;
  align-items: center;
  gap: 0.5rem;
}
.chart-label {
  font-size: 0.85rem;
}
.chart-label code {
  font-family: 'Cascadia Code', 'Consolas', monospace;
}
.chart-track {
  position: relative;
  height: 22px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  overflow: hidden;
}
.chart-fill {
  height: 100%;
  transition: width 0.4s ease-in-out;
  opacity: 0.85;
}
.chart-value {
  position: absolute;
  top: 50%;
  right: 0.5rem;
  transform: translateY(-50%);
  font-size: 0.75rem;
  font-weight: 600;
}
.chart-meta {
  text-align: right;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.55);
}
</style>
