<script setup lang="ts">
/**
 * SolverMatchRateChart — horizontal bar chart of "fraction of attempted
 * instances where the solver hit the documented optimum" per solver.
 *
 * Pure-SVG / CSS so we don't pull a chart library into the bundle.
 * Sorted descending — the most reliable solver appears at the top.
 * Tier coloring matches the LandingPage / SolverPicker palette.
 */
import { computed } from 'vue'
import type { SolverSummary } from '@/stores/benchmarks'

const props = defineProps<{ summaries: SolverSummary[] }>()

// Hardcoded tier map — same one the backend's /api/solvers exposes.
// Keeping a small copy here avoids a second HTTP roundtrip from the
// Findings page when the registry isn't already in the solve store.
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

const rows = computed(() =>
  [...props.summaries]
    .map((s) => ({
      name: s.solver_name,
      rate: s.instances_attempted ? s.instances_with_match / s.instances_attempted : 0,
      attempted: s.instances_attempted,
      matched: s.instances_with_match,
      runs: s.total_runs,
      color: TIER_COLORS[s.solver_name] || 'default',
    }))
    .sort((a, b) => b.rate - a.rate),
)

function fmtPct(p: number): string {
  return `${Math.round(p * 100)}%`
}
</script>

<template>
  <v-card class="pa-4">
    <div class="d-flex align-center mb-3">
      <v-icon icon="mdi-trophy-outline" class="mr-2" />
      <div class="flex-grow-1">
        <div class="text-subtitle-1 font-weight-medium">Match rate by solver</div>
        <div class="text-caption text-medium-emphasis">
          Fraction of attempted instances where the solver hit the documented optimum.
        </div>
      </div>
    </div>

    <div class="chart-rows">
      <div v-for="r in rows" :key="r.name" class="chart-row">
        <div class="chart-label">
          <code>{{ r.name }}</code>
        </div>
        <div class="chart-track">
          <div
            class="chart-fill"
            :style="{
              width: `${r.rate * 100}%`,
              background: `rgb(var(--v-theme-${r.color}))`,
            }"
          />
          <span class="chart-value">{{ fmtPct(r.rate) }}</span>
        </div>
        <div class="chart-meta">
          {{ r.matched }} / {{ r.attempted }}
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
