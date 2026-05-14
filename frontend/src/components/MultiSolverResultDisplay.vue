<script setup lang="ts">
/**
 * MultiSolverResultDisplay — per-solver comparison panel.
 *
 * Renders one row per solver that ran on this job, with energy,
 * elapsed time, feasibility, and a "winner" badge on the row that
 * drove the final ``interpreted_solution``. Rendered above the
 * standard ``ResultDisplay`` (which keeps showing the winning solver's
 * solution / cqm / validation tabs).
 */
import { computed } from 'vue'
import type { Job, SolverResult } from '@/stores/solve'
import { useSolveStore } from '@/stores/solve'

const props = defineProps<{ job: Job }>()
const solve = useSolveStore()

const isMaximize = computed(
  () => props.job.solver_results?.sense === 'maximize',
)

const rows = computed(() => {
  const sr = props.job.solver_results
  if (!sr || !sr.solvers) return []
  const primary = sr.primary
  const out = Object.entries(sr.solvers).map(([name, r]: [string, SolverResult]) => {
    const meta = solve.solvers.find((s) => s.name === name)
    return {
      name,
      meta,
      tier_label: meta?.tier_label || 'Solver',
      tier_color: meta?.tier_color || 'default',
      status: r.status,
      energy: r.energy,
      feasible: r.feasible ?? null,
      elapsed_ms: r.elapsed_ms,
      version: r.version || meta?.version,
      hardware: r.hardware ?? meta?.hardware ?? null,
      error: r.error,
      is_primary: name === primary,
    }
  })
  // Sort: completed-feasible by energy asc, then completed-only by energy asc,
  // then errors at the bottom. Keeps the winner visually at the top.
  const rank = (r: typeof out[number]): number => {
    if (r.status === 'error') return 3
    if (!r.feasible) return 2
    return 1
  }
  return out.sort((a, b) => {
    const ra = rank(a)
    const rb = rank(b)
    if (ra !== rb) return ra - rb
    if (a.energy === undefined) return 1
    if (b.energy === undefined) return -1
    // For maximize problems, higher energy is better — sort descending
    // so the optimum row floats to the top alongside its trophy.
    return isMaximize.value ? b.energy - a.energy : a.energy - b.energy
  })
})

const bestEnergy = computed(() => {
  const completed = rows.value.filter((r) => r.status === 'complete' && r.feasible)
  if (!completed.length) return null
  const energies = completed.map((r) => r.energy as number)
  return isMaximize.value ? Math.max(...energies) : Math.min(...energies)
})

const fastestMs = computed(() => {
  const completed = rows.value.filter((r) => r.status === 'complete')
  if (!completed.length) return null
  return Math.min(...completed.map((r) => r.elapsed_ms))
})

function fmtEnergy(e: number | undefined): string {
  if (e === undefined || e === null) return '—'
  if (Math.abs(e) >= 1000 || (Math.abs(e) < 0.01 && e !== 0)) {
    return e.toExponential(3)
  }
  return e.toFixed(4)
}

function fmtTime(ms: number): string {
  if (ms < 10) return `${ms.toFixed(1)} ms`
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

const completedCount = computed(
  () => rows.value.filter((r) => r.status === 'complete').length,
)
const errorCount = computed(
  () => rows.value.filter((r) => r.status === 'error').length,
)
</script>

<template>
  <v-card class="pa-5 mb-3" v-if="rows.length">
    <div class="d-flex align-center mb-3">
      <v-icon icon="mdi-chart-box-multiple" class="mr-2" />
      <div class="flex-grow-1">
        <div class="text-h6">Solver comparison</div>
        <div class="text-caption text-medium-emphasis">
          {{ rows.length }} solver{{ rows.length === 1 ? '' : 's' }} ·
          {{ completedCount }} completed<span v-if="errorCount > 0">, {{ errorCount }} errored</span>
        </div>
      </div>
      <v-chip
        v-if="bestEnergy !== null"
        size="small"
        color="success"
        variant="tonal"
        prepend-icon="mdi-trophy"
      >
        Best energy {{ fmtEnergy(bestEnergy) }}
      </v-chip>
    </div>

    <v-table density="compact" class="comparison-table">
      <thead>
        <tr>
          <th class="text-left">Solver</th>
          <th class="text-left">Tier</th>
          <th class="text-right">Energy</th>
          <th class="text-center">Feasible</th>
          <th class="text-right">Time</th>
          <th class="text-left">Status</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="r in rows"
          :key="r.name"
          :class="{
            'row-best': r.is_primary,
            'row-error': r.status === 'error',
          }"
        >
          <td>
            <div class="d-flex align-center">
              <v-icon
                v-if="r.is_primary"
                icon="mdi-trophy"
                size="x-small"
                color="success"
                class="mr-1"
              />
              <code class="solver-name">{{ r.name }}</code>
            </div>
            <div class="text-caption text-medium-emphasis">
              <span v-if="r.version">v{{ r.version }}</span>
              <span v-if="r.hardware"> · {{ r.hardware }}</span>
            </div>
          </td>
          <td>
            <v-chip size="x-small" :color="r.tier_color" variant="tonal">
              {{ r.tier_label }}
            </v-chip>
          </td>
          <td class="text-right">
            <span
              v-if="r.status === 'complete'"
              :class="{ 'best-energy': r.energy === bestEnergy }"
            >
              {{ fmtEnergy(r.energy) }}
            </span>
            <span v-else class="text-medium-emphasis">—</span>
          </td>
          <td class="text-center">
            <v-icon
              v-if="r.status === 'complete' && r.feasible"
              icon="mdi-check-circle"
              color="success"
              size="small"
            />
            <v-icon
              v-else-if="r.status === 'complete' && !r.feasible"
              icon="mdi-alert-circle"
              color="warning"
              size="small"
            />
            <span v-else class="text-medium-emphasis">—</span>
          </td>
          <td class="text-right">
            <span :class="{ 'best-time': r.status === 'complete' && r.elapsed_ms === fastestMs }">
              {{ fmtTime(r.elapsed_ms) }}
            </span>
          </td>
          <td>
            <v-chip
              v-if="r.status === 'complete'"
              size="x-small"
              color="success"
              variant="tonal"
            >
              complete
            </v-chip>
            <v-tooltip v-else location="top">
              <template #activator="{ props: tipProps }">
                <v-chip
                  v-bind="tipProps"
                  size="x-small"
                  color="error"
                  variant="tonal"
                >
                  error
                </v-chip>
              </template>
              <span>{{ r.error || 'Unknown error' }}</span>
            </v-tooltip>
          </td>
        </tr>
      </tbody>
    </v-table>

    <div v-if="completedCount > 0" class="text-caption text-medium-emphasis mt-2">
      Best feasible energy drives the interpreted solution shown below.
      Hover the error chips for details.
    </div>
  </v-card>
</template>

<style scoped>
.comparison-table :deep(td),
.comparison-table :deep(th) {
  padding: 6px 8px;
}
.solver-name {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
}
.row-best {
  background: rgba(76, 175, 80, 0.06);
}
.row-error :deep(td) {
  opacity: 0.7;
}
.best-energy {
  color: rgb(var(--v-theme-success));
  font-weight: 600;
}
.best-time {
  color: rgb(var(--v-theme-info));
  font-weight: 500;
}
</style>
