<script setup lang="ts">
/**
 * TimeQualityScatter — speed-vs-accuracy Pareto plot.
 *
 * One dot per (solver, instance) cell that has both a known optimum
 * and a recorded elapsed time. X is log-time, Y is relative gap to
 * the documented optimum (0% = hit it exactly). Colors map to tier so
 * the visual "Pareto frontier" reads as "classical SOTA at bottom-left,
 * quantum at top-right" or vice versa.
 *
 * Pure SVG — no chart library needed.
 */
import { computed, ref } from 'vue'
import type { FindingsCell } from '@/stores/benchmarks'

const props = defineProps<{ cells: FindingsCell[] }>()

const TIER_COLORS: Record<string, string> = {
  exact_cqm: '#4caf50',
  cpsat: '#4caf50',
  highs: '#4caf50',
  gpu_sa: '#7c3aed',
  cpu_sa_neal: '#7c3aed',
  parallel_tempering: '#06b6d4',
  simulated_bifurcation: '#06b6d4',
  qaoa_sim: '#f59e0b',
  qaoa_originqc: '#ef4444',
}

const W = 640
const H = 320
const PAD_L = 60
const PAD_R = 16
const PAD_T = 16
const PAD_B = 44

const points = computed(() => {
  const out: Array<{
    solver: string
    instance: string
    ms: number
    gap_pct: number
    color: string
  }> = []
  for (const c of props.cells) {
    if (c.expected_optimum == null) continue
    if (c.best_user_energy == null) continue
    if (c.best_elapsed_ms == null) continue
    const denom = Math.max(1e-9, Math.abs(c.expected_optimum))
    const gap = Math.abs(c.best_user_energy - c.expected_optimum) / denom
    out.push({
      solver: c.solver_name,
      instance: c.instance_id,
      ms: c.best_elapsed_ms,
      gap_pct: gap * 100,
      color: TIER_COLORS[c.solver_name] || '#888888',
    })
  }
  return out
})

const minMs = computed(() => Math.max(0.5, Math.min(...points.value.map((p) => p.ms), 0.5)))
const maxMs = computed(() => Math.max(...points.value.map((p) => p.ms), 1))
const maxGap = computed(() => Math.max(...points.value.map((p) => p.gap_pct), 1))

const logMin = computed(() => Math.log10(minMs.value))
const logMax = computed(() => Math.log10(maxMs.value))

function xFor(ms: number): number {
  const lo = logMin.value
  const hi = logMax.value
  const t = hi <= lo ? 0.5 : (Math.log10(ms) - lo) / (hi - lo)
  return PAD_L + Math.max(0, Math.min(1, t)) * (W - PAD_L - PAD_R)
}

function yFor(gap_pct: number): number {
  // Cap at maxGap for display; clamp so points at exactly 0 hug the X-axis.
  const t = Math.max(0, Math.min(1, gap_pct / maxGap.value))
  return PAD_T + (1 - t) * (H - PAD_T - PAD_B)
}

// X-axis ticks: power-of-10 between min and max.
const xTicks = computed(() => {
  const lo = Math.floor(logMin.value)
  const hi = Math.ceil(logMax.value)
  const ticks: Array<{ ms: number; x: number; label: string }> = []
  for (let p = lo; p <= hi; p++) {
    const ms = Math.pow(10, p)
    if (ms < minMs.value / 2 || ms > maxMs.value * 2) continue
    ticks.push({ ms, x: xFor(ms), label: fmtTime(ms) })
  }
  return ticks
})

// Y-axis ticks: 0%, 25%, 50%, 75%, 100% of maxGap.
const yTicks = computed(() => {
  const out: Array<{ y: number; label: string }> = []
  for (let pct = 0; pct <= 1.001; pct += 0.25) {
    const g = pct * maxGap.value
    out.push({ y: yFor(g), label: `${g.toFixed(g < 1 ? 1 : 0)}%` })
  }
  return out
})

function fmtTime(ms: number): string {
  if (ms < 1) return `${ms.toFixed(1)} ms`
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`
  return `${(ms / 60000).toFixed(1)} min`
}

// Unique solvers in the data, for the legend.
const legendItems = computed(() => {
  const seen = new Map<string, string>()
  for (const p of points.value) {
    if (!seen.has(p.solver)) seen.set(p.solver, p.color)
  }
  return Array.from(seen.entries()).map(([solver, color]) => ({ solver, color }))
})

const hovered = ref<typeof points.value[number] | null>(null)
</script>

<template>
  <v-card class="pa-4">
    <div class="d-flex align-center mb-3">
      <v-icon icon="mdi-chart-scatter-plot" class="mr-2" />
      <div class="flex-grow-1">
        <div class="text-subtitle-1 font-weight-medium">Speed vs accuracy</div>
        <div class="text-caption text-medium-emphasis">
          Each dot is one (solver, instance). Bottom-left is fast and accurate;
          top-right is slow and far from the documented optimum.
        </div>
      </div>
    </div>

    <div v-if="!points.length" class="text-medium-emphasis text-body-2">
      No runs with a documented optimum on file.
    </div>

    <template v-else>
      <svg
        :viewBox="`0 0 ${W} ${H}`"
        :width="W"
        :height="H"
        class="scatter"
        preserveAspectRatio="xMidYMid meet"
      >
        <!-- Y-axis grid lines + labels -->
        <g class="grid">
          <line
            v-for="t in yTicks"
            :key="`yg-${t.y}`"
            :x1="PAD_L"
            :y1="t.y"
            :x2="W - PAD_R"
            :y2="t.y"
            stroke="rgba(255,255,255,0.06)"
          />
          <text
            v-for="t in yTicks"
            :key="`yl-${t.y}`"
            :x="PAD_L - 6"
            :y="t.y + 4"
            text-anchor="end"
            fill="rgba(255,255,255,0.45)"
            font-size="10"
          >
            {{ t.label }}
          </text>
        </g>
        <!-- X-axis labels -->
        <g class="axis">
          <line
            :x1="PAD_L"
            :y1="H - PAD_B"
            :x2="W - PAD_R"
            :y2="H - PAD_B"
            stroke="rgba(255,255,255,0.15)"
          />
          <text
            v-for="t in xTicks"
            :key="`xl-${t.x}`"
            :x="t.x"
            :y="H - PAD_B + 16"
            text-anchor="middle"
            fill="rgba(255,255,255,0.55)"
            font-size="10"
          >
            {{ t.label }}
          </text>
        </g>
        <!-- Axis titles -->
        <text
          :x="PAD_L + (W - PAD_L - PAD_R) / 2"
          :y="H - 6"
          text-anchor="middle"
          fill="rgba(255,255,255,0.45)"
          font-size="11"
        >
          best wall time (log)
        </text>
        <text
          :x="14"
          :y="PAD_T + (H - PAD_T - PAD_B) / 2"
          fill="rgba(255,255,255,0.45)"
          font-size="11"
          :transform="`rotate(-90, 14, ${PAD_T + (H - PAD_T - PAD_B) / 2})`"
          text-anchor="middle"
        >
          gap to optimum
        </text>

        <!-- Data points -->
        <g>
          <circle
            v-for="(p, i) in points"
            :key="i"
            :cx="xFor(p.ms)"
            :cy="yFor(p.gap_pct)"
            :r="hovered === p ? 6 : 4"
            :fill="p.color"
            stroke="rgba(0,0,0,0.4)"
            stroke-width="0.5"
            fill-opacity="0.78"
            class="point"
            @mouseenter="hovered = p"
            @mouseleave="hovered = null"
          >
            <title>
              {{ p.solver }} · {{ p.instance.split('/').pop() }}
              · {{ fmtTime(p.ms) }} · gap {{ p.gap_pct.toFixed(1) }}%
            </title>
          </circle>
        </g>
      </svg>

      <div class="d-flex flex-wrap ga-2 mt-2">
        <div
          v-for="l in legendItems"
          :key="l.solver"
          class="d-flex align-center"
        >
          <span class="legend-swatch" :style="{ background: l.color }" />
          <code class="text-caption ml-1">{{ l.solver }}</code>
        </div>
      </div>

      <div v-if="hovered" class="hovered-info mt-2">
        <code class="hovered-solver">{{ hovered.solver }}</code>
        on
        <span class="text-medium-emphasis">{{ hovered.instance }}</span>:
        <strong>{{ fmtTime(hovered.ms) }}</strong>
        ·
        <strong>{{ hovered.gap_pct.toFixed(2) }}%</strong> gap
      </div>
    </template>
  </v-card>
</template>

<style scoped>
.scatter {
  width: 100%;
  height: auto;
  max-width: 100%;
}
.point {
  cursor: pointer;
  transition: r 0.12s ease-in-out;
}
.legend-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  opacity: 0.85;
}
.hovered-info {
  font-size: 0.85rem;
  padding: 0.4rem 0.6rem;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 4px;
}
.hovered-solver,
.hovered-info code {
  font-family: 'Cascadia Code', 'Consolas', monospace;
}
</style>
