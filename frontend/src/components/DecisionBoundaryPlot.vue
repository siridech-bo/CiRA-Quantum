<script setup lang="ts">
/**
 * DecisionBoundaryPlot — VQC predictions rendered as a heatmap behind
 * the training scatter so a student can see, geometrically, *what the
 * model learned*.
 *
 * The probability heatmap interpolates blue → white → orange as
 * P(class=1) goes 0 → 0.5 → 1. The 0.5 contour is the decision
 * boundary itself; anywhere on it the VQC has no opinion. Training
 * points sit on top, colored by their true label — when blue dots are
 * in the blue region and orange dots are in the orange region, the
 * model classifies correctly.
 *
 * Implementation: SVG `<rect>` per grid cell (typically 20×20 = 400
 * cells). For the resolutions we use this is cheap and crisp; canvas
 * would be slightly faster but loses the print-friendly SVG output.
 */
import { computed } from 'vue'
import type { QmlDecisionGrid, QmlScatterPoint } from '@/stores/qml'

const props = defineProps<{
  grid: QmlDecisionGrid
  points?: QmlScatterPoint[]
  classes?: string[]
  featureNames?: string[]
  /** Caption shown in the bottom-right of the chart, e.g. "epoch 12/30"
   *  during training or "final" on completion. */
  epochLabel?: string | null
  /** When true, train + test points are visually distinguished by
   *  border style (solid = train, dashed = test). */
  showSplit?: boolean
}>()

const W = 520
const H = 360
const PAD = 36

const cellW = computed(
  () => (W - PAD * 2) / props.grid.resolution,
)
const cellH = computed(
  () => (H - PAD * 2) / props.grid.resolution,
)

/** Map a probability ∈ [0,1] to a divergent blue-white-orange color.
 *  0 → blue (class 0), 0.5 → white-ish (decision boundary), 1 → orange (class 1). */
function probColor(p: number): string {
  // Anchors:   0 → #3b82f6 (blue),   0.5 → #f5f5f4 (warm white),   1 → #f59e0b (orange)
  const lerp = (a: number, b: number, t: number) => Math.round(a + (b - a) * t)
  let r: number, g: number, b: number
  if (p < 0.5) {
    const t = p / 0.5
    r = lerp(59, 245, t)
    g = lerp(130, 245, t)
    b = lerp(246, 244, t)
  } else {
    const t = (p - 0.5) / 0.5
    r = lerp(245, 245, t)
    g = lerp(245, 158, t)
    b = lerp(244, 11, t)
  }
  return `rgb(${r}, ${g}, ${b})`
}

interface Cell {
  x: number
  y: number
  fill: string
  /** Distance from the 0.5 boundary, used to highlight the boundary band. */
  boundary_strength: number
}

const cells = computed<Cell[]>(() => {
  const g = props.grid
  const xs = Array.from({ length: g.resolution }, (_, i) =>
    g.x_min + ((i + 0.5) / g.resolution) * (g.x_max - g.x_min),
  )
  const ys = Array.from({ length: g.resolution }, (_, i) =>
    g.y_min + ((i + 0.5) / g.resolution) * (g.y_max - g.y_min),
  )
  const out: Cell[] = []
  for (let row = 0; row < g.resolution; row++) {
    for (let col = 0; col < g.resolution; col++) {
      const p = g.probabilities[row * g.resolution + col]
      out.push({
        x: xs[col],
        y: ys[row],
        fill: probColor(p),
        boundary_strength: 1 - Math.abs(p - 0.5) * 2,  // 1 at p=0.5, 0 at edges
      })
    }
  }
  return out
})

function dataToSvgX(x: number): number {
  const g = props.grid
  return PAD + ((x - g.x_min) / (g.x_max - g.x_min)) * (W - PAD * 2)
}
function dataToSvgY(y: number): number {
  // Flip Y so positive y goes up.
  const g = props.grid
  return H - PAD - ((y - g.y_min) / (g.y_max - g.y_min)) * (H - PAD * 2)
}

const CLASS_BORDER_COLORS = ['#1d4ed8', '#b45309']  // darker blue / amber for contrast
</script>

<template>
  <div class="boundary-wrap">
    <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" class="boundary-svg">
      <!-- Probability heatmap. Cells positioned in DATA coordinates so
           the grid lines up with the scatter exactly. -->
      <g>
        <rect
          v-for="(c, i) in cells"
          :key="i"
          :x="dataToSvgX(c.x) - cellW / 2"
          :y="dataToSvgY(c.y) - cellH / 2"
          :width="cellW + 0.5"
          :height="cellH + 0.5"
          :fill="c.fill"
          fill-opacity="0.85"
        />
      </g>

      <!-- 0.5 boundary highlight: a soft white ribbon over cells where
           the model is on the fence. Pure cosmetic — pulls the eye to
           where the decision is happening. -->
      <g>
        <rect
          v-for="(c, i) in cells"
          v-show="c.boundary_strength > 0.85"
          :key="`b-${i}`"
          :x="dataToSvgX(c.x) - cellW / 2"
          :y="dataToSvgY(c.y) - cellH / 2"
          :width="cellW + 0.5"
          :height="cellH + 0.5"
          fill="white"
          :fill-opacity="(c.boundary_strength - 0.85) * 2"
        />
      </g>

      <!-- Axes (thin frames around the heatmap) -->
      <rect
        :x="PAD" :y="PAD"
        :width="W - PAD * 2" :height="H - PAD * 2"
        fill="none"
        stroke="currentColor" stroke-opacity="0.5"
      />

      <!-- Scatter points -->
      <g v-if="points">
        <circle
          v-for="(p, i) in points"
          :key="`pt-${i}`"
          :cx="dataToSvgX(p.x)"
          :cy="dataToSvgY(p.y)"
          r="4"
          :fill="p.label === 0 ? '#3b82f6' : '#f59e0b'"
          :stroke="CLASS_BORDER_COLORS[p.label] || '#000'"
          stroke-width="1"
          :stroke-dasharray="showSplit && p.split === 'test' ? '2 1.5' : ''"
        />
      </g>

      <!-- Axis labels -->
      <text
        :x="W / 2" :y="H - 6"
        text-anchor="middle"
        font-size="11"
        fill="currentColor"
        opacity="0.75"
      >{{ featureNames?.[0] || 'x' }}</text>
      <text
        :x="12" :y="H / 2"
        text-anchor="middle"
        font-size="11"
        fill="currentColor"
        opacity="0.75"
        :transform="`rotate(-90 12 ${H / 2})`"
      >{{ featureNames?.[1] || 'y' }}</text>

      <!-- Epoch label (top-right) -->
      <text
        v-if="epochLabel"
        :x="W - 8" :y="20"
        text-anchor="end"
        font-size="11"
        font-weight="600"
        fill="currentColor"
        opacity="0.75"
      >{{ epochLabel }}</text>
    </svg>

    <!-- Legend -->
    <div class="d-flex align-center flex-wrap ga-3 mt-2 text-caption">
      <span class="d-flex align-center">
        <span class="legend-dot" style="background:#3b82f6"></span>
        {{ classes?.[0] || 'class 0' }} (P → 0)
      </span>
      <span class="d-flex align-center">
        <span class="legend-gradient"></span>
        decision boundary
      </span>
      <span class="d-flex align-center">
        <span class="legend-dot" style="background:#f59e0b"></span>
        {{ classes?.[1] || 'class 1' }} (P → 1)
      </span>
      <v-spacer />
      <span v-if="showSplit" class="text-medium-emphasis">
        solid border = train, dashed = test
      </span>
    </div>
  </div>
</template>

<style scoped>
.boundary-wrap {
  width: 100%;
}
.boundary-svg {
  width: 100%;
  height: auto;
  display: block;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
}
.legend-dot {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
  border: 1px solid rgba(0, 0, 0, 0.3);
}
.legend-gradient {
  display: inline-block;
  width: 36px;
  height: 10px;
  margin-right: 6px;
  vertical-align: middle;
  background: linear-gradient(
    90deg,
    #3b82f6 0%,
    #f5f5f4 50%,
    #f59e0b 100%
  );
  border-radius: 2px;
}
</style>
