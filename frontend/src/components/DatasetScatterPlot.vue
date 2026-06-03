<script setup lang="ts">
/**
 * DatasetScatterPlot — tiny inline-SVG 2D scatter of a dataset.
 *
 * Renders the points the backend's ``/preview`` endpoint returned. For
 * inherently-2D datasets that's the original features; for higher-
 * dimensional ones it's a 2-PC PCA projection (the backend marks this
 * via the ``pcaApplied`` prop so the chart can label its axes).
 *
 * No charting dependency — same convention as TrainingLossChart and
 * the QAOA explainer's histogram.
 */
import { computed } from 'vue'

interface Point {
  x: number
  y: number
  label: number
}

const props = defineProps<{
  points: Point[]
  classes: string[]
  featureNames: string[]
  pcaApplied: boolean
}>()

const W = 480
const H = 320
const PAD = 28

const bounds = computed(() => {
  if (!props.points.length) {
    return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 }
  }
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
  for (const p of props.points) {
    if (p.x < xMin) xMin = p.x
    if (p.x > xMax) xMax = p.x
    if (p.y < yMin) yMin = p.y
    if (p.y > yMax) yMax = p.y
  }
  // Pad by 8% so points aren't flush with the chart edge.
  const xPad = (xMax - xMin) * 0.08 || 0.5
  const yPad = (yMax - yMin) * 0.08 || 0.5
  return {
    xMin: xMin - xPad,
    xMax: xMax + xPad,
    yMin: yMin - yPad,
    yMax: yMax + yPad,
  }
})

function px(x: number): number {
  const b = bounds.value
  return PAD + ((x - b.xMin) / (b.xMax - b.xMin)) * (W - PAD * 2)
}
function py(y: number): number {
  const b = bounds.value
  // Flip Y so positive points are upper.
  return H - PAD - ((y - b.yMin) / (b.yMax - b.yMin)) * (H - PAD * 2)
}

const CLASS_COLORS = ['#3b82f6', '#f59e0b'] // class 0 = blue, class 1 = orange
</script>

<template>
  <div class="scatter-wrap">
    <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" class="scatter-svg">
      <!-- Axes -->
      <line :x1="PAD" :y1="H - PAD" :x2="W - PAD" :y2="H - PAD"
            stroke="currentColor" stroke-opacity="0.4" />
      <line :x1="PAD" :y1="PAD" :x2="PAD" :y2="H - PAD"
            stroke="currentColor" stroke-opacity="0.4" />

      <!-- Zero lines (only meaningful for standard-scaled data, but we
           always emit them for consistency) -->
      <line
        v-if="bounds.xMin < 0 && bounds.xMax > 0"
        :x1="px(0)" :y1="PAD" :x2="px(0)" :y2="H - PAD"
        stroke="currentColor" stroke-opacity="0.10" stroke-dasharray="3 3"
      />
      <line
        v-if="bounds.yMin < 0 && bounds.yMax > 0"
        :x1="PAD" :y1="py(0)" :x2="W - PAD" :y2="py(0)"
        stroke="currentColor" stroke-opacity="0.10" stroke-dasharray="3 3"
      />

      <!-- Points -->
      <circle
        v-for="(p, i) in points"
        :key="i"
        :cx="px(p.x)"
        :cy="py(p.y)"
        r="3.5"
        :fill="CLASS_COLORS[p.label] || '#9ca3af'"
        fill-opacity="0.75"
        stroke="rgba(0,0,0,0.3)"
        stroke-width="0.5"
      />

      <!-- Axis labels -->
      <text
        :x="W / 2" :y="H - 6"
        text-anchor="middle"
        font-size="11"
        fill="currentColor"
        opacity="0.65"
      >{{ featureNames[0] || 'x' }}</text>
      <text
        :x="10" :y="H / 2"
        text-anchor="middle"
        font-size="11"
        fill="currentColor"
        opacity="0.65"
        :transform="`rotate(-90 10 ${H / 2})`"
      >{{ featureNames[1] || 'y' }}</text>
    </svg>

    <!-- Legend + PCA caption -->
    <div class="d-flex align-center flex-wrap ga-3 mt-2 text-caption">
      <span>
        <span class="legend-swatch" :style="{ background: CLASS_COLORS[0] }"></span>
        {{ classes[0] || 'class 0' }}
      </span>
      <span>
        <span class="legend-swatch" :style="{ background: CLASS_COLORS[1] }"></span>
        {{ classes[1] || 'class 1' }}
      </span>
      <v-spacer />
      <span class="text-medium-emphasis">
        {{ points.length }} points
        <span v-if="pcaApplied">· projected to first 2 PCs</span>
      </span>
    </div>
  </div>
</template>

<style scoped>
.scatter-wrap {
  width: 100%;
}
.scatter-svg {
  width: 100%;
  height: auto;
  display: block;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
}
.legend-swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
}
</style>
