<script setup lang="ts">
/**
 * TrainingLossChart — inline-SVG loss + accuracy curve.
 *
 * No Chart.js dependency — the codebase renders all its charts via raw
 * SVG (see ``SolverMeanTimeChart``, ``TimeQualityScatter``). This keeps
 * the bundle small and the layout predictable across Vuetify breakpoints.
 *
 * Two y-axes:
 *   - Loss (red, left)     plotted as a line on log scale if loss > 0.
 *   - Accuracy (blue/green, right) plotted on a linear [0, 1] scale.
 */
import { computed } from 'vue'
import type { QmlEpochPoint } from '@/stores/qml'

const props = defineProps<{
  history: QmlEpochPoint[]
  totalEpochs?: number | null
}>()

const W = 720
const H = 280
const PAD_L = 50
const PAD_R = 50
const PAD_T = 16
const PAD_B = 36

const xMax = computed(() =>
  Math.max(props.totalEpochs || 0, ...props.history.map((p) => p.epoch), 1),
)

const lossMax = computed(() => {
  if (!props.history.length) return 1
  return Math.max(...props.history.map((p) => p.loss), 0.0001)
})

function xAt(epoch: number): number {
  return PAD_L + ((epoch - 1) / Math.max(1, xMax.value - 1)) * (W - PAD_L - PAD_R)
}
function lossY(loss: number): number {
  return PAD_T + (1 - loss / lossMax.value) * (H - PAD_T - PAD_B)
}
function accY(acc: number): number {
  return PAD_T + (1 - acc) * (H - PAD_T - PAD_B)
}

const lossPath = computed(() => {
  if (!props.history.length) return ''
  return props.history
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xAt(p.epoch)} ${lossY(p.loss)}`)
    .join(' ')
})

const trainAccPath = computed(() => {
  if (!props.history.length) return ''
  return props.history
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xAt(p.epoch)} ${accY(p.train_accuracy)}`)
    .join(' ')
})

const testAccPath = computed(() => {
  if (!props.history.length) return ''
  return props.history
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xAt(p.epoch)} ${accY(p.test_accuracy)}`)
    .join(' ')
})

const last = computed<QmlEpochPoint | null>(() =>
  props.history.length ? props.history[props.history.length - 1] : null,
)
</script>

<template>
  <div class="loss-chart">
    <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" class="w-100">
      <!-- left y-axis (loss) -->
      <line :x1="PAD_L" :y1="PAD_T" :x2="PAD_L" :y2="H - PAD_B" stroke="currentColor" opacity="0.3" />
      <!-- right y-axis (accuracy) -->
      <line
        :x1="W - PAD_R" :y1="PAD_T"
        :x2="W - PAD_R" :y2="H - PAD_B"
        stroke="currentColor" opacity="0.3"
      />
      <!-- x-axis -->
      <line
        :x1="PAD_L" :y1="H - PAD_B"
        :x2="W - PAD_R" :y2="H - PAD_B"
        stroke="currentColor" opacity="0.3"
      />

      <!-- gridlines for accuracy at 0.5 and 1.0 -->
      <line
        v-for="g in [0.5, 1]"
        :key="g"
        :x1="PAD_L" :x2="W - PAD_R"
        :y1="accY(g)" :y2="accY(g)"
        stroke="currentColor" opacity="0.1"
        stroke-dasharray="2 4"
      />

      <!-- left axis label: loss -->
      <text :x="PAD_L - 8" :y="PAD_T + 4" text-anchor="end" font-size="10" fill="#ef4444">
        {{ lossMax.toFixed(2) }}
      </text>
      <text :x="PAD_L - 8" :y="H - PAD_B" text-anchor="end" font-size="10" fill="#ef4444">
        0
      </text>
      <text
        :x="12" :y="H / 2"
        text-anchor="middle" font-size="11" fill="#ef4444"
        :transform="`rotate(-90 12 ${H / 2})`"
      >loss</text>

      <!-- right axis label: accuracy -->
      <text :x="W - PAD_R + 8" :y="PAD_T + 4" text-anchor="start" font-size="10" fill="#3b82f6">
        100%
      </text>
      <text :x="W - PAD_R + 8" :y="accY(0.5)" text-anchor="start" font-size="10" fill="#3b82f6">
        50%
      </text>
      <text :x="W - PAD_R + 8" :y="H - PAD_B" text-anchor="start" font-size="10" fill="#3b82f6">
        0
      </text>

      <!-- x-axis label -->
      <text
        :x="(W - PAD_L - PAD_R) / 2 + PAD_L" :y="H - 8"
        text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7"
      >
        epoch ({{ history.length }} / {{ totalEpochs || xMax }})
      </text>

      <!-- loss line -->
      <path :d="lossPath" fill="none" stroke="#ef4444" stroke-width="2" />
      <!-- test accuracy line (green) -->
      <path :d="testAccPath" fill="none" stroke="#22c55e" stroke-width="2" />
      <!-- train accuracy line (blue, lighter) -->
      <path :d="trainAccPath" fill="none" stroke="#3b82f6" stroke-width="2" stroke-opacity="0.6" stroke-dasharray="4 2" />
    </svg>

    <!-- Legend + current values -->
    <div class="d-flex flex-wrap ga-3 mt-2 text-caption">
      <span><span class="swatch" style="background:#ef4444"></span> loss</span>
      <span><span class="swatch" style="background:#22c55e"></span> test accuracy</span>
      <span><span class="swatch swatch-dashed" style="background:#3b82f6"></span> train accuracy</span>
      <v-spacer />
      <span v-if="last" class="text-medium-emphasis">
        epoch {{ last.epoch }} —
        loss {{ last.loss.toFixed(4) }} ·
        train {{ (last.train_accuracy * 100).toFixed(1) }}% ·
        test {{ (last.test_accuracy * 100).toFixed(1) }}%
      </span>
    </div>
  </div>
</template>

<style scoped>
.loss-chart {
  width: 100%;
}
.swatch {
  display: inline-block;
  width: 12px;
  height: 8px;
  margin-right: 4px;
  vertical-align: middle;
  border-radius: 2px;
}
.swatch-dashed {
  background-image: repeating-linear-gradient(
    90deg, currentColor 0 4px, transparent 4px 6px
  );
}
</style>
