<script setup lang="ts">
/**
 * QrcFidTimeChart — time-domain FID plot (real + imaginary vs time).
 *
 * uPlot (not the codebase's usual inline-SVG convention — see
 * ``TrainingLossChart``) because a 2048-point FID re-rendered on every
 * step-slider move needs canvas-speed drawing; SVG with 2k+ path points
 * redrawn on every scrub would visibly lag.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import type { QrcFidData } from '@/stores/qrc'

const props = defineProps<{
  fid: QrcFidData | null
}>()

const container = ref<HTMLDivElement | null>(null)
let plot: uPlot | null = null
let resizeObserver: ResizeObserver | null = null

function buildData(fid: QrcFidData): uPlot.AlignedData {
  return [
    Float64Array.from(fid.t),
    Float64Array.from(fid.real),
    Float64Array.from(fid.imag),
  ]
}

function makeOpts(width: number): uPlot.Options {
  return {
    width,
    height: 260,
    padding: [8, 12, 0, 4],
    cursor: { drag: { x: true, y: false } },
    scales: { x: { time: false } },
    axes: [
      { label: 'time (s)', stroke: '#ccc', grid: { stroke: 'rgba(255,255,255,0.08)' } },
      { label: 'amplitude', stroke: '#ccc', grid: { stroke: 'rgba(255,255,255,0.08)' } },
    ],
    series: [
      {},
      { label: 'real', stroke: '#3b82f6', width: 1.5 },
      { label: 'imag', stroke: '#f97316', width: 1.5 },
    ],
    legend: { show: true },
  }
}

function render() {
  if (!container.value || !props.fid) return
  const width = container.value.clientWidth || 640
  if (!plot) {
    plot = new uPlot(makeOpts(width), buildData(props.fid), container.value)
  } else {
    plot.setData(buildData(props.fid))
  }
}

onMounted(() => {
  render()
  if (container.value) {
    resizeObserver = new ResizeObserver(() => {
      if (plot && container.value) {
        plot.setSize({ width: container.value.clientWidth || 640, height: 260 })
      }
    })
    resizeObserver.observe(container.value)
  }
})

watch(() => props.fid, () => render())

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  plot?.destroy()
  plot = null
})
</script>

<template>
  <div class="fid-time-chart">
    <div v-if="!fid" class="text-body-2 text-medium-emphasis pa-4 text-center">
      No FID data loaded yet.
    </div>
    <div ref="container" class="chart-host"></div>
    <div v-if="fid" class="text-caption text-medium-emphasis mt-1 record-name">
      <v-icon icon="mdi-database-outline" size="x-small" />
      record: <code>{{ fid.trace_name || '(unknown source)' }}</code> · step {{ fid.step }}
    </div>
  </div>
</template>

<style scoped>
.fid-time-chart {
  width: 100%;
}
.chart-host {
  width: 100%;
}
:deep(.u-legend) {
  font-size: 0.75rem;
}
</style>
