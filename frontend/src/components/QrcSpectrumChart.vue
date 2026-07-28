<script setup lang="ts">
/**
 * QrcSpectrumChart — frequency-domain magnitude spectrum with the
 * selected 653 ``peaks_hz`` highlighted as markers on the curve.
 *
 * Same uPlot rationale as QrcFidTimeChart — canvas-speed redraw on
 * step-slider scrub.
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

/** Build the sparse "peaks" series: null everywhere except at the
 *  freq_hz index nearest each entry in peaks_hz, where it takes the
 *  curve's own magnitude value (so the marker sits exactly on the line). */
function buildPeaksSeries(fid: QrcFidData): (number | null)[] {
  const out: (number | null)[] = new Array(fid.freq_hz.length).fill(null)
  for (const peakHz of fid.peaks_hz) {
    let bestIdx = -1
    let bestDist = Infinity
    for (let i = 0; i < fid.freq_hz.length; i++) {
      const d = Math.abs(fid.freq_hz[i] - peakHz)
      if (d < bestDist) {
        bestDist = d
        bestIdx = i
      }
    }
    if (bestIdx >= 0) out[bestIdx] = fid.spectrum_mag[bestIdx]
  }
  return out
}

function buildData(fid: QrcFidData): uPlot.AlignedData {
  return [
    Float64Array.from(fid.freq_hz),
    Float64Array.from(fid.spectrum_mag),
    buildPeaksSeries(fid) as any,
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
      { label: 'frequency (Hz)', stroke: '#ccc', grid: { stroke: 'rgba(255,255,255,0.08)' } },
      { label: '|spectrum|', stroke: '#ccc', grid: { stroke: 'rgba(255,255,255,0.08)' } },
    ],
    series: [
      {},
      { label: 'magnitude', stroke: '#22c55e', width: 1.5 },
      {
        label: 'peaks',
        stroke: 'transparent',
        width: 0,
        points: { show: true, size: 7, fill: '#f43f5e', stroke: '#f43f5e' },
        paths: () => null,
      },
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
  <div class="fid-spectrum-chart">
    <div v-if="!fid" class="text-body-2 text-medium-emphasis pa-4 text-center">
      No spectrum data loaded yet.
    </div>
    <div ref="container" class="chart-host"></div>
    <div v-if="fid" class="text-caption text-medium-emphasis mt-1">
      <v-icon icon="mdi-map-marker" size="x-small" color="#f43f5e" />
      {{ fid.peaks_hz.length }} selected peak bins marked
    </div>
  </div>
</template>

<style scoped>
.fid-spectrum-chart {
  width: 100%;
}
.chart-host {
  width: 100%;
}
:deep(.u-legend) {
  font-size: 0.75rem;
}
</style>
