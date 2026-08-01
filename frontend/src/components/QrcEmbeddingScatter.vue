<script setup lang="ts">
/**
 * QrcEmbeddingScatter — 2-D feature-space scatter for the Feature-Lab view.
 *
 * Each point is one input step's reservoir feature vector projected to 2-D
 * (PCA or UMAP, computed server-side by ``GET /runs/<id>/embedding``). Points
 * are coloured by the step's target (continuous viridis map) so structure in
 * the feature manifold — clusters, gradients — is visible; test points get a
 * ring so the train/test split is legible. Plain canvas (not uPlot) because we
 * need genuine per-point continuous colour, which uPlot's series model can't do.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { QrcEmbedding } from '@/stores/qrc'

const props = defineProps<{
  embedding: QrcEmbedding | null
}>()

const container = ref<HTMLDivElement | null>(null)
const canvas = ref<HTMLCanvasElement | null>(null)
let resizeObserver: ResizeObserver | null = null

/** Viridis-ish perceptual colour map, t in [0,1] → "rgb(...)". A handful of
 *  anchor stops linearly interpolated — enough for a smooth scatter. */
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
]
function viridis(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1)
  const i = Math.floor(x)
  const f = x - i
  const a = VIRIDIS[i]
  const b = VIRIDIS[Math.min(VIRIDIS.length - 1, i + 1)]
  const r = Math.round(a[0] + (b[0] - a[0]) * f)
  const g = Math.round(a[1] + (b[1] - a[1]) * f)
  const bl = Math.round(a[2] + (b[2] - a[2]) * f)
  return `rgb(${r},${g},${bl})`
}

function draw() {
  const emb = props.embedding
  const cv = canvas.value
  const host = container.value
  if (!cv || !host || !emb || !emb.points.length) return

  const cssW = host.clientWidth || 560
  const cssH = 340
  const dpr = window.devicePixelRatio || 1
  cv.width = cssW * dpr
  cv.height = cssH * dpr
  cv.style.width = `${cssW}px`
  cv.style.height = `${cssH}px`
  const ctx = cv.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)

  const padL = 12
  const padR = 72 // room for the colour bar
  const padT = 12
  const padB = 12
  const plotW = cssW - padL - padR
  const plotH = cssH - padT - padB

  const xs = emb.points.map((p) => p[0])
  const ys = emb.points.map((p) => p[1])
  const xmin = Math.min(...xs)
  const xmax = Math.max(...xs)
  const ymin = Math.min(...ys)
  const ymax = Math.max(...ys)
  const xspan = xmax - xmin || 1
  const yspan = ymax - ymin || 1
  const cmin = Math.min(...emb.color)
  const cmax = Math.max(...emb.color)
  const cspan = cmax - cmin || 1

  const sx = (x: number) => padL + ((x - xmin) / xspan) * plotW
  const sy = (y: number) => padT + (1 - (y - ymin) / yspan) * plotH

  // Draw in two passes so test rings sit on top.
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < emb.points.length; i++) {
      const isTest = emb.split[i] === 'test'
      const isWashout = emb.split[i] === 'washout'
      if (pass === 0 && isTest) continue
      if (pass === 1 && !isTest) continue
      const px = sx(emb.points[i][0])
      const py = sy(emb.points[i][1])
      const t = (emb.color[i] - cmin) / cspan
      ctx.beginPath()
      ctx.arc(px, py, isWashout ? 2.5 : 4, 0, Math.PI * 2)
      ctx.fillStyle = viridis(t)
      ctx.globalAlpha = isWashout ? 0.35 : 0.9
      ctx.fill()
      if (isTest) {
        ctx.globalAlpha = 1
        ctx.lineWidth = 1.5
        ctx.strokeStyle = '#e6edf3'
        ctx.stroke()
      }
    }
  }
  ctx.globalAlpha = 1

  // Colour bar (right gutter).
  const barX = cssW - padR + 22
  const barW = 12
  const barTop = padT + 6
  const barH = plotH - 12
  const steps = 48
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1)
    ctx.fillStyle = viridis(t)
    ctx.fillRect(barX, barTop + (1 - t) * barH, barW, barH / steps + 1)
  }
  ctx.fillStyle = '#9aa7b4'
  ctx.font = '10px system-ui, sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText(cmax.toFixed(2), barX + barW + 4, barTop + 8)
  ctx.fillText(cmin.toFixed(2), barX + barW + 4, barTop + barH)
}

onMounted(() => {
  draw()
  if (container.value) {
    resizeObserver = new ResizeObserver(() => draw())
    resizeObserver.observe(container.value)
  }
})
watch(() => props.embedding, () => draw())
onBeforeUnmount(() => resizeObserver?.disconnect())
</script>

<template>
  <div ref="container" class="embed-scatter">
    <div v-if="!embedding || !embedding.points.length" class="text-body-2 text-medium-emphasis pa-6 text-center">
      No embedding yet — pick a projection and press Compute.
    </div>
    <canvas v-show="embedding && embedding.points.length" ref="canvas"></canvas>
    <div v-if="embedding && embedding.points.length" class="text-caption text-medium-emphasis mt-1 d-flex flex-wrap ga-3">
      <span>colour: {{ embedding.color_label }}</span>
      <span>
        <v-icon icon="mdi-circle-outline" size="x-small" /> test points ringed ·
        faint = washout
      </span>
      <span>{{ embedding.n_points }} steps · {{ embedding.n_features_in }} features → 2-D ({{ embedding.method.toUpperCase() }})</span>
    </div>
    <div v-if="embedding && embedding.points.length" class="text-caption text-medium-emphasis record-name">
      <v-icon icon="mdi-database-outline" size="x-small" />
      record: <code>{{ embedding.trace_name || '(unknown source)' }}</code>
    </div>
  </div>
</template>

<style scoped>
.embed-scatter {
  width: 100%;
}
canvas {
  display: block;
  width: 100%;
}
</style>
