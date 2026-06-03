<script setup lang="ts">
/**
 * Inline-SVG heatmap of a 0/1 (or higher-cardinality) matrix.
 *
 * Sprint 1 ships this for the qLDPC parity-check matrices (H_X, H_Z).
 * Each row of ``matrix`` is rendered as a horizontal strip of cells;
 * non-zero cells are filled with ``fillColor``, zero cells are left
 * background-transparent. The component scrolls horizontally for wide
 * matrices (BBCode is 72×144 = 144 columns) and vertically as needed.
 *
 * Mirrors the pattern of [DecisionBoundaryPlot.vue] / [TrainingLossChart.vue]:
 * pure CSS-styled inline SVG, no external charting deps.
 */
import { computed } from 'vue'

interface Props {
  /** Row-major matrix of 0/1 entries. */
  matrix: number[][]
  /** Optional caption rendered above the grid. */
  title?: string
  /** Subtitle rendered below the title (e.g. shape string). */
  subtitle?: string
  /** Pixel size per cell. Falls back to an auto-scale capped at 12 px. */
  cellSize?: number
  /** SVG fill color for non-zero entries. */
  fillColor?: string
  /** Background grid color (subtle gridlines on the 0 cells). */
  gridColor?: string
  /** Maximum on-screen width before horizontal scroll kicks in. */
  maxWidth?: number
  /** Maximum on-screen height before vertical scroll kicks in. */
  maxHeight?: number
  /** Show row/column count + non-zero count badge below the grid. */
  showFooter?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  cellSize: 0,
  fillColor: '#42a5f5',
  gridColor: 'rgba(255,255,255,0.05)',
  maxWidth: 720,
  maxHeight: 360,
  showFooter: true,
  title: '',
  subtitle: '',
})

const rows = computed(() => props.matrix.length)
const cols = computed(() => (props.matrix[0]?.length ?? 0))

const autoCellSize = computed(() => {
  if (props.cellSize > 0) return props.cellSize
  // Pick a size that fits the matrix in maxWidth × maxHeight without
  // shrinking below 2 px (else the cells become unreadable).
  if (cols.value === 0 || rows.value === 0) return 8
  const byWidth = Math.floor(props.maxWidth / cols.value)
  const byHeight = Math.floor(props.maxHeight / rows.value)
  return Math.max(2, Math.min(12, byWidth, byHeight))
})

const width = computed(() => cols.value * autoCellSize.value)
const height = computed(() => rows.value * autoCellSize.value)

const nonzeroCount = computed(() => {
  let total = 0
  for (const row of props.matrix) {
    for (const v of row) if (v !== 0) total++
  }
  return total
})

// Flatten to {x, y} list of non-zero cells. SVG renders this as a
// single big <path> or as many small <rect>s — we use rects because
// they're easier to inspect in devtools and the matrices we render
// (up to ~84×169) have ≤ ~1500 non-zero cells which performs fine.
const nonZeroCells = computed(() => {
  const cells: { x: number; y: number }[] = []
  const cs = autoCellSize.value
  for (let r = 0; r < rows.value; r++) {
    const row = props.matrix[r]
    for (let c = 0; c < cols.value; c++) {
      if (row[c] !== 0) cells.push({ x: c * cs, y: r * cs })
    }
  }
  return cells
})
</script>

<template>
  <div class="matrix-viewer">
    <div v-if="title" class="text-subtitle-2 font-weight-medium mb-1">{{ title }}</div>
    <div v-if="subtitle" class="text-caption text-medium-emphasis mb-2">{{ subtitle }}</div>

    <div
      class="matrix-scroll"
      :style="{ maxWidth: `${maxWidth}px`, maxHeight: `${maxHeight}px` }"
    >
      <svg
        :width="width"
        :height="height"
        :viewBox="`0 0 ${width} ${height}`"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        :aria-label="title || 'matrix heatmap'"
      >
        <!-- Background grid: faint outlines so 0-cells are visible. -->
        <rect
          v-for="r in rows"
          :key="`r${r}`"
          :x="0"
          :y="(r - 1) * autoCellSize"
          :width="width"
          :height="autoCellSize"
          fill="transparent"
          :stroke="gridColor"
          stroke-width="0.5"
        />
        <!-- Non-zero cells: filled rects. -->
        <rect
          v-for="(cell, i) in nonZeroCells"
          :key="`c${i}`"
          :x="cell.x"
          :y="cell.y"
          :width="autoCellSize"
          :height="autoCellSize"
          :fill="fillColor"
        />
      </svg>
    </div>

    <div v-if="showFooter" class="d-flex align-center mt-2 ga-2 flex-wrap">
      <v-chip size="x-small" variant="outlined">
        {{ rows }} rows × {{ cols }} cols
      </v-chip>
      <v-chip size="x-small" variant="tonal" color="primary">
        {{ nonzeroCount.toLocaleString() }} non-zeros
      </v-chip>
      <span class="text-caption text-medium-emphasis">
        density {{ rows && cols ? ((nonzeroCount / (rows * cols)) * 100).toFixed(1) : 0 }}%
      </span>
    </div>
  </div>
</template>

<style scoped>
.matrix-viewer {
  font-family: inherit;
}
.matrix-scroll {
  overflow: auto;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 4px;
  background: rgba(0, 0, 0, 0.2);
  padding: 4px;
}
.matrix-scroll svg {
  display: block;
}
</style>
