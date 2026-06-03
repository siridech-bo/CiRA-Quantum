<script setup lang="ts">
/**
 * QuirkPlayground — iframe wrapper around the vendored Quirk-E build.
 *
 * Quirk-E (https://github.com/DEQSE-Project/Quirk-E, Apache-2.0) is a
 * single-file HTML drag-drop quantum-circuit simulator that encodes its
 * entire state in the URL hash fragment as JSON:
 *
 *   /quirk/quirk.html#circuit={"cols":[...]}
 *
 * The build is self-hosted under /public/quirk/ — see public/quirk/NOTICE
 * for license + commit-SHA provenance. Same-origin so we keep the option
 * to read the iframe's hash later (Phase 3 bridge).
 *
 * Initial-circuit prop: the Quirk JSON shape, NOT Qni's. Quirk uses
 * column-major arrays where each column is one moment; gate IDs are
 * strings like "H", "X", "•" (control), "Measure". A `null` slot means
 * identity. The bridge utility in Phase 3 will build these from `info`.
 */
import { computed } from 'vue'

export interface QuirkCircuitJson {
  cols: Array<Array<string | number | null>>
}

const props = withDefaults(
  defineProps<{
    /** Quirk-format circuit JSON. Omitted → empty grid. */
    initialCircuit?: QuirkCircuitJson
    /** Pixel height for the iframe. Quirk's layout works well at >= 480px. */
    height?: number
  }>(),
  {
    height: 640,
  },
)

const src = computed(() => {
  const base = '/quirk/quirk.html'
  if (!props.initialCircuit) return base
  return `${base}#circuit=${encodeURIComponent(JSON.stringify(props.initialCircuit))}`
})
</script>

<template>
  <iframe
    :src="src"
    :style="{ height: height + 'px' }"
    class="quirk-iframe"
    title="Quirk-E quantum circuit playground"
    sandbox="allow-scripts allow-same-origin"
    loading="lazy"
  />
</template>

<style scoped>
.quirk-iframe {
  display: block;
  width: 100%;
  border: 0;
  background: #ffffff;
  border-radius: 6px;
}
</style>
