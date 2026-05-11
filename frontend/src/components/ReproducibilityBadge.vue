<script setup lang="ts">
/**
 * Phase 5C — reproducibility badge.
 *
 * Shows a compact "repro:abc123 · cuda:..." chip with hover tooltip
 * carrying the full hash + code version + hardware ID. The same
 * triple is what `replay_record` checks against on re-execution; if
 * any of them drift, the dashboard's time-series view will surface
 * the divergence.
 */
const props = defineProps<{
  reproHash: string
  hardwareId?: string | null
  codeVersion?: string | null
  size?: 'x-small' | 'small' | 'default'
}>()

function shortHash(h: string): string {
  return h.length > 8 ? h.slice(0, 8) : h
}

function shortHardware(hw?: string | null): string {
  if (!hw) return ''
  // CUDA hardware IDs look like "cuda:NVIDIA GeForce RTX 5070 Ti|cc12.0|cuda_runtime=12.8".
  // Pull the leading "cuda" / "cpu" prefix for a compact label.
  if (hw.startsWith('cuda:')) return 'cuda'
  if (hw.startsWith('cpu')) return 'cpu'
  return hw.split('|')[0]
}
</script>

<template>
  <v-tooltip location="bottom">
    <template #activator="{ props: tProps }">
      <v-chip
        v-bind="tProps"
        :size="size || 'small'"
        variant="outlined"
        prepend-icon="mdi-fingerprint"
      >
        repro:{{ shortHash(reproHash) }}
        <span v-if="hardwareId" class="ml-1 text-medium-emphasis">
          · {{ shortHardware(hardwareId) }}
        </span>
      </v-chip>
    </template>
    <div class="text-caption">
      <div><strong>Hash:</strong> {{ reproHash }}</div>
      <div v-if="hardwareId"><strong>Hardware:</strong> {{ hardwareId }}</div>
      <div v-if="codeVersion"><strong>Code version:</strong> {{ codeVersion }}</div>
      <div class="mt-1 text-medium-emphasis">
        These three together pin the exact run. Replay will report
        agreement against this triple.
      </div>
    </div>
  </v-tooltip>
</template>
