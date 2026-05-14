<script setup lang="ts">
/**
 * CiRA Quantum logo — Option B (abstract-quantum).
 *
 * Two overlapping circles in the platform's primary / accent colors
 * represent the superposition of basis states; the white core where
 * they overlap is the qubit itself. The thin outer ring is the
 * Bloch-sphere allusion that ties the mark to the physics rather
 * than just decoration.
 *
 * Sizing is via the ``size`` prop (height in CSS pixels). The mark
 * preserves its aspect ratio. ``withWordmark`` controls whether the
 * "CiRA Quantum" wordmark renders to the right of the mark.
 */
const props = withDefaults(
  defineProps<{
    size?: number          // height in CSS pixels
    withWordmark?: boolean
    monochrome?: boolean   // single-color mode for tight contexts (favicon, etc.)
  }>(),
  {
    size: 48,
    withWordmark: true,
    monochrome: false,
  },
)
</script>

<template>
  <svg
    v-if="withWordmark"
    :height="size"
    viewBox="0 0 400 100"
    xmlns="http://www.w3.org/2000/svg"
    role="img"
    aria-label="CiRA Quantum"
    class="cira-logo"
  >
    <!-- Mark on the left -->
    <g transform="translate(50, 50)">
      <!-- Outer Bloch-sphere ring -->
      <circle
        cx="0" cy="0" r="40"
        fill="none"
        :stroke="monochrome ? 'currentColor' : '#a855f7'"
        stroke-width="2"
        :opacity="monochrome ? 0.55 : 0.60"
      />
      <!-- Superposition basis state #1 -->
      <circle
        cx="-12" cy="0" r="22"
        :fill="monochrome ? 'currentColor' : '#7c3aed'"
        :opacity="monochrome ? 0.85 : 0.92"
      />
      <!-- Superposition basis state #2 -->
      <circle
        cx="12" cy="0" r="22"
        :fill="monochrome ? 'currentColor' : '#06b6d4'"
        :opacity="monochrome ? 0.85 : 0.92"
      />
      <!-- Qubit core (where the two basis states overlap) -->
      <circle cx="0" cy="0" r="5" fill="#ffffff" />
    </g>
    <!-- Wordmark -->
    <text
      x="110" y="58"
      font-family="Inter, system-ui, sans-serif"
      font-size="32"
      font-weight="700"
      :fill="monochrome ? 'currentColor' : '#e7e5e4'"
      letter-spacing="-0.5"
    >CiRA Quantum</text>
  </svg>

  <svg
    v-else
    :height="size"
    :width="size"
    viewBox="0 0 100 100"
    xmlns="http://www.w3.org/2000/svg"
    role="img"
    aria-label="CiRA Quantum"
    class="cira-logo"
  >
    <g transform="translate(50, 50)">
      <circle
        cx="0" cy="0" r="40"
        fill="none"
        :stroke="monochrome ? 'currentColor' : '#a855f7'"
        stroke-width="2"
        :opacity="monochrome ? 0.55 : 0.60"
      />
      <circle
        cx="-12" cy="0" r="22"
        :fill="monochrome ? 'currentColor' : '#7c3aed'"
        :opacity="monochrome ? 0.85 : 0.92"
      />
      <circle
        cx="12" cy="0" r="22"
        :fill="monochrome ? 'currentColor' : '#06b6d4'"
        :opacity="monochrome ? 0.85 : 0.92"
      />
      <circle cx="0" cy="0" r="5" fill="#ffffff" />
    </g>
  </svg>
</template>

<style scoped>
.cira-logo {
  user-select: none;
}
</style>
