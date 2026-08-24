<script setup lang="ts">
/**
 * SimulatorPage — "Quantum System Simulation with Real Physics" at
 * /learn/simulator.
 *
 * Public (no auth). Sister to PlaygroundPage / Quantum101Page under the Learn
 * umbrella. Thin Vuetify shell around a full-height iframe; the vendored
 * CiRA QuantumSim app (public/sim/) owns the entire interaction surface inside
 * the frame — NMR spin substrate, Jaynes-Cummings reservoir, and the trapped-ion
 * lab, all switched from the sim's own top bar.
 *
 * The build is self-hosted under /public/sim/ (see public/sim/NOTICE for source
 * repo + commit provenance). Runtime deps (Three.js, math.js, pdf.js) are
 * vendored under /sim/vendor/ — no CDN at runtime. Same-origin, so we could
 * bridge to the frame later if needed.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import CiraLogo from '@/components/CiraLogo.vue'
import { useThemeToggle } from '@/composables/useThemeToggle'

const router = useRouter()
const { isDark, toggleTheme, setTheme } = useThemeToggle()

// Default entry is the NMR substrate; the sim's own top bar switches to the
// Jaynes-Cummings and ion-trap pages.
const SIM_SRC = '/sim/index.html'

// Keep the embedded sim in step with the SPA theme. The sim is same-origin
// and reads its own per-page theme from localStorage at (re)load
// ('nmr-theme' / 'jc-theme' / 'ion-theme', values 'dark' | 'light'). We
// write those keys so the frame's FIRST load already matches; this runs
// synchronously during setup, before the iframe element renders.
const frame = ref<HTMLIFrameElement | null>(null)
function syncSimTheme() {
  const t = isDark.value ? 'dark' : 'light'
  try {
    localStorage.setItem('nmr-theme', t)
    localStorage.setItem('jc-theme', t)
    localStorage.setItem('ion-theme', t)
  } catch {
    /* localStorage may be unavailable — nothing to sync */
  }
}
syncSimTheme()

// On a later toggle, re-write the keys and reload the frame's CURRENT page so
// the sim re-reads them (it doesn't watch storage). Reloading contentWindow
// (rather than resetting src) preserves whichever substrate the user is on.
watch(isDark, () => {
  syncSimTheme()
  try {
    frame.value?.contentWindow?.location.reload()
  } catch {
    if (frame.value) frame.value.src = frame.value.src // fallback: hard reset
  }
})

// Reverse sync: if the user flips the theme with the sim's OWN toggle inside
// the frame, it writes one of the sim theme keys. That fires a 'storage'
// event in this parent document (same origin), so we mirror the SPA theme —
// keeping the wrapper bar consistent with the frame. Guarded on a real change
// so it never loops with the write above.
function onStorage(e: StorageEvent) {
  if (e.key !== 'nmr-theme' && e.key !== 'jc-theme' && e.key !== 'ion-theme') return
  if (e.newValue !== 'light' && e.newValue !== 'dark') return
  const want = e.newValue === 'dark'
  if (want !== isDark.value) setTheme(e.newValue)
}
onMounted(() => window.addEventListener('storage', onStorage))
onBeforeUnmount(() => window.removeEventListener('storage', onStorage))
</script>

<template>
  <v-app-bar color="surface" flat aria-label="Simulator app bar">
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/')" />
    <div
      class="d-flex align-center logo-link"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="28" />
    </div>
    <v-app-bar-title class="text-medium-emphasis ml-2">
      <span class="font-weight-medium">Quantum System Simulation</span>
      <span class="text-medium-emphasis"> · Real Physics</span>
    </v-app-bar-title>
    <v-spacer />
    <v-btn
      :icon="isDark ? 'mdi-weather-night' : 'mdi-white-balance-sunny'"
      variant="text"
      class="mr-1"
      :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
      :aria-label="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
      @click="toggleTheme"
    />
    <v-chip size="small" color="info" variant="tonal" prepend-icon="mdi-school" class="mr-2">
      Learn
    </v-chip>
  </v-app-bar>

  <v-main>
    <!-- Full-bleed frame: the sim carries its own three-column shell, top bar,
         and substrate switcher, so we hand it the whole viewport. WebGL needs
         allow-scripts; same-origin lets pdf.js + module fetches resolve. -->
    <iframe
      ref="frame"
      :src="SIM_SRC"
      class="sim-iframe"
      title="CiRA QuantumSim — real quantum spin physics visualizer"
      sandbox="allow-scripts allow-same-origin allow-downloads allow-popups"
    />
  </v-main>
</template>

<style scoped>
.logo-link {
  cursor: pointer;
}
.sim-iframe {
  display: block;
  width: 100%;
  /* Fill the viewport below the 64px app bar. */
  height: calc(100vh - 64px);
  border: 0;
  background: #0b0e14;
}
</style>
