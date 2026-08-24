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
import { useRouter } from 'vue-router'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()

// Default entry is the NMR substrate; the sim's own top bar switches to the
// Jaynes-Cummings and ion-trap pages.
const SIM_SRC = '/sim/index.html'
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
    <v-chip size="small" color="info" variant="tonal" prepend-icon="mdi-school" class="mr-2">
      Learn
    </v-chip>
  </v-app-bar>

  <v-main>
    <!-- Full-bleed frame: the sim carries its own three-column shell, top bar,
         and substrate switcher, so we hand it the whole viewport. WebGL needs
         allow-scripts; same-origin lets pdf.js + module fetches resolve. -->
    <iframe
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
