<script setup lang="ts">
/**
 * PlaygroundPage — standalone Quirk-E playground at /learn/playground.
 *
 * Public (no auth). Sister to Quantum101Page and QmlLearnPage in the
 * Learn umbrella. The page is a thin Vuetify shell around the iframe;
 * Quirk-E owns the entire interaction surface inside the frame.
 *
 * Optional `?circuit=<encoded-json>` query parameter — Phase 3 will use
 * this to let learn pages link in with a preloaded circuit. For now we
 * read it but expect callers to be absent.
 */
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import CiraLogo from '@/components/CiraLogo.vue'
import QuirkPlayground, { type QuirkCircuitJson } from '@/components/QuirkPlayground.vue'

const route = useRoute()
const router = useRouter()

// Decode an inbound `?circuit=` query into a Quirk JSON shape. If the
// param is absent or malformed, fall back to an empty grid — Quirk-E's
// default landing state is already pedagogically reasonable.
const initialCircuit = computed<QuirkCircuitJson | undefined>(() => {
  const raw = route.query.circuit
  if (typeof raw !== 'string' || raw.length === 0) return undefined
  try {
    return JSON.parse(decodeURIComponent(raw)) as QuirkCircuitJson
  } catch {
    return undefined
  }
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="Playground app bar">
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
      <span class="font-weight-medium">Circuit Playground</span>
    </v-app-bar-title>
    <v-spacer />
    <v-btn
      variant="text"
      size="small"
      prepend-icon="mdi-github"
      href="https://github.com/DEQSE-Project/Quirk-E"
      target="_blank"
      rel="noopener"
    >
      Quirk-E
    </v-btn>
  </v-app-bar>

  <v-main>
    <v-container class="pt-4 pb-6">
      <!-- Framing — explain what this is and how it relates to the
           rest of the platform. Kept short; the iframe is the point. -->
      <div class="d-flex align-center mb-2 flex-wrap">
        <v-icon icon="mdi-cursor-default-click-outline" class="mr-2" />
        <div class="text-h6 flex-grow-1">Drag-drop quantum circuit playground</div>
        <v-chip size="small" color="info" variant="tonal" prepend-icon="mdi-school">
          Learn
        </v-chip>
      </div>
      <div class="text-body-2 text-medium-emphasis mb-4" style="max-width: 720px">
        Open-ended sandbox for exploring small quantum circuits — drag gates
        onto the grid, watch the state evolve in real time, and experiment
        with whatever ideas the rest of the platform sparked. This page
        embeds <a href="https://github.com/DEQSE-Project/Quirk-E" target="_blank" rel="noopener">Quirk-E</a>
        (an extended fork of Quirk by Craig Gidney), self-hosted under
        Apache-2.0. Nothing you do here touches the trainer or the
        benchmarks — it is a free-play surface.
      </div>

      <QuirkPlayground :initial-circuit="initialCircuit" :height="720" />

      <div class="text-caption text-medium-emphasis mt-3">
        Best on desktop — Quirk-E uses mouse drag for gate placement.
        Works on tablets in landscape; phones are not supported.
      </div>
    </v-container>
  </v-main>
</template>

<style scoped>
.logo-link {
  cursor: pointer;
}
</style>
