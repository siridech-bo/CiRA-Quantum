<script setup lang="ts">
import { useRouter } from 'vue-router'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA qLDPC primer app bar">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— qLDPC primer</span>
    </div>
    <v-spacer />
    <v-btn variant="text" prepend-icon="mdi-format-list-bulleted" @click="router.push('/qldpc')">
      Code families
    </v-btn>
    <v-btn variant="text" @click="router.push('/')">Home</v-btn>
  </v-app-bar>

  <v-main>
    <v-container style="max-width: 880px">

      <!-- Hero -->
      <div class="pt-8 pb-2">
        <div class="text-overline text-medium-emphasis">Five-step primer</div>
        <div class="text-h4 font-weight-bold mb-2">
          Quantum LDPC codes from first principles
        </div>
        <div class="text-body-1 text-medium-emphasis">
          A working researcher's tour of stabilizer error correction:
          how the math becomes a chip layout becomes a microwave pulse.
        </div>
      </div>

      <!-- Section 1 — Stabilizer codes -->
      <v-card variant="outlined" class="pa-5 my-4">
        <div class="text-overline text-medium-emphasis">Step 1</div>
        <div class="text-h6 mb-2">What is a stabilizer code?</div>
        <p class="text-body-1 mb-2">
          A <strong>stabilizer code</strong> protects logical quantum
          information by entangling many physical qubits into a state
          that is the +1 eigenvector of a chosen set of mutually-commuting
          Pauli operators called <em>stabilizers</em>. Errors flip the
          eigenvalue of one or more stabilizers — measuring them gives
          you the <strong>syndrome</strong>, which a classical decoder
          translates back into a Pauli correction.
        </p>
        <p class="text-body-2 text-medium-emphasis">
          The point of error correction isn't to prevent errors — it's
          to detect them so quickly and locally that you can undo them
          before they accumulate into a logical fault.
        </p>
      </v-card>

      <!-- Section 2 — CSS construction -->
      <v-card variant="outlined" class="pa-5 my-4">
        <div class="text-overline text-medium-emphasis">Step 2</div>
        <div class="text-h6 mb-2">
          The CSS construction: two classical codes, glued
        </div>
        <p class="text-body-1 mb-2">
          Calderbank–Shor–Steane (CSS) codes split the stabilizer set
          into two pieces: an <strong>X-stabilizer</strong> matrix
          <code>H_X</code> that detects Z-errors (bit flips) and a
          <strong>Z-stabilizer</strong> matrix <code>H_Z</code> that
          detects X-errors (phase flips). The two pieces must commute,
          which over <em>GF(2)</em> reduces to the single algebraic
          condition:
        </p>
        <div class="text-h6 text-center my-3">
          H<sub>X</sub> · H<sub>Z</sub><sup>T</sup> = 0 (mod 2)
        </div>
        <p class="text-body-2 text-medium-emphasis">
          Every qLDPC code in the gallery satisfies this — Sprint 1's
          <code>/code-families/&lt;id&gt;/matrix</code> endpoint will
          surface both halves so you can verify it numerically.
        </p>
      </v-card>

      <!-- Section 3 — Tanner graphs -->
      <v-card variant="outlined" class="pa-5 my-4">
        <div class="text-overline text-medium-emphasis">Step 3</div>
        <div class="text-h6 mb-2">Tanner graphs make the structure visible</div>
        <p class="text-body-1 mb-2">
          A <strong>Tanner graph</strong> is the bipartite graph you
          get when you put <em>data qubits</em> on one side (one node
          per column of <code>H_X</code> / <code>H_Z</code>) and
          <em>check operators</em> on the other (one node per row),
          drawing an edge wherever the matrix has a 1.
        </p>
        <p class="text-body-1 mb-2">
          The "LDPC" part of qLDPC means <em>low-density parity check</em>:
          every check node touches only a constant number of data nodes,
          regardless of the code size. That's what makes the decoder
          fast — and what makes Sprint 2's routing problem hard, because
          long-range edges have to fit on a physical chip without
          crossing destructively.
        </p>
        <v-alert type="info" variant="tonal" density="compact" class="mt-3">
          Sprint 2 ships an interactive Tanner-graph viewer with
          edge-length and crossing metrics. Drag node positions and
          watch the routing feasibility score update live.
        </v-alert>
      </v-card>

      <!-- Section 4 — Distance and threshold -->
      <v-card variant="outlined" class="pa-5 my-4">
        <div class="text-overline text-medium-emphasis">Step 4</div>
        <div class="text-h6 mb-2">Distance &amp; threshold — the two numbers that matter</div>
        <p class="text-body-1 mb-2">
          A code's <strong>distance</strong> <code>d</code> is the
          minimum number of physical-qubit Pauli errors needed to flip
          a logical operator — a single uncorrectable failure. Larger
          <code>d</code> means more error suppression. Computing
          <code>d</code> exactly is NP-hard in general; Sprint 1 will
          use the <code>qldpc</code> Python lib's ILP-based exact
          distance solver (with a numpy heuristic fallback).
        </p>
        <p class="text-body-1 mb-2">
          The <strong>threshold</strong> is the physical-qubit error
          rate below which logical errors start <em>decreasing</em>
          exponentially as you scale up the code. Surface codes have
          a threshold around 1%; the best qLDPC codes sit in the
          0.5–1% range depending on noise model. Sprint 3 measures
          this empirically via Monte Carlo simulation with
          <code>stim</code>.
        </p>
        <p class="text-body-2 text-medium-emphasis">
          Code parameters are usually written as
          <code>⟦n, k, d⟧</code> — physical qubits, logical qubits,
          distance. The gallery cards display these.
        </p>
      </v-card>

      <!-- Section 5 — Why qLDPC vs surface codes -->
      <v-card variant="outlined" class="pa-5 my-4">
        <div class="text-overline text-medium-emphasis">Step 5</div>
        <div class="text-h6 mb-2">Why qLDPC vs surface codes?</div>
        <p class="text-body-1 mb-2">
          Surface codes encode <strong>one</strong> logical qubit per
          patch — protecting 1000 logical qubits costs roughly 1
          million physical qubits at <code>d = 30</code>. qLDPC codes
          with finite rate (Bicycle, Hypergraph product) encode
          <em>many</em> logical qubits per patch, which can reduce
          this overhead by 10× or more.
        </p>
        <p class="text-body-1 mb-2">
          The price: qLDPC stabilizers are <em>not</em> geometrically
          local. The check operators talk to data qubits across the
          chip, which means real hardware needs long-range
          connectivity — coaxial lines, photonic couplers, or
          time-multiplexed routing. This is exactly the bottleneck
          Sprint 2 of this module is built to analyze.
        </p>
        <v-alert type="success" variant="tonal" class="mt-3">
          <div class="text-subtitle-1 font-weight-medium">Ready to explore?</div>
          <div class="text-body-2 mb-2">
            Open the code-family gallery to compare Bicycle,
            Surface, Hypergraph product, and Toric codes side by side.
          </div>
          <v-btn
            color="primary"
            variant="flat"
            prepend-icon="mdi-arrow-right"
            @click="router.push('/qldpc')"
          >
            Open code-family gallery
          </v-btn>
        </v-alert>
      </v-card>

      <!-- References -->
      <v-card variant="tonal" color="surface-variant" class="pa-5 my-6">
        <div class="text-subtitle-1 font-weight-medium mb-2">Further reading</div>
        <ul class="text-body-2">
          <li>Gottesman, "An introduction to quantum error correction
            and fault-tolerant quantum computation" (2009)</li>
          <li>Breuckmann &amp; Eberhardt, "Quantum low-density
            parity-check codes" (PRX Quantum, 2021)</li>
          <li>Tillich &amp; Zémor, "Quantum LDPC codes with positive
            rate and minimum distance proportional to √n" (2014)</li>
          <li>MacKay, Mitchison &amp; McFadden, "Sparse-graph codes
            for quantum error-correction" (2004)</li>
        </ul>
      </v-card>

    </v-container>
  </v-main>
</template>

<style scoped>
.logo-link {
  cursor: pointer;
  transition: opacity 0.15s ease-in-out;
}
.logo-link:hover {
  opacity: 0.8;
}
.logo-link:focus-visible {
  outline: 2px solid currentColor;
  outline-offset: 4px;
  border-radius: 4px;
}
</style>
