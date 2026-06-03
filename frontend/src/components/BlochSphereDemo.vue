<script setup lang="ts">
/**
 * BlochSphereDemo — interactive 2D projection of a single qubit's
 * state under an RX rotation.
 *
 * Pedagogical contract: the slider IS the input feature ``x``. As the
 * student drags it, the state vector tilts away from |0⟩ toward |1⟩,
 * and the readout ⟨Z⟩ ∈ [−1, 1] updates in real time. The "sigmoid
 * gauge" at the bottom shows how that real number becomes a class
 * probability — the same mapping the VQC uses after measurement.
 */
import { computed, ref } from 'vue'

const x = ref(0)            // input feature value, range matches sliders
const bias = ref(0)         // trainable bias b (added after ⟨Z⟩)

// AngleEmbedding (rotation='X') maps x → RX(x), so the state vector
// becomes |ψ⟩ = cos(x/2)|0⟩ − i·sin(x/2)|1⟩.
// On the Bloch sphere this is a rotation by x around the X axis: from
// |0⟩ (top, θ=0) toward |1⟩ (bottom, θ=π). At x=π the state is |1⟩;
// at x=2π it's back to |0⟩ (up to a global phase).
const theta = computed(() => x.value)  // angle from +Z

// Bloch coordinates after RX(x) applied to |0⟩:
//   |ψ⟩ = cos(x/2)|0⟩ − i sin(x/2)|1⟩
//   ⟨X⟩ = 0,  ⟨Y⟩ = −sin(x),  ⟨Z⟩ = cos(x)
const blochY = computed(() => -Math.sin(theta.value))
const blochZ = computed(() => Math.cos(theta.value))
const expectZ = computed(() => blochZ.value)
const prob1 = computed(
  () => 1 / (1 + Math.exp(-(expectZ.value + bias.value))),
)

// SVG layout
const W = 360
const H = 280
const cx = W / 2
const cy = H / 2 + 8
const R = 90

// 2D projection: the Bloch sphere lies in the XZ plane on the screen,
// with Y being the "into the page" axis. We approximate depth with a
// slight perspective: blochY shrinks the apparent X coordinate by
// (1 - 0.2·|y|). The arrow only moves in the YZ plane for this demo
// (X stays 0), so it stays on the screen circle.
function svgPoint(by: number, bz: number) {
  return {
    x: cx + by * R,
    y: cy - bz * R,
  }
}

const arrow = computed(() => svgPoint(blochY.value, blochZ.value))

function fmt(n: number) {
  return n.toFixed(3)
}
</script>

<template>
  <v-card variant="outlined" class="pa-4">
    <div class="d-flex align-center mb-2">
      <v-icon icon="mdi-cursor-default-click-outline" class="mr-2" />
      <div class="text-subtitle-1 flex-grow-1">
        Try it: one qubit, one RX rotation
      </div>
    </div>
    <div class="text-body-2 text-medium-emphasis mb-3">
      This is the exact gate the encoding column applies to qubit 0
      when the input feature is <code>x</code>. Drag the slider to see
      the qubit's state rotate on the Bloch sphere — and watch the
      measurement <code>⟨Z⟩</code> and the final class probability
      change in lockstep.
    </div>

    <v-row>
      <v-col cols="12" md="6">
        <svg :viewBox="`0 0 ${W} ${H}`" class="bloch-svg">
          <!-- Sphere outline -->
          <circle
            :cx="cx" :cy="cy" :r="R"
            fill="rgba(255,255,255,0.04)"
            stroke="currentColor" stroke-opacity="0.4"
          />
          <!-- Equator (foreshortened ellipse, just for visual sphere-ness) -->
          <ellipse
            :cx="cx" :cy="cy" :rx="R" :ry="R * 0.22"
            fill="none"
            stroke="currentColor" stroke-opacity="0.18"
            stroke-dasharray="2 3"
          />
          <!-- X axis (we don't move along X in this demo; greyed out) -->
          <line
            :x1="cx - R - 12" :y1="cy"
            :x2="cx + R + 12" :y2="cy"
            stroke="currentColor" stroke-opacity="0.25"
          />
          <text :x="cx + R + 16" :y="cy + 4" font-size="11" fill="currentColor" opacity="0.55">X</text>

          <!-- Z axis (vertical) -->
          <line
            :x1="cx" :y1="cy - R - 12"
            :x2="cx" :y2="cy + R + 12"
            stroke="currentColor" stroke-opacity="0.5"
          />
          <text :x="cx + 6" :y="cy - R - 8" font-size="11" fill="currentColor" opacity="0.75">Z</text>
          <text :x="cx - 22" :y="cy - R - 8" font-size="11" fill="#22c55e" font-weight="600">|0⟩</text>
          <text :x="cx - 22" :y="cy + R + 18" font-size="11" fill="#ef4444" font-weight="600">|1⟩</text>

          <!-- State vector arrow -->
          <line
            :x1="cx" :y1="cy"
            :x2="arrow.x" :y2="arrow.y"
            stroke="#a855f7"
            stroke-width="3"
            stroke-linecap="round"
          />
          <circle
            :cx="arrow.x" :cy="arrow.y"
            r="5"
            fill="#a855f7"
            stroke="white" stroke-width="1.5"
          />

          <!-- Caption -->
          <text
            :x="cx" :y="H - 8"
            text-anchor="middle"
            font-size="10.5"
            fill="currentColor" opacity="0.7"
          >|ψ⟩ = cos(x/2)|0⟩ − i sin(x/2)|1⟩</text>
        </svg>
      </v-col>

      <v-col cols="12" md="6" class="d-flex flex-column">
        <div class="mb-3">
          <div class="d-flex align-center mb-1">
            <span class="text-body-2 mr-2">Input feature</span>
            <code>x =</code>
            <code class="ml-2">{{ fmt(x) }}</code>
            <v-spacer />
            <span class="text-caption text-medium-emphasis">RX(x) gate</span>
          </div>
          <v-slider
            v-model="x"
            :min="-Math.PI"
            :max="Math.PI"
            :step="0.01"
            color="primary"
            hide-details
            density="compact"
          />
          <div class="d-flex justify-space-between text-caption text-medium-emphasis">
            <span>−π</span>
            <span>0</span>
            <span>+π</span>
          </div>
        </div>

        <div class="mb-3">
          <div class="d-flex align-center mb-1">
            <span class="text-body-2 mr-2">Trainable bias</span>
            <code>b =</code>
            <code class="ml-2">{{ fmt(bias) }}</code>
            <v-spacer />
            <span class="text-caption text-medium-emphasis">added after measurement</span>
          </div>
          <v-slider
            v-model="bias"
            :min="-2"
            :max="2"
            :step="0.01"
            color="accent"
            hide-details
            density="compact"
          />
        </div>

        <v-divider class="my-2" />

        <div class="readout">
          <div class="r-row">
            <span class="r-label">⟨Z⟩ (measurement)</span>
            <span class="r-bar">
              <span
                class="r-fill r-z"
                :style="{ width: `${((expectZ + 1) / 2) * 100}%` }"
              />
            </span>
            <span class="r-value">{{ fmt(expectZ) }}</span>
          </div>
          <div class="r-row">
            <span class="r-label">σ(⟨Z⟩ + b) (P class 1)</span>
            <span class="r-bar">
              <span
                class="r-fill r-p"
                :style="{ width: `${prob1 * 100}%` }"
              />
            </span>
            <span class="r-value">{{ fmt(prob1) }}</span>
          </div>
          <div class="text-caption text-medium-emphasis mt-2">
            Threshold the probability at <code>0.5</code> to pick a class.
            At <code>x = 0</code> the state is <code>|0⟩</code>, so
            <code>⟨Z⟩ = +1</code> and the model is most confident about
            class 0. At <code>x = π</code> it flips: state
            <code>|1⟩</code>, <code>⟨Z⟩ = −1</code>, confident about
            class 1.
          </div>
        </div>
      </v-col>
    </v-row>
  </v-card>
</template>

<style scoped>
.bloch-svg {
  width: 100%;
  height: auto;
  display: block;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
}
.readout {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.r-row {
  display: grid;
  grid-template-columns: 1.2fr 2fr 0.6fr;
  align-items: center;
  gap: 8px;
  font-size: 0.85rem;
}
.r-label {
  color: rgba(255, 255, 255, 0.7);
}
.r-bar {
  position: relative;
  height: 14px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 3px;
  overflow: hidden;
}
.r-fill {
  display: block;
  height: 100%;
  transition: width 0.05s linear;
}
.r-fill.r-z {
  background: linear-gradient(90deg, #ef4444 0%, #f5f5f4 50%, #22c55e 100%);
}
.r-fill.r-p {
  background: linear-gradient(90deg, #3b82f6 0%, #f59e0b 100%);
}
.r-value {
  text-align: right;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
</style>
