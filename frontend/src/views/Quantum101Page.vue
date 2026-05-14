<script setup lang="ts">
/**
 * Quantum 101 — interactive QAOA demo with (γ, β) sliders.
 *
 * Fixed problem: 3-node MaxCut on K_3 (triangle graph). The
 * probability distribution at every (γ, β) point on a 41×41 grid
 * was pre-computed offline (see backend `scripts/bake_quantum101_grid.py`)
 * and shipped as a static JSON asset, so dragging the sliders
 * updates the histogram in real time with zero backend cost.
 *
 * Pedagogy goals, in order of priority:
 *   1. Show that QAOA is *parameterized* — different (γ, β) values
 *      give very different distributions. The variational loop
 *      (which we don't run here) is searching this 2D landscape.
 *   2. Show that the histogram concentrates on the optimal cuts when
 *      (γ, β) is well-chosen — those bars go green; the bad-cut bars
 *      go red.
 *   3. Show the K_3 graph itself, colored by the most-probable
 *      bitstring, so the connection between "bitstring" and "node
 *      assignment" is concrete.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()

interface QGrid {
  problem: {
    name: string
    description: string
    n_qubits: number
    edges: [number, number][]
    bitstrings: string[]
    cut_values: number[]
    max_cut: number
  }
  gamma_grid: number[]
  beta_grid: number[]
  probabilities: number[][][]  // [g_idx][b_idx][bitstring_idx]
}

const grid = ref<QGrid | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)

// User-tunable values. We track them as continuous floats but snap to
// the nearest grid point when looking up probabilities.
const gamma = ref(0.0)
const beta = ref(0.0)

onMounted(async () => {
  try {
    const r = await fetch('/quantum101_grid.json')
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    grid.value = await r.json()
    // Default to a known-good (γ, β) for K_3 MaxCut: γ = 2π/3, β = π/8
    // is in the "optimal" basin per QAOA theory. Snap to nearest grid.
    if (grid.value) {
      gamma.value = (2 * Math.PI) / 3
      beta.value = Math.PI / 8
    }
  } catch (e: any) {
    loadError.value = e?.message || 'Failed to load Quantum 101 data'
  } finally {
    loading.value = false
  }
})

function nearestIndex(arr: number[], target: number): number {
  let best = 0
  let bestDelta = Infinity
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - target)
    if (d < bestDelta) {
      bestDelta = d
      best = i
    }
  }
  return best
}

const currentProbs = computed<number[]>(() => {
  if (!grid.value) return []
  const gi = nearestIndex(grid.value.gamma_grid, gamma.value)
  const bi = nearestIndex(grid.value.beta_grid, beta.value)
  return grid.value.probabilities[gi][bi]
})

const expectedCut = computed<number>(() => {
  if (!grid.value) return 0
  let e = 0
  const cuts = grid.value.problem.cut_values
  for (let i = 0; i < currentProbs.value.length; i++) {
    e += currentProbs.value[i] * cuts[i]
  }
  return e
})

const probOptimal = computed<number>(() => {
  if (!grid.value) return 0
  const cuts = grid.value.problem.cut_values
  const max = grid.value.problem.max_cut
  let p = 0
  for (let i = 0; i < currentProbs.value.length; i++) {
    if (cuts[i] === max) p += currentProbs.value[i]
  }
  return p
})

const mostProbableBitstring = computed<{ bs: string; p: number; cut: number } | null>(() => {
  if (!grid.value) return null
  let best = 0
  for (let i = 1; i < currentProbs.value.length; i++) {
    if (currentProbs.value[i] > currentProbs.value[best]) best = i
  }
  return {
    bs: grid.value.problem.bitstrings[best],
    p: currentProbs.value[best],
    cut: grid.value.problem.cut_values[best],
  }
})

// SVG layout for the K_3 triangle (3 nodes at the corners of an
// equilateral triangle, edges between every pair). Node coordinates
// are baked here — K_3 is so small there's no point computing them.
const triangle = {
  width: 240,
  height: 220,
  // Vertex positions for q0 (top), q1 (bottom-left), q2 (bottom-right)
  nodes: [
    { x: 120, y: 30 },
    { x: 30, y: 180 },
    { x: 210, y: 180 },
  ],
}

function nodeColor(qubit: number): string {
  // Color node by the assigned bit (0 = light blue, 1 = purple) in
  // the most-probable bitstring.
  if (!mostProbableBitstring.value) return '#888'
  const bit = mostProbableBitstring.value.bs[qubit]
  return bit === '1' ? '#7c3aed' : '#e0e7ff'
}

function edgeIsCut(e: [number, number]): boolean {
  if (!mostProbableBitstring.value) return false
  const bs = mostProbableBitstring.value.bs
  return bs[e[0]] !== bs[e[1]]
}

function isOptimalCut(cut: number): boolean {
  return grid.value !== null && cut === grid.value.problem.max_cut
}

function fmtPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`
}

// Preset (γ, β) values that demonstrate different regimes.
const presets = [
  { label: 'Origin (γ=0, β=0)', g: 0.0, b: 0.0,
    note: 'No phasing yet — uniform 1/8 on every bitstring.' },
  { label: 'Optimal basin', g: (2 * Math.PI) / 3, b: Math.PI / 8,
    note: 'Probability mass concentrates on the max-cut bitstrings.' },
  { label: 'Mid-mixing', g: Math.PI / 3, b: Math.PI / 4,
    note: 'A point between origin and optimum — partial concentration.' },
  { label: 'Bad point', g: Math.PI, b: Math.PI / 4,
    note: 'Wrong γ — back to nearly uniform; the algorithm gives no benefit.' },
]

function applyPreset(p: typeof presets[number]) {
  gamma.value = p.g
  beta.value = p.b
}
</script>

<template>
  <v-app-bar color="surface" flat aria-label="Quantum 101 app bar">
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
      <span class="font-weight-medium">Quantum 101</span>
    </v-app-bar-title>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-skeleton-loader v-if="loading" type="article, article" />
      <v-alert v-else-if="loadError" type="error" variant="tonal">
        {{ loadError }}
      </v-alert>

      <template v-else-if="grid">
        <v-card class="pa-6 mb-4" variant="tonal" color="primary">
          <v-card-title class="pa-0 text-h5 mb-2">
            <v-icon icon="mdi-atom" start /> Interactive QAOA — see the parameters change the answer
          </v-card-title>
          <v-card-text class="pa-0 text-body-2">
            {{ grid.problem.description }}
            QAOA prepares a parameterized quantum state, where two
            knobs <strong>γ</strong> (problem unitary strength) and
            <strong>β</strong> (mixer strength) control how the
            probability distribution over bitstrings is shaped. The
            real algorithm <em>learns</em> these values via classical
            optimization; here you're the optimizer — drag the
            sliders and watch the histogram redistribute. The goal:
            concentrate probability on the bitstrings that represent
            the maximum cut of the triangle graph.
          </v-card-text>
        </v-card>

        <v-row>
          <!-- Left column: graph + sliders + stats -->
          <v-col cols="12" md="5">
            <v-card class="pa-4 mb-3">
              <div class="text-subtitle-2 mb-2">
                <v-icon icon="mdi-graph-outline" size="small" start />
                K<sub>3</sub> graph (3 nodes, all pairs connected)
              </div>
              <svg
                :viewBox="`0 0 ${triangle.width} ${triangle.height}`"
                :height="triangle.height"
                class="graph-svg"
                preserveAspectRatio="xMidYMid meet"
              >
                <!-- Edges -->
                <line
                  v-for="(e, i) in grid.problem.edges"
                  :key="`edge-${i}`"
                  :x1="triangle.nodes[e[0]].x"
                  :y1="triangle.nodes[e[0]].y"
                  :x2="triangle.nodes[e[1]].x"
                  :y2="triangle.nodes[e[1]].y"
                  :stroke="edgeIsCut(e) ? '#22c55e' : 'rgba(255,255,255,0.25)'"
                  :stroke-width="edgeIsCut(e) ? 3 : 2"
                  :stroke-dasharray="edgeIsCut(e) ? '0' : '4 3'"
                />
                <!-- Nodes -->
                <g v-for="(node, i) in triangle.nodes" :key="`node-${i}`">
                  <circle
                    :cx="node.x"
                    :cy="node.y"
                    r="24"
                    :fill="nodeColor(i)"
                    stroke="rgba(255,255,255,0.5)"
                    stroke-width="1.5"
                  />
                  <text
                    :x="node.x"
                    :y="node.y + 5"
                    text-anchor="middle"
                    font-size="14"
                    font-weight="600"
                    :fill="mostProbableBitstring && mostProbableBitstring.bs[i] === '1' ? '#fff' : '#1e293b'"
                  >q{{ i }}</text>
                </g>
              </svg>
              <div class="text-caption text-medium-emphasis text-center">
                Solid green edges = cut. Dashed gray edges = endpoints agree.
                Colors come from the <em>most-probable</em> bitstring at the
                current (γ, β).
              </div>
            </v-card>

            <v-card class="pa-4 mb-3">
              <div class="text-subtitle-2 mb-3">
                <v-icon icon="mdi-tune" size="small" start />
                Tune the parameters
              </div>
              <div class="slider-row">
                <label>γ (gamma)</label>
                <v-slider
                  v-model="gamma"
                  :min="0"
                  :max="Math.PI"
                  :step="Math.PI / 40"
                  hide-details
                  density="compact"
                  color="primary"
                />
                <span class="slider-val">{{ gamma.toFixed(3) }}</span>
              </div>
              <div class="slider-row">
                <label>β (beta)</label>
                <v-slider
                  v-model="beta"
                  :min="0"
                  :max="Math.PI / 2"
                  :step="Math.PI / 80"
                  hide-details
                  density="compact"
                  color="info"
                />
                <span class="slider-val">{{ beta.toFixed(3) }}</span>
              </div>

              <div class="text-caption text-medium-emphasis mb-2 mt-2">Or try a preset:</div>
              <div class="d-flex flex-wrap ga-1">
                <v-btn
                  v-for="p in presets"
                  :key="p.label"
                  size="x-small"
                  variant="outlined"
                  @click="applyPreset(p)"
                >
                  {{ p.label }}
                </v-btn>
              </div>
            </v-card>

            <v-card class="pa-4">
              <div class="text-subtitle-2 mb-3">
                <v-icon icon="mdi-trophy-variant" size="small" start />
                Live stats
              </div>
              <div class="stat-row">
                <span>Probability of optimal cut</span>
                <strong :class="probOptimal > 0.5 ? 'text-success' : 'text-warning'">
                  {{ fmtPct(probOptimal) }}
                </strong>
              </div>
              <div class="stat-row">
                <span>Expected cut value</span>
                <strong>{{ expectedCut.toFixed(3) }} / {{ grid.problem.max_cut }}</strong>
              </div>
              <div class="stat-row">
                <span>Most-probable bitstring</span>
                <strong>
                  <code>{{ mostProbableBitstring?.bs }}</code>
                  <v-chip
                    size="x-small"
                    :color="isOptimalCut(mostProbableBitstring?.cut ?? 0) ? 'success' : 'warning'"
                    variant="tonal"
                    class="ml-2"
                  >
                    cut {{ mostProbableBitstring?.cut }}/{{ grid.problem.max_cut }}
                  </v-chip>
                </strong>
              </div>
            </v-card>
          </v-col>

          <!-- Right column: histogram -->
          <v-col cols="12" md="7">
            <v-card class="pa-4 h-100">
              <div class="text-subtitle-2 mb-3">
                <v-icon icon="mdi-poll" size="small" start />
                Probability distribution at the current (γ, β)
              </div>
              <div class="histogram">
                <div
                  v-for="(prob, i) in currentProbs"
                  :key="i"
                  class="hist-row"
                  :class="{ 'hist-row--optimal': isOptimalCut(grid.problem.cut_values[i]) }"
                >
                  <span class="hist-label">
                    <code>{{ grid.problem.bitstrings[i] }}</code>
                    <v-chip
                      size="x-small"
                      :color="isOptimalCut(grid.problem.cut_values[i]) ? 'success' : 'default'"
                      variant="tonal"
                      class="ml-1"
                    >
                      cut {{ grid.problem.cut_values[i] }}
                    </v-chip>
                  </span>
                  <div class="hist-track">
                    <div
                      class="hist-fill"
                      :style="{
                        width: `${prob * 100}%`,
                        background: isOptimalCut(grid.problem.cut_values[i])
                          ? '#22c55e'
                          : '#a855f7',
                      }"
                    />
                    <span class="hist-pct">{{ fmtPct(prob) }}</span>
                  </div>
                </div>
              </div>
              <div class="text-caption text-medium-emphasis mt-3">
                Green bars are the bitstrings that achieve the maximum
                cut (2 of 3 edges). Watch how their combined probability
                changes as you move the sliders. The variational QAOA
                training step in our real solvers is automating exactly
                this search — finding (γ, β) that maximizes the green
                mass.
              </div>
            </v-card>
          </v-col>
        </v-row>

        <v-card class="pa-5 mt-4">
          <div class="text-subtitle-1 font-weight-medium mb-2">
            <v-icon icon="mdi-lightbulb-on-outline" start />
            What you just demonstrated
          </div>
          <ul class="learning-points">
            <li>
              <strong>QAOA is parameterized.</strong> The quantum
              computer's job is to prepare a state and measure it; the
              <em>parameters</em> (γ, β) are what make the resulting
              distribution useful or useless.
            </li>
            <li>
              <strong>Measurement is probabilistic.</strong> Each shot
              gives you one bitstring sampled from the distribution
              above — not a deterministic answer. Good (γ, β) means
              high probability on the optimal bitstrings.
            </li>
            <li>
              <strong>The variational loop is classical optimization
              over a quantum-prepared distribution.</strong> The real
              QAOA wraps this page's two sliders in a classical
              optimizer (SLSQP, COBYLA, …) that searches for the (γ, β)
              with the highest expected cut value. In CiRA Quantum we
              do this locally on a statevector simulator, then submit
              just the trained circuit to the QPU.
            </li>
            <li>
              <strong>Larger problems → more layers.</strong> K<sub>3</sub>
              is small enough that one layer (p = 1) can concentrate
              most probability on the optimum. For harder MaxCut
              instances, you'd add more layers — each with its own
              (γ, β) — to get sharper concentration.
            </li>
          </ul>
        </v-card>
      </template>
    </v-container>
  </v-main>
</template>

<style scoped>
.logo-link {
  cursor: pointer;
  transition: opacity 0.15s ease-in-out;
}
.logo-link:hover { opacity: 0.8; }
.logo-link:focus-visible {
  outline: 2px solid currentColor;
  outline-offset: 4px;
  border-radius: 4px;
}

.graph-svg {
  width: 100%;
  max-width: 280px;
  display: block;
  margin: 0 auto;
}

.slider-row {
  display: grid;
  grid-template-columns: 80px 1fr 60px;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}
.slider-row label {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.9rem;
}
.slider-val {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
  text-align: right;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  padding: 0.3rem 0;
  font-size: 0.9rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.stat-row:last-child { border-bottom: none; }

.histogram {
  display: grid;
  gap: 0.3rem;
}
.hist-row {
  display: grid;
  grid-template-columns: 140px 1fr;
  align-items: center;
  gap: 0.5rem;
}
.hist-row--optimal .hist-label code {
  font-weight: 700;
}
.hist-label code {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
}
.hist-track {
  position: relative;
  height: 22px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 4px;
  overflow: hidden;
}
.hist-fill {
  height: 100%;
  opacity: 0.85;
  transition: width 0.18s ease;
}
.hist-pct {
  position: absolute;
  top: 50%;
  left: 0.5rem;
  transform: translateY(-50%);
  font-size: 0.75rem;
  font-weight: 600;
  pointer-events: none;
  text-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
}

.learning-points {
  padding-left: 1.2rem;
}
.learning-points li {
  margin-bottom: 0.5rem;
  line-height: 1.5;
}
</style>
