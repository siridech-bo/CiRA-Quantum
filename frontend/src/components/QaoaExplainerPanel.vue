<script setup lang="ts">
/**
 * QaoaExplainerPanel — four-panel educational view that opens when a
 * user expands a QAOA solver row in the multi-solver comparison table.
 *
 * Phase 10A. Panels:
 *   1. Polynomial (the BQM the solver is actually optimizing)
 *   2. Trained circuit (Hadamard prep + p layers + measurement)
 *   3. Measurement histogram (probability vs bitstring, colored by energy)
 *   4. Top-K classical filter (why "most-probable" ≠ "best")
 *
 * Drives entirely from data on the Job: `cqm_json` for panel 1,
 * `qaoa_extras` for panels 2-4. No backend roundtrips.
 */
import { computed } from 'vue'
import type { Job, QaoaExtras } from '@/stores/solve'

const props = defineProps<{
  job: Job
  extras: QaoaExtras
  /** Color used for tier accents (matches the row's tier_color). */
  tierColor: string
  /** "minimize" / "maximize" — drives best-energy direction in panel 4. */
  sense: 'minimize' | 'maximize'
}>()

// ----------- Panel 1: the polynomial ------------------------------------------

interface ObjectiveTerm {
  coeff: number
  vars: string[]  // 1-var → linear; 2-var → quadratic
}

const objectiveTerms = computed<ObjectiveTerm[]>(() => {
  const cqm = props.job.cqm_json
  if (!cqm || !cqm.objective) return []
  const linear = cqm.objective.linear || {}
  const quadratic = cqm.objective.quadratic || {}
  const terms: ObjectiveTerm[] = []
  for (const [v, c] of Object.entries(linear)) {
    const n = Number(c)
    if (n === 0) continue
    terms.push({ coeff: n, vars: [v] })
  }
  for (const [k, c] of Object.entries(quadratic)) {
    const n = Number(c)
    if (n === 0) continue
    // dimod uses "var1*var2" or ["var1", "var2"] depending on format.
    // The compiler emits the latter as nested arrays, but the JSON
    // serializer flattens — handle both shapes defensively.
    let vs: string[]
    if (Array.isArray(k)) {
      vs = (k as string[]).slice(0, 2)
    } else if (typeof k === 'string' && k.includes(',')) {
      vs = k.split(',').map((s) => s.trim())
    } else {
      vs = [String(k)]
    }
    terms.push({ coeff: n, vars: vs })
  }
  return terms
})

const constraintCount = computed(() => {
  const cqm = props.job.cqm_json
  return cqm?.constraints?.length ?? 0
})

function fmtCoeff(n: number, leading = false): string {
  if (n === 1) return leading ? '' : '+ '
  if (n === -1) return leading ? '−' : '− '
  if (n < 0) return leading ? `−${Math.abs(n)}·` : `− ${Math.abs(n)}·`
  return leading ? `${n}·` : `+ ${n}·`
}

// ----------- Panel 2: trained circuit (gate-level SVG) -----------------------

/**
 * Build a gate-level circuit description from the QAOA extras. Each
 * column represents one "moment" in time. Gates within a column are
 * applied simultaneously (in real hardware they may be transpiled
 * differently, but logically they commute by qubit-disjointness).
 *
 * For a p-layer QAOA the structure is:
 *   col 0: H on every qubit (uniform superposition prep)
 *   per layer:
 *     for each (i, j) coupling: CNOT(i, j), RZ(j, 2*γ*Jij), CNOT(i, j)
 *     for each i linear bias: RZ(i, 2*γ*hi)
 *     for each qubit: RX(2*β) (mixer)
 *   final: measure on every qubit
 */
interface Gate {
  kind: 'H' | 'RZ' | 'RX' | 'CX_ctrl' | 'CX_target' | 'measure'
  qubit: number
  label?: string
  /** For CX pairs, the partner qubit index — used to draw the connector. */
  partner?: number
}
type Column = Gate[]

function sub(i: number): string {
  const s = '₀₁₂₃₄₅₆₇₈₉'
  return String(i).split('').map((d) => s[parseInt(d, 10)]).join('')
}

const circuitColumns = computed<Column[]>(() => {
  const n = props.extras.num_qubits
  const p = props.extras.layer ?? 0
  const gammas = props.extras.trained_gammas
  const betas = props.extras.trained_betas
  const cols: Column[] = []

  // --- Initial Hadamard layer ---------------------------------------
  cols.push(
    Array.from({ length: n }, (_, q) => ({ kind: 'H' as const, qubit: q })),
  )

  // --- p QAOA layers ------------------------------------------------
  for (let layer = 0; layer < p; layer++) {
    const g = gammas[layer] ?? 0
    const b = betas[layer] ?? 0

    // Problem unitary: for each ZZ coupling, CNOT — RZ — CNOT
    for (const [qi, qj, coupling] of props.extras.quadratic_terms) {
      cols.push([
        { kind: 'CX_ctrl', qubit: qi, partner: qj },
        { kind: 'CX_target', qubit: qj, partner: qi },
      ])
      cols.push([
        { kind: 'RZ', qubit: qj, label: `RZ(${(2 * g * coupling).toFixed(2)})` },
      ])
      cols.push([
        { kind: 'CX_ctrl', qubit: qi, partner: qj },
        { kind: 'CX_target', qubit: qj, partner: qi },
      ])
    }
    // Local Z biases: each h_i → RZ(2*γ*h_i)
    const localRzGates: Gate[] = props.extras.linear_terms.map(
      ([qi, hi]) => ({
        kind: 'RZ' as const,
        qubit: qi,
        label: `RZ(${(2 * g * hi).toFixed(2)})`,
      }),
    )
    if (localRzGates.length) {
      cols.push(localRzGates)
    }

    // Mixer: RX(2β) on every qubit
    cols.push(
      Array.from({ length: n }, (_, q) => ({
        kind: 'RX' as const,
        qubit: q,
        label: `RX(${(2 * b).toFixed(2)})`,
      })),
    )
  }

  // --- Measurement --------------------------------------------------
  cols.push(
    Array.from({ length: n }, (_, q) => ({ kind: 'measure' as const, qubit: q })),
  )

  return cols
})

// SVG layout constants
const RAIL_HEIGHT = 36
const COL_WIDTH = 50
const COL_PAD_LEFT = 70  // room for "q[0] |0⟩" labels
const COL_PAD_TOP = 20   // room for column numbers
const COL_PAD_BOTTOM = 10
const COL_PAD_RIGHT = 12

const circuitSvgWidth = computed(
  () => COL_PAD_LEFT + circuitColumns.value.length * COL_WIDTH + COL_PAD_RIGHT,
)
const circuitSvgHeight = computed(
  () => COL_PAD_TOP + props.extras.num_qubits * RAIL_HEIGHT + COL_PAD_BOTTOM,
)
function railY(q: number): number {
  return COL_PAD_TOP + q * RAIL_HEIGHT + RAIL_HEIGHT / 2
}
function colX(c: number): number {
  return COL_PAD_LEFT + c * COL_WIDTH + COL_WIDTH / 2
}
function isSlackQubit(q: number): boolean {
  const logical = props.extras.num_logical_vars
  if (logical == null) return false
  return q >= logical
}

// Layer-boundary annotations (so the student can see where one
// QAOA layer ends and the next begins). Position computed from the
// column structure: H, then (per-layer: many ZZ cols + local-RZ + mixer),
// then measure.
const layerBoundaries = computed<{ x: number; label: string }[]>(() => {
  const n = props.extras.num_qubits
  const p = props.extras.layer ?? 0
  const out: { x: number; label: string }[] = []

  let colIdx = 1  // skip the H column
  for (let layer = 0; layer < p; layer++) {
    const nZZ = props.extras.quadratic_terms.length
    const nLocal = props.extras.linear_terms.length > 0 ? 1 : 0
    const layerCols = nZZ * 3 + nLocal + 1  // ZZ triplets + local-RZ + mixer
    out.push({
      x: colX(colIdx) - COL_WIDTH / 2,
      label: `layer ${layer + 1}`,
    })
    colIdx += layerCols
  }
  // Final boundary (start of measurement)
  out.push({
    x: colX(colIdx) - COL_WIDTH / 2,
    label: 'measure',
  })
  return out
})

// ----------- Panel 3: measurement histogram -----------------------------------

interface HistBar {
  bitstring: string
  prob: number
  energy: number
  is_winner: boolean
  /** Normalized 0..1 where 0 = best energy, 1 = worst — drives color. */
  energy_rank: number
}

const histogramBars = computed<HistBar[]>(() => {
  const xs = props.extras.top_bitstrings
  const ps = props.extras.top_probabilities
  const es = props.extras.top_energies
  if (!xs.length) return []

  // Determine the energy-rank used for coloring. Reverse for maximize
  // since "best" = highest there.
  const sortedEnergies = [...es].sort((a, b) => a - b)
  const eMin = sortedEnergies[0]
  const eMax = sortedEnergies[sortedEnergies.length - 1]
  const range = Math.max(1e-9, eMax - eMin)

  // Winner is the top-K member with the best energy.
  let winnerIdx = 0
  for (let i = 1; i < es.length; i++) {
    const better =
      props.sense === 'maximize' ? es[i] > es[winnerIdx] : es[i] < es[winnerIdx]
    if (better) winnerIdx = i
  }

  return xs.map((x, i) => {
    // For maximize, invert the rank so highest energy gets rank 0 (best).
    const raw = (es[i] - eMin) / range
    const rank = props.sense === 'maximize' ? 1 - raw : raw
    return {
      bitstring: x,
      prob: ps[i] ?? 0,
      energy: es[i],
      is_winner: i === winnerIdx,
      energy_rank: rank,
    }
  })
})

const maxProb = computed(() =>
  histogramBars.value.length
    ? Math.max(...histogramBars.value.map((b) => b.prob))
    : 1,
)

function energyColor(rank: number): string {
  // 0 = best (green) → 1 = worst (red). Linear interpolate in HSL.
  // Green hue 130, red hue 0. Saturation 55, lightness 50.
  const hue = 130 * (1 - rank)
  return `hsl(${hue.toFixed(0)}, 60%, 50%)`
}

// ----------- Panel 4: top-K filter --------------------------------------------

interface FilterRow {
  bitstring: string
  prob: number
  prob_rank: number
  energy: number
  energy_rank: number
  is_winner: boolean
}

const filterRows = computed<FilterRow[]>(() => {
  const bars = histogramBars.value
  if (!bars.length) return []
  // Probability rank: already sorted by descending probability in the backend.
  const byProb = [...bars].sort((a, b) => b.prob - a.prob)
  const probRank = new Map(byProb.map((b, i) => [b.bitstring, i + 1]))
  // Energy rank: ascending for minimize, descending for maximize (lower number = better).
  const byEnergy = [...bars].sort((a, b) =>
    props.sense === 'maximize' ? b.energy - a.energy : a.energy - b.energy,
  )
  const energyRank = new Map(byEnergy.map((b, i) => [b.bitstring, i + 1]))
  return bars
    .map((b) => ({
      bitstring: b.bitstring,
      prob: b.prob,
      prob_rank: probRank.get(b.bitstring) ?? 0,
      energy: b.energy,
      energy_rank: energyRank.get(b.bitstring) ?? 0,
      is_winner: b.is_winner,
    }))
    .sort((a, b) => a.prob_rank - b.prob_rank)
})

function fmtPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`
}
function fmtEnergy(e: number): string {
  if (Math.abs(e) >= 1000) return e.toExponential(2)
  return e.toFixed(3)
}
</script>

<template>
  <div class="qaoa-explainer pa-3">
    <!-- ============ Panel 1: the polynomial ============ -->
    <section class="explainer-section">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-function-variant" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">1 · The polynomial we're optimizing</h4>
      </header>
      <div class="polynomial">
        <span
          v-for="(t, i) in objectiveTerms"
          :key="i"
          class="poly-term"
        >
          {{ fmtCoeff(t.coeff, i === 0) }}{{ t.vars.join('·') }}
        </span>
        <span v-if="!objectiveTerms.length" class="text-medium-emphasis">
          (objective is constant)
        </span>
      </div>
      <div class="text-caption text-medium-emphasis mt-1">
        {{ extras.num_qubits }} binary variable{{ extras.num_qubits === 1 ? '' : 's' }}
        ·
        {{ constraintCount }} constraint{{ constraintCount === 1 ? '' : 's' }}
        · sense: <strong>{{ sense }}</strong>
        ·
        objective.energy(assignment) is what every solver, classical or
        quantum, is trying to optimize.
      </div>
    </section>

    <!-- ============ Panel 2: trained circuit (gate-level) ============ -->
    <section class="explainer-section mt-4">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-vector-line" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">
          2 · The trained circuit
          ({{ extras.num_qubits }} qubit{{ extras.num_qubits === 1 ? '' : 's' }},
          {{ extras.layer ?? '?' }} QAOA layer{{ extras.layer === 1 ? '' : 's' }})
        </h4>
      </header>
      <div
        v-if="extras.num_logical_vars != null && extras.num_qubits > extras.num_logical_vars"
        class="slack-banner mb-2"
      >
        <v-icon icon="mdi-information-outline" size="small" class="mr-1" />
        Qubits q<sub>0</sub>–q<sub>{{ extras.num_logical_vars - 1 }}</sub>
        encode your problem's logical variables; q<sub>{{ extras.num_logical_vars }}</sub>–q<sub>{{ extras.num_qubits - 1 }}</sub>
        are <strong>slack qubits</strong> dimod added when lowering the
        constraint to a BQM. They take ±1 freely; their job is to soak
        up the inequality's slack so the penalty term is exactly zero
        when the original constraint is satisfied.
      </div>
      <div class="circuit-scroll">
        <svg
          :width="circuitSvgWidth"
          :height="circuitSvgHeight"
          :viewBox="`0 0 ${circuitSvgWidth} ${circuitSvgHeight}`"
          class="qcircuit-svg"
        >
          <!-- Qubit rails + |0> labels -->
          <g v-for="q in extras.num_qubits" :key="`rail-${q - 1}`">
            <line
              :x1="COL_PAD_LEFT - 10"
              :y1="railY(q - 1)"
              :x2="circuitSvgWidth - COL_PAD_RIGHT"
              :y2="railY(q - 1)"
              stroke="rgba(255,255,255,0.22)"
              stroke-width="1"
            />
            <text
              :x="COL_PAD_LEFT - 60"
              :y="railY(q - 1) + 4"
              font-size="11"
              font-family="Cascadia Code, Consolas, monospace"
              :fill="isSlackQubit(q - 1) ? 'rgba(255,200,100,0.75)' : 'rgba(255,255,255,0.65)'"
            >q[{{ q - 1 }}]</text>
            <text
              :x="COL_PAD_LEFT - 22"
              :y="railY(q - 1) + 4"
              font-size="11"
              font-family="Cascadia Code, Consolas, monospace"
              fill="rgba(255,255,255,0.45)"
            >|0⟩</text>
          </g>
          <!-- Column numbers -->
          <g>
            <text
              v-for="(_, ci) in circuitColumns"
              :key="`colnum-${ci}`"
              :x="colX(ci)"
              :y="14"
              text-anchor="middle"
              font-size="9"
              fill="rgba(255,255,255,0.40)"
            >{{ ci + 1 }}</text>
          </g>
          <!-- Layer boundary tick marks -->
          <g v-if="layerBoundaries.length">
            <line
              v-for="(boundary, bi) in layerBoundaries"
              :key="`lb-${bi}`"
              :x1="boundary.x"
              :y1="COL_PAD_TOP - 4"
              :x2="boundary.x"
              :y2="circuitSvgHeight - COL_PAD_BOTTOM"
              stroke="rgba(124,58,237,0.18)"
              stroke-width="1"
              stroke-dasharray="2 3"
            />
            <text
              v-for="(boundary, bi) in layerBoundaries"
              :key="`lbl-${bi}`"
              :x="boundary.x + 4"
              :y="circuitSvgHeight - 1"
              font-size="9"
              fill="rgba(124,58,237,0.5)"
            >{{ boundary.label }}</text>
          </g>

          <!-- Gates, column by column -->
          <g v-for="(col, ci) in circuitColumns" :key="`col-${ci}`">
            <!-- CX connectors first so they sit underneath the gate boxes -->
            <template v-for="gate in col" :key="`conn-${ci}-${gate.qubit}`">
              <line
                v-if="gate.kind === 'CX_ctrl' && gate.partner !== undefined"
                :x1="colX(ci)"
                :y1="railY(gate.qubit)"
                :x2="colX(ci)"
                :y2="railY(gate.partner)"
                stroke="#7c3aed"
                stroke-width="2"
              />
            </template>
            <!-- Gate symbols -->
            <template v-for="gate in col" :key="`g-${ci}-${gate.qubit}`">
              <!-- Single-qubit boxed gate (H, RZ, RX) -->
              <g v-if="['H', 'RZ', 'RX'].includes(gate.kind)">
                <rect
                  :x="colX(ci) - 18"
                  :y="railY(gate.qubit) - 12"
                  width="36"
                  height="24"
                  rx="3"
                  :fill="gate.kind === 'H' ? '#22c55e' : (gate.kind === 'RX' ? '#06b6d4' : '#a855f7')"
                  stroke="rgba(255,255,255,0.4)"
                  stroke-width="1"
                />
                <text
                  :x="colX(ci)"
                  :y="railY(gate.qubit) + 4"
                  text-anchor="middle"
                  font-size="11"
                  font-weight="600"
                  font-family="Cascadia Code, Consolas, monospace"
                  fill="white"
                >{{ gate.kind === 'H' ? 'H' : gate.kind }}</text>
                <title v-if="gate.label">{{ gate.label }}</title>
              </g>
              <!-- CX control: filled dot -->
              <circle
                v-else-if="gate.kind === 'CX_ctrl'"
                :cx="colX(ci)"
                :cy="railY(gate.qubit)"
                r="4"
                fill="#7c3aed"
              />
              <!-- CX target: plus-in-circle (XOR symbol) -->
              <g v-else-if="gate.kind === 'CX_target'">
                <circle
                  :cx="colX(ci)"
                  :cy="railY(gate.qubit)"
                  r="10"
                  fill="rgb(var(--v-theme-surface))"
                  stroke="#7c3aed"
                  stroke-width="2"
                />
                <line
                  :x1="colX(ci) - 7"
                  :y1="railY(gate.qubit)"
                  :x2="colX(ci) + 7"
                  :y2="railY(gate.qubit)"
                  stroke="#7c3aed"
                  stroke-width="2"
                />
                <line
                  :x1="colX(ci)"
                  :y1="railY(gate.qubit) - 7"
                  :x2="colX(ci)"
                  :y2="railY(gate.qubit) + 7"
                  stroke="#7c3aed"
                  stroke-width="2"
                />
              </g>
              <!-- Measurement symbol -->
              <g v-else-if="gate.kind === 'measure'">
                <rect
                  :x="colX(ci) - 14"
                  :y="railY(gate.qubit) - 12"
                  width="28"
                  height="24"
                  rx="3"
                  fill="rgba(255,255,255,0.06)"
                  stroke="rgba(255,255,255,0.45)"
                  stroke-width="1.5"
                  stroke-dasharray="3 2"
                />
                <path
                  :d="`M ${colX(ci) - 8} ${railY(gate.qubit) + 4} A 8 8 0 0 1 ${colX(ci) + 8} ${railY(gate.qubit) + 4}`"
                  fill="none"
                  stroke="rgba(255,255,255,0.7)"
                  stroke-width="1.5"
                />
                <line
                  :x1="colX(ci)"
                  :y1="railY(gate.qubit) + 4"
                  :x2="colX(ci) + 5"
                  :y2="railY(gate.qubit) - 5"
                  stroke="rgba(255,255,255,0.7)"
                  stroke-width="1.5"
                />
              </g>
            </template>
          </g>
        </svg>
      </div>
      <div class="text-caption text-medium-emphasis mt-2">
        Color key: <span style="color:#22c55e">H</span> (Hadamard, prep
        superposition) ·
        <span style="color:#a855f7">RZ</span> (Z-axis phase, scaled by γ ·
        coupling) ·
        <span style="color:#06b6d4">RX</span> (X-axis mixer, scaled by β) ·
        <span style="color:#7c3aed">●–⊕</span> (CNOT, entangles a pair of qubits).
        The Hadamards put all qubits into uniform superposition. Each
        QAOA layer phases the state by the polynomial (via CNOT–RZ–CNOT
        decompositions, one per ZZ coupling) and then mixes amplitude
        around the X axis. The (γ, β) values were trained locally on a
        statevector simulator so the QPU submission is one-shot.
      </div>
    </section>

    <!-- ============ Panel 3: measurement histogram ============ -->
    <section class="explainer-section mt-4">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-poll" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">
          3 · Measurement histogram (top {{ histogramBars.length }} by probability)
        </h4>
      </header>
      <div v-if="!histogramBars.length" class="text-medium-emphasis">
        No probability distribution recorded.
      </div>
      <div v-else class="histogram">
        <div
          v-for="b in histogramBars"
          :key="b.bitstring"
          class="hist-row"
          :class="{ 'hist-row--winner': b.is_winner }"
          :title="`${b.bitstring} · probability ${fmtPct(b.prob)} · energy ${fmtEnergy(b.energy)}`"
        >
          <span class="hist-bitstring">
            <v-icon
              v-if="b.is_winner"
              icon="mdi-trophy"
              size="x-small"
              color="success"
              class="mr-1"
            />
            <code>{{ b.bitstring }}</code>
          </span>
          <div class="hist-bar-track">
            <div
              class="hist-bar"
              :style="{
                width: `${(b.prob / maxProb) * 100}%`,
                background: energyColor(b.energy_rank),
              }"
            />
            <span class="hist-label">
              {{ fmtPct(b.prob) }}
              <span class="text-caption text-medium-emphasis ml-2">
                E = {{ fmtEnergy(b.energy) }}
              </span>
            </span>
          </div>
        </div>
      </div>
      <div class="text-caption text-medium-emphasis mt-2">
        Bar widths are probability; bar colors are <strong>classical
        energy</strong> (green = best, red = worst). Notice the bar
        with the trophy is the row with the best energy — which may not
        be the tallest bar. That's the punchline of panel 4.
      </div>
    </section>

    <!-- ============ Panel 4: top-K filter ============ -->
    <section class="explainer-section mt-4">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-filter-variant" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">4 · The classical top-K filter (where the answer is actually picked)</h4>
      </header>
      <v-table density="compact" class="filter-table">
        <thead>
          <tr>
            <th class="text-left">Bitstring</th>
            <th class="text-right">Probability</th>
            <th class="text-right">Prob rank</th>
            <th class="text-right">Energy</th>
            <th class="text-right">Energy rank</th>
            <th class="text-center">Winner</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="r in filterRows"
            :key="r.bitstring"
            :class="{ 'filter-winner': r.is_winner }"
          >
            <td><code>{{ r.bitstring }}</code></td>
            <td class="text-right">{{ fmtPct(r.prob) }}</td>
            <td class="text-right">#{{ r.prob_rank }}</td>
            <td class="text-right">{{ fmtEnergy(r.energy) }}</td>
            <td class="text-right">#{{ r.energy_rank }}</td>
            <td class="text-center">
              <v-icon
                v-if="r.is_winner"
                icon="mdi-trophy"
                size="small"
                color="success"
              />
            </td>
          </tr>
        </tbody>
      </v-table>
      <div class="text-caption text-medium-emphasis mt-2">
        The QPU returns a probability distribution; <em>classical post-processing</em>
        picks the actual winner. We grade each top-K bitstring's
        energy by plugging it back into the polynomial from panel 1
        (deterministic, no quantum noise). The bitstring with the best
        energy wins — even if it wasn't the most-probable measurement.
        This is why a noisy QPU can still produce an exact answer.
      </div>
    </section>
  </div>
</template>

<style scoped>
.qaoa-explainer {
  background: rgba(124, 58, 237, 0.04);
  border-left: 3px solid rgb(var(--v-theme-primary));
  border-radius: 4px;
}
.explainer-section {
  font-size: 0.9rem;
}
.polynomial {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.95rem;
  padding: 0.6rem 0.8rem;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 4px;
  word-spacing: 0.2em;
}
.poly-term {
  white-space: nowrap;
}

.circuit-scroll {
  overflow-x: auto;
  padding: 0.4rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
}
.qcircuit-svg {
  display: block;
  min-width: max-content;
}
.slack-banner {
  background: rgba(255, 200, 100, 0.06);
  border-left: 3px solid rgba(255, 200, 100, 0.5);
  padding: 0.5rem 0.75rem;
  border-radius: 3px;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.75);
}

.histogram {
  display: grid;
  gap: 0.25rem;
}
.hist-row {
  display: grid;
  grid-template-columns: 80px 1fr;
  align-items: center;
  gap: 0.5rem;
  padding: 0.15rem 0.3rem;
  border-radius: 4px;
}
.hist-row--winner {
  background: rgba(80, 200, 120, 0.08);
}
.hist-bitstring code {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
}
.hist-bar-track {
  position: relative;
  height: 22px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 3px;
  overflow: hidden;
}
.hist-bar {
  height: 100%;
  opacity: 0.85;
  transition: width 0.3s ease;
}
.hist-label {
  position: absolute;
  top: 50%;
  left: 0.5rem;
  transform: translateY(-50%);
  font-size: 0.75rem;
  font-weight: 600;
  pointer-events: none;
  text-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
}

.filter-table :deep(td),
.filter-table :deep(th) {
  padding: 4px 8px;
}
.filter-winner {
  background: rgba(76, 175, 80, 0.06);
}
</style>
