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

// ----------- Panel 2: trained circuit -----------------------------------------

const circuitColumns = computed(() => {
  const layer = props.extras.layer ?? 0
  // Each "layer" is (problem-unitary, mixer-unitary). Plus the leading
  // Hadamard column and the trailing measurement column.
  const cols: { kind: string; label: string; sublabel?: string }[] = [
    { kind: 'H', label: 'H', sublabel: 'superposition' },
  ]
  for (let p = 0; p < layer; p++) {
    cols.push({
      kind: 'P',
      label: `P(γ${sub(p)})`,
      sublabel: `γ${sub(p)}=${(props.extras.trained_gammas[p] ?? 0).toFixed(2)}`,
    })
    cols.push({
      kind: 'M',
      label: `M(β${sub(p)})`,
      sublabel: `β${sub(p)}=${(props.extras.trained_betas[p] ?? 0).toFixed(2)}`,
    })
  }
  cols.push({ kind: 'meas', label: 'measure', sublabel: 'collapse' })
  return cols
})

function sub(i: number): string {
  // Unicode subscripts for tidy γ₀, β₀ labels.
  const s = '₀₁₂₃₄₅₆₇₈₉'
  return String(i).split('').map((d) => s[parseInt(d, 10)]).join('')
}

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

    <!-- ============ Panel 2: trained circuit ============ -->
    <section class="explainer-section mt-4">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-vector-line" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">
          2 · The trained circuit ({{ extras.layer ?? '?' }} QAOA layer{{ extras.layer === 1 ? '' : 's' }})
        </h4>
      </header>
      <div class="circuit-scroll">
        <div class="circuit">
          <div class="circuit-rail-stack">
            <div
              v-for="q in extras.num_qubits"
              :key="q"
              class="circuit-rail"
            >
              <span class="rail-label">q{{ q - 1 }}</span>
              <div class="rail-line"></div>
            </div>
          </div>
          <div class="circuit-cols">
            <div
              v-for="(col, ci) in circuitColumns"
              :key="ci"
              class="circuit-col"
              :class="`col-${col.kind}`"
            >
              <div class="col-block">
                {{ col.label }}
              </div>
              <div class="col-sub">{{ col.sublabel }}</div>
            </div>
          </div>
        </div>
      </div>
      <div class="text-caption text-medium-emphasis mt-2">
        Hadamards put all qubits into uniform superposition. Each layer
        applies the <em>problem unitary</em> (phases each basis state by
        the polynomial's value, scaled by γ) then the <em>mixer</em>
        (rotates amplitude around X, scaled by β). The (γ, β) values
        above were trained locally on a statevector simulator so the
        QPU submission is one-shot — no expensive variational round-trips.
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
  padding-bottom: 0.4rem;
}
.circuit {
  position: relative;
  display: flex;
  gap: 0.5rem;
  align-items: stretch;
  min-height: 80px;
  padding: 0.6rem 0.4rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
}
.circuit-rail-stack {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  padding: 0.4rem 0;
}
.circuit-rail {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  height: 28px;
  position: relative;
}
.rail-label {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.55);
  width: 30px;
  text-align: right;
}
.rail-line {
  position: absolute;
  left: 36px;
  right: -3000px;  /* extend through all columns; clipped by container */
  height: 1px;
  background: rgba(255, 255, 255, 0.18);
}
.circuit-cols {
  display: flex;
  gap: 0.4rem;
  flex: 1;
  align-items: center;
  position: relative;
  z-index: 1;
}
.circuit-col {
  text-align: center;
  min-width: 70px;
}
.col-block {
  background: rgb(var(--v-theme-primary));
  color: white;
  padding: 0.4rem 0.5rem;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.82rem;
}
.col-H .col-block { background: rgb(var(--v-theme-success)); }
.col-P .col-block { background: rgb(var(--v-theme-primary)); }
.col-M .col-block { background: rgb(var(--v-theme-info)); }
.col-meas .col-block {
  background: transparent;
  color: rgba(255, 255, 255, 0.65);
  border: 1px dashed rgba(255, 255, 255, 0.35);
}
.col-sub {
  margin-top: 0.25rem;
  font-size: 0.7rem;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  color: rgba(255, 255, 255, 0.55);
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
