<script setup lang="ts">
/**
 * VqcCircuitExplainer — gate-level visualization of the VQC.
 *
 * Mirrors the QaoaExplainerPanel circuit conventions so the look-and-feel
 * stays consistent across the platform: one column = one moment in time,
 * opaque gate boxes draw on top of the rail, CNOTs live in their own
 * columns, ``|0⟩`` init state on the left, column numbers across the top,
 * layer-boundary dashed lines underneath.
 */
import { computed } from 'vue'
import type { QmlCircuitInfo } from '@/stores/qml'

const props = defineProps<{
  info: QmlCircuitInfo
  featureNames?: string[]
  classes?: string[]
  /** Trained weights, shape (n_layers, n_qubits). Absent during training
   *  → we render θ placeholders instead. */
  weights?: number[][]
}>()

// ----- Column model --------------------------------------------------------

type GateKind = 'RX' | 'RY' | 'CX_ctrl' | 'CX_target' | 'measure'
interface Gate {
  kind: GateKind
  qubit: number
  label?: string
  partner?: number  // for CX pairs — used to draw the vertical connector
}
type Column = Gate[]

/** Trained RY angle for layer ``l``, qubit ``q``, or ``θ`` if untrained. */
function ryLabel(l: number, q: number): string {
  const w = props.weights?.[l]?.[q]
  if (w === undefined || w === null) return `θ${sub(l)}${sub(q)}`
  return w.toFixed(2)
}

function sub(i: number): string {
  const s = '₀₁₂₃₄₅₆₇₈₉'
  return String(i).split('').map((d) => s[parseInt(d, 10)]).join('')
}

// Build the full column list. Structure mirrors what the trainer
// actually runs:
//   col 0:                RX(x_q) on every qubit       (AngleEmbedding)
//   per layer:
//     RY(θ_{l,q}) on every qubit
//     ring of CNOTs: (0→1), (1→2), …, (n-1→0)         (one column each)
//   final:                ⟨Z⟩ measurement on qubit 0
const circuitColumns = computed<Column[]>(() => {
  const n = props.info.n_qubits
  const L = props.info.n_layers
  const fn = props.featureNames || []
  const cols: Column[] = []

  // Encoding column.
  cols.push(
    Array.from({ length: n }, (_, q) => ({
      kind: 'RX' as const,
      qubit: q,
      label: `RX(${fn[q] ? shortFeature(fn[q]) : `x${q}`})`,
    })),
  )

  // Layers.
  for (let l = 0; l < L; l++) {
    // RY column.
    cols.push(
      Array.from({ length: n }, (_, q) => ({
        kind: 'RY' as const,
        qubit: q,
        label: `RY(${ryLabel(l, q)})`,
      })),
    )
    // Ring CNOTs — one column per pair so each gets its own moment.
    if (n >= 2) {
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n
        cols.push([
          { kind: 'CX_ctrl' as const, qubit: i, partner: j },
          { kind: 'CX_target' as const, qubit: j, partner: i },
        ])
      }
    }
  }

  // Measurement (only on qubit 0).
  cols.push([{ kind: 'measure' as const, qubit: 0 }])

  return cols
})

function shortFeature(name: string, max = 8): string {
  if (name.length <= max) return name
  return name.slice(0, max - 1) + '…'
}

// ----- SVG layout ----------------------------------------------------------

const RAIL_HEIGHT = 38
const COL_WIDTH = 52
const COL_PAD_LEFT = 78  // "q[0] |0⟩" labels
const COL_PAD_TOP = 28   // column numbers + section headers
const COL_PAD_BOTTOM = 22 // layer-boundary annotations
const COL_PAD_RIGHT = 80 // measurement readout + arrow

const svgWidth = computed(
  () => COL_PAD_LEFT + circuitColumns.value.length * COL_WIDTH + COL_PAD_RIGHT,
)
const svgHeight = computed(
  () => COL_PAD_TOP + props.info.n_qubits * RAIL_HEIGHT + COL_PAD_BOTTOM,
)

function railY(q: number): number {
  return COL_PAD_TOP + q * RAIL_HEIGHT + RAIL_HEIGHT / 2
}
function colX(c: number): number {
  return COL_PAD_LEFT + c * COL_WIDTH + COL_WIDTH / 2
}

// ----- Section headers (Encoding / Layer N / Measure) ----------------------

/** Returns the [startCol, endCol) range for each named section. Used
 *  to draw the "1. Encoding", "2. Layer 1", … "3. Measure" labels above
 *  the right group of columns. */
const sectionHeaders = computed<{
  label: string
  color: string
  startCol: number
  endCol: number
}[]>(() => {
  const n = props.info.n_qubits
  const out: { label: string; color: string; startCol: number; endCol: number }[] = []
  out.push({ label: '1. Encoding', color: '#3b82f6', startCol: 0, endCol: 1 })
  let cursor = 1
  for (let l = 0; l < props.info.n_layers; l++) {
    const layerCols = 1 + (n >= 2 ? n : 0)  // 1 RY column + n CNOT columns
    out.push({
      label: `2. Layer ${l + 1}`,
      color: '#a855f7',
      startCol: cursor,
      endCol: cursor + layerCols,
    })
    cursor += layerCols
  }
  out.push({
    label: '3. Measure',
    color: '#22c55e',
    startCol: cursor,
    endCol: cursor + 1,
  })
  return out
})

// ----- Layer-boundary tick marks (dashed verticals under the rail area) ----

const layerBoundaries = computed<{ x: number; label: string }[]>(() => {
  const n = props.info.n_qubits
  const out: { x: number; label: string }[] = []
  let cursor = 1  // skip encoding column
  for (let l = 0; l < props.info.n_layers; l++) {
    const layerCols = 1 + (n >= 2 ? n : 0)
    out.push({
      x: colX(cursor) - COL_WIDTH / 2,
      label: `layer ${l + 1}`,
    })
    cursor += layerCols
  }
  out.push({
    x: colX(cursor) - COL_WIDTH / 2,
    label: 'measure',
  })
  return out
})

// ----- Backend badge (unchanged from the previous iteration) ---------------

const backendBadge = computed(() => {
  if (props.info.is_real_hardware) {
    return {
      label: 'Real QPU',
      color: 'error',
      icon: 'mdi-atom',
      tagline:
        'Submitted to a superconducting quantum processor. ' +
        `Shots: ${props.info.shots ?? '?'}. Results carry shot noise.`,
    }
  }
  if (props.info.backend_kind === 'shot_simulator') {
    return {
      label: 'Shot simulator',
      color: 'warning',
      icon: 'mdi-dice-multiple',
      tagline:
        `Classical simulator drawing ${props.info.shots} shots per circuit. ` +
        'Matches what a real QPU would look like (with shot noise) but ' +
        'runs locally and is reproducible.',
    }
  }
  return {
    label: 'Statevector simulator',
    color: 'info',
    icon: 'mdi-laptop',
    tagline:
      'Exact classical simulation of the wavefunction — no shot noise, ' +
      'no queue. This is the right backend for learning, and for ' +
      'training before you spend real-QPU shots.',
  }
})
</script>

<template>
  <v-card variant="outlined" class="pa-4">
    <!-- Backend transparency header -->
    <div class="d-flex align-center mb-2 flex-wrap">
      <v-icon icon="mdi-flask-outline" class="mr-2" />
      <div class="text-subtitle-1 flex-grow-1">
        Circuit &amp; backend
      </div>
      <v-chip
        :color="backendBadge.color"
        :prepend-icon="backendBadge.icon"
        size="small"
        variant="flat"
      >
        {{ backendBadge.label }}
      </v-chip>
    </div>
    <div class="text-body-2 text-medium-emphasis mb-3">
      <strong>{{ info.backend_name }}</strong> —
      {{ backendBadge.tagline }}
    </div>

    <!-- Quick stats -->
    <div class="d-flex flex-wrap ga-3 mb-3 stat-strip">
      <div>
        <div class="text-caption text-medium-emphasis">qubits</div>
        <div class="text-subtitle-2">{{ info.n_qubits }}</div>
      </div>
      <div>
        <div class="text-caption text-medium-emphasis">layers</div>
        <div class="text-subtitle-2">{{ info.n_layers }}</div>
      </div>
      <div>
        <div class="text-caption text-medium-emphasis">trainable params</div>
        <div class="text-subtitle-2">
          {{ info.n_trainable_params }}
          <span class="text-caption text-medium-emphasis">
            (= {{ info.n_layers }}·{{ info.n_qubits }} + 1 bias)
          </span>
        </div>
      </div>
      <div v-if="info.shots !== null">
        <div class="text-caption text-medium-emphasis">shots</div>
        <div class="text-subtitle-2">{{ info.shots }}</div>
      </div>
      <div v-else>
        <div class="text-caption text-medium-emphasis">shots</div>
        <div class="text-subtitle-2">∞ (exact)</div>
      </div>
    </div>

    <!-- Section-header strip (rendered as plain text above the circuit) -->
    <div
      v-if="featureNames && featureNames.length === info.n_qubits"
      class="circuit-banner mb-2"
    >
      <v-icon icon="mdi-information-outline" size="small" class="mr-1" />
      Each input feature is encoded as an
      <code>RX</code> rotation on its own qubit:
      <code v-for="(f, i) in featureNames" :key="f">
        q<sub>{{ i }}</sub> ← {{ f }}<span v-if="i < featureNames.length - 1">, </span>
      </code>.
      The
      <strong>{{ info.n_layers }}</strong>
      entangler layer{{ info.n_layers === 1 ? '' : 's' }} learn
      {{ info.n_layers * info.n_qubits }} trainable rotation angles +
      1 bias.
    </div>

    <!-- Circuit SVG -->
    <div class="circuit-scroll">
      <svg
        :width="svgWidth"
        :height="svgHeight"
        :viewBox="`0 0 ${svgWidth} ${svgHeight}`"
        class="vqc-svg"
      >
        <!-- Section headers across the top -->
        <g>
          <text
            v-for="(sec, si) in sectionHeaders"
            :key="`sec-${si}`"
            :x="(colX(sec.startCol) + colX(sec.endCol - 1)) / 2"
            :y="13"
            text-anchor="middle"
            font-size="10.5"
            font-weight="600"
            :fill="sec.color"
          >{{ sec.label }}</text>
        </g>

        <!-- Column numbers (tiny, under the section headers) -->
        <g>
          <text
            v-for="(_, ci) in circuitColumns"
            :key="`colnum-${ci}`"
            :x="colX(ci)"
            :y="25"
            text-anchor="middle"
            font-size="9"
            fill="rgba(255,255,255,0.40)"
          >{{ ci + 1 }}</text>
        </g>

        <!-- Layer-boundary dashed verticals + their labels at the bottom -->
        <g v-if="layerBoundaries.length">
          <line
            v-for="(b, bi) in layerBoundaries"
            :key="`lb-${bi}`"
            :x1="b.x"
            :y1="COL_PAD_TOP - 4"
            :x2="b.x"
            :y2="svgHeight - COL_PAD_BOTTOM + 4"
            stroke="rgba(168,85,247,0.18)"
            stroke-width="1"
            stroke-dasharray="2 3"
          />
          <text
            v-for="(b, bi) in layerBoundaries"
            :key="`lbl-${bi}`"
            :x="b.x + 4"
            :y="svgHeight - 4"
            font-size="9"
            fill="rgba(168,85,247,0.55)"
          >{{ b.label }}</text>
        </g>

        <!-- Qubit rails (drawn first so gate boxes can cover them) + labels -->
        <g v-for="q in info.n_qubits" :key="`rail-${q - 1}`">
          <line
            :x1="COL_PAD_LEFT - 10"
            :y1="railY(q - 1)"
            :x2="svgWidth - COL_PAD_RIGHT + 10"
            :y2="railY(q - 1)"
            stroke="rgba(255,255,255,0.22)"
            stroke-width="1"
          />
          <text
            :x="COL_PAD_LEFT - 60"
            :y="railY(q - 1) + 4"
            font-size="11"
            font-family="Cascadia Code, Consolas, monospace"
            fill="rgba(255,255,255,0.65)"
          >q[{{ q - 1 }}]</text>
          <text
            :x="COL_PAD_LEFT - 22"
            :y="railY(q - 1) + 4"
            font-size="11"
            font-family="Cascadia Code, Consolas, monospace"
            fill="rgba(255,255,255,0.45)"
          >|0⟩</text>
        </g>

        <!-- Gates, column by column. CX connectors drawn first so they
             sit under the control/target glyphs. -->
        <g v-for="(col, ci) in circuitColumns" :key="`col-${ci}`">
          <!-- Vertical CX wire (between control + target) -->
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
          <!-- Gate symbols. -->
          <template v-for="gate in col" :key="`g-${ci}-${gate.qubit}`">
            <!-- RX (encoding) — solid blue box -->
            <g v-if="gate.kind === 'RX'">
              <rect
                :x="colX(ci) - 22"
                :y="railY(gate.qubit) - 12"
                width="44"
                height="24"
                rx="3"
                fill="#1e3a8a"
                stroke="#3b82f6"
                stroke-width="1.5"
              />
              <text
                :x="colX(ci)"
                :y="railY(gate.qubit) + 4"
                text-anchor="middle"
                font-size="10"
                font-weight="600"
                font-family="Cascadia Code, Consolas, monospace"
                fill="#bfdbfe"
              >{{ gate.label }}</text>
            </g>
            <!-- RY (entangler) — solid purple box -->
            <g v-else-if="gate.kind === 'RY'">
              <rect
                :x="colX(ci) - 22"
                :y="railY(gate.qubit) - 12"
                width="44"
                height="24"
                rx="3"
                fill="#581c87"
                stroke="#a855f7"
                stroke-width="1.5"
              />
              <text
                :x="colX(ci)"
                :y="railY(gate.qubit) + 4"
                text-anchor="middle"
                font-size="10"
                font-weight="600"
                font-family="Cascadia Code, Consolas, monospace"
                fill="#e9d5ff"
              >{{ gate.label }}</text>
            </g>
            <!-- CX control: filled dot -->
            <circle
              v-else-if="gate.kind === 'CX_ctrl'"
              :cx="colX(ci)"
              :cy="railY(gate.qubit)"
              r="4"
              fill="#7c3aed"
            />
            <!-- CX target: ⊕ symbol -->
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
            <!-- Measurement glyph: meter dial -->
            <g v-else-if="gate.kind === 'measure'">
              <rect
                :x="colX(ci) - 16"
                :y="railY(gate.qubit) - 12"
                width="32"
                height="24"
                rx="3"
                fill="rgba(34,197,94,0.10)"
                stroke="#22c55e"
                stroke-width="1.5"
              />
              <path
                :d="`M ${colX(ci) - 9} ${railY(gate.qubit) + 4} A 9 9 0 0 1 ${colX(ci) + 9} ${railY(gate.qubit) + 4}`"
                fill="none"
                stroke="#22c55e"
                stroke-width="1.5"
              />
              <line
                :x1="colX(ci)"
                :y1="railY(gate.qubit) + 4"
                :x2="colX(ci) + 6"
                :y2="railY(gate.qubit) - 5"
                stroke="#22c55e"
                stroke-width="1.5"
              />
              <text
                :x="colX(ci)"
                :y="railY(gate.qubit) - 14"
                text-anchor="middle"
                font-size="9"
                fill="#22c55e"
              >⟨Z⟩</text>
              <!-- Classical wire out → σ(·+b) -->
              <line
                :x1="colX(ci) + 16"
                :y1="railY(gate.qubit)"
                :x2="svgWidth - 8"
                :y2="railY(gate.qubit)"
                stroke="#22c55e"
                stroke-width="1.5"
                stroke-dasharray="3 3"
              />
              <text
                :x="svgWidth - 4"
                :y="railY(gate.qubit) + 4"
                text-anchor="end"
                font-size="10"
                fill="#22c55e"
              >σ(·+b)</text>
            </g>
          </template>
        </g>
      </svg>
    </div>
    <div class="text-caption text-medium-emphasis mt-2">
      Color key:
      <span style="color:#3b82f6">RX</span>
      (X-axis rotation, encodes one input feature) ·
      <span style="color:#a855f7">RY</span>
      (Y-axis rotation, trainable angle) ·
      <span style="color:#7c3aed">●–⊕</span>
      (CNOT, entangles a pair of qubits) ·
      <span style="color:#22c55e">⟨Z⟩</span>
      (measurement, classical wire to <code>σ</code>).
      The CNOT ring lets each layer's rotations couple every qubit, so
      gradients flow through the entangler and the model learns feature
      interactions, not just per-feature mappings.
    </div>

    <!-- Plain-language walkthrough (expandable, same content as before) -->
    <v-expansion-panels variant="accordion" class="mt-3">
      <v-expansion-panel>
        <v-expansion-panel-title class="text-body-2">
          <v-icon icon="mdi-school" size="small" class="mr-2" />
          What's happening in this circuit?
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <ol class="explainer-list">
            <li>
              <strong>Encoding (blue).</strong> Each input feature
              <code>x<sub>i</sub></code> becomes an angle in an RX
              rotation on its own qubit. A point in the dataset thus
              becomes a specific state on the Bloch sphere — one rotation
              per feature.
              <span v-if="featureNames && featureNames.length === info.n_qubits">
                For this dataset:
                <code v-for="(f, i) in featureNames" :key="f">
                  q<sub>{{ i }}</sub> ← {{ f }}<span v-if="i < featureNames.length - 1">, </span>
                </code>.
              </span>
            </li>
            <li>
              <strong>Entangler layers (purple).</strong> Each layer
              applies a trainable
              <code>RY(θ<sub>l,q</sub>)</code> on every qubit, then a
              ring of CNOTs that lets the qubits "talk to each other."
              Stacking <strong>{{ info.n_layers }}</strong> of these
              gives the model {{ info.n_layers * info.n_qubits }}
              trainable rotation parameters.
            </li>
            <li>
              <strong>Measurement (green).</strong> We measure the
              expectation of <code>Z</code> on qubit 0. That's a number
              in <code>[−1, 1]</code>. Add a trainable bias <code>b</code>,
              squash through a sigmoid → class probability. Threshold at
              <code>0.5</code> to pick
              <code v-if="classes && classes.length === 2">
                "{{ classes[0] }}" vs "{{ classes[1] }}"
              </code><code v-else>class 0 vs class 1</code>.
            </li>
          </ol>
          <v-divider class="my-2" />
          <div class="text-body-2 text-medium-emphasis">
            Total parameters:
            <strong>{{ info.n_trainable_params }}</strong> trainable
            scalars. That's all the model has — the dataset's structure
            has to fit inside this tiny capacity. The optimizer (Adam,
            on binary cross-entropy) tunes those scalars over the
            training history above.
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>

      <v-expansion-panel>
        <v-expansion-panel-title class="text-body-2">
          <v-icon icon="mdi-information-outline" size="small" class="mr-2" />
          Why a simulator, and what would change on a real QPU?
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <p class="mb-2">
            We run the VQC on a classical simulator
            (<code>{{ info.backend_name }}</code>) that tracks the full
            wavefunction exactly. Gradients are exact too, which means
            Adam converges quickly and reproducibly.
          </p>
          <p class="mb-2">
            On a <strong>real QPU</strong> (QML-5 IBM, QML-6 Origin):
          </p>
          <ul class="ml-4 mb-2">
            <li>
              The expectation <code>⟨Z⟩</code> is estimated from a finite
              number of <strong>shots</strong>, so it has noise.
            </li>
            <li>
              Gradients are computed via <strong>parameter shift</strong>
              — two extra circuit runs per parameter per training step,
              so each epoch costs many cloud submissions.
            </li>
            <li>
              The device has its own decoherence + gate-error profile;
              circuits with many CNOTs lose fidelity fast.
            </li>
          </ul>
          <p>
            That's why we train locally first, and only push the trained
            parameters to a QPU for a final evaluation. The simulator is
            the right tool for learning the algorithm; the QPU is the
            right tool for showing what the hardware actually does.
          </p>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </v-card>
</template>

<style scoped>
.circuit-banner {
  background: rgba(168, 85, 247, 0.08);
  border-left: 3px solid #a855f7;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 0.875rem;
  line-height: 1.4;
}
.circuit-scroll {
  overflow-x: auto;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
  padding: 4px;
}
.vqc-svg {
  display: block;
  min-width: 100%;
}
.stat-strip > div {
  min-width: 80px;
}
.explainer-list {
  padding-left: 1.2rem;
}
.explainer-list li {
  margin-bottom: 0.5rem;
}
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
</style>
