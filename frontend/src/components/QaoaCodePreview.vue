<script setup lang="ts">
/**
 * QaoaCodePreview — three-tab source view for a QAOA submission.
 *
 * Tabs:
 *   - pyqpanda3 (what qaoa_originqc actually sends to Origin's cloud)
 *   - Qiskit    (what qaoa_ibmq builds against qiskit-ibm-runtime)
 *   - BQM       (the plain dimod Hamiltonian the local qaoa_sim samples)
 *
 * All three are generated from the same ``qaoa_extras`` block the
 * orchestrator stamps onto the row — so the queued and completed views
 * show identical source, just at different points in the job lifecycle.
 *
 * The generators are intentionally verbatim-runnable: a user can copy
 * any tab, paste it into a fresh Python file, drop their API key in,
 * and reproduce the submission end-to-end. This is the demo-honesty
 * anchor — we're not hiding the quantum step behind an abstraction.
 */
import { computed, ref } from 'vue'
import type { QaoaExtras } from '@/stores/solve'

const props = defineProps<{
  extras: QaoaExtras
  /** Which solver produced this — drives which tab is active by default. */
  solverName: string
}>()

// Default tab per solver: originqc → pyqpanda, ibmq → qiskit, sim → bqm.
const initialTab = (() => {
  if (props.solverName === 'qaoa_originqc') return 'pyqpanda'
  if (props.solverName === 'qaoa_ibmq') return 'qiskit'
  return 'bqm'
})()
const tab = ref<'pyqpanda' | 'qiskit' | 'bqm'>(initialTab)
const justCopied = ref(false)

function fmt(n: number): string {
  return Number.isFinite(n) ? n.toFixed(6) : '0.000000'
}

const pyqpandaCode = computed(() => {
  const {
    num_qubits, layer, trained_gammas, trained_betas,
    linear_terms, quadratic_terms, backend_name, shots,
  } = props.extras
  const p = layer ?? 1
  const shotCount = shots ?? 200
  const backend = backend_name || 'full_amplitude'
  const lines: string[] = []
  lines.push('from pyqpanda3.core import *')
  lines.push('from pyqpanda3.qcloud import QCloudService')
  lines.push('')
  lines.push(`# ${num_qubits} qubits · ${p} QAOA layer${p === 1 ? '' : 's'} · ${shotCount} shots`)
  lines.push('prog = QProg()')
  lines.push('')
  lines.push('# Uniform superposition prep')
  for (let q = 0; q < num_qubits; q++) {
    lines.push(`prog << H(${q})`)
  }
  for (let li = 0; li < p; li++) {
    const g = trained_gammas[li] ?? 0
    const b = trained_betas[li] ?? 0
    lines.push('')
    lines.push(`# ---- QAOA layer ${li + 1} · γ = ${fmt(g)}, β = ${fmt(b)} ----`)
    if (quadratic_terms.length) {
      lines.push('# ZZ couplings (CNOT · RZ(2γJ) · CNOT for each edge)')
      for (const [qi, qj, jij] of quadratic_terms) {
        const angle = 2 * g * jij
        lines.push(`prog << CNOT(${qi}, ${qj})`)
        lines.push(`prog << RZ(${qj}, ${fmt(angle)})`)
        lines.push(`prog << CNOT(${qi}, ${qj})`)
      }
    }
    if (linear_terms.length) {
      lines.push('# Local Z biases (RZ(2γh) per qubit)')
      for (const [qi, hi] of linear_terms) {
        const angle = 2 * g * hi
        lines.push(`prog << RZ(${qi}, ${fmt(angle)})`)
      }
    }
    lines.push('# Mixer (RX(2β) on every qubit)')
    for (let q = 0; q < num_qubits; q++) {
      const angle = 2 * b
      lines.push(`prog << RX(${q}, ${fmt(angle)})`)
    }
  }
  lines.push('')
  lines.push('# Measurement')
  for (let q = 0; q < num_qubits; q++) {
    lines.push(`prog << measure(${q}, ${q})`)
  }
  lines.push('')
  lines.push('# Submit to Origin cloud')
  lines.push('service = QCloudService(api_key="YOUR_API_KEY")')
  lines.push(`backend = service.backend(${JSON.stringify(backend)})`)
  lines.push(`job = backend.run(prog, ${shotCount})`)
  lines.push('print(job.job_id())')
  lines.push('result = job.result()')
  lines.push('probs = result.get_probs()')
  lines.push('print(probs)')
  return lines.join('\n')
})

const qiskitCode = computed(() => {
  const {
    num_qubits, layer, trained_gammas, trained_betas,
    linear_terms, quadratic_terms, backend_name, shots,
  } = props.extras
  const p = layer ?? 1
  const shotCount = shots ?? 200
  const backend = backend_name || 'ibm_brisbane'
  const lines: string[] = []
  lines.push('from qiskit import QuantumCircuit')
  lines.push('from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2')
  lines.push('')
  lines.push(`# ${num_qubits} qubits · ${p} QAOA layer${p === 1 ? '' : 's'} · ${shotCount} shots`)
  lines.push(`qc = QuantumCircuit(${num_qubits}, ${num_qubits})`)
  lines.push('')
  lines.push('# Uniform superposition prep')
  for (let q = 0; q < num_qubits; q++) {
    lines.push(`qc.h(${q})`)
  }
  for (let li = 0; li < p; li++) {
    const g = trained_gammas[li] ?? 0
    const b = trained_betas[li] ?? 0
    lines.push('')
    lines.push(`# ---- QAOA layer ${li + 1} · γ = ${fmt(g)}, β = ${fmt(b)} ----`)
    if (quadratic_terms.length) {
      lines.push('# ZZ couplings')
      for (const [qi, qj, jij] of quadratic_terms) {
        const angle = 2 * g * jij
        lines.push(`qc.cx(${qi}, ${qj})`)
        lines.push(`qc.rz(${fmt(angle)}, ${qj})`)
        lines.push(`qc.cx(${qi}, ${qj})`)
      }
    }
    if (linear_terms.length) {
      lines.push('# Local Z biases')
      for (const [qi, hi] of linear_terms) {
        const angle = 2 * g * hi
        lines.push(`qc.rz(${fmt(angle)}, ${qi})`)
      }
    }
    lines.push('# Mixer')
    for (let q = 0; q < num_qubits; q++) {
      lines.push(`qc.rx(${fmt(2 * b)}, ${q})`)
    }
  }
  lines.push('')
  lines.push('# Measure')
  for (let q = 0; q < num_qubits; q++) {
    lines.push(`qc.measure(${q}, ${q})`)
  }
  lines.push('')
  lines.push('# Submit to IBM Quantum')
  lines.push('service = QiskitRuntimeService(channel="ibm_quantum_platform", token="YOUR_API_KEY")')
  lines.push(`backend = service.backend(${JSON.stringify(backend)})`)
  lines.push('sampler = SamplerV2(backend)')
  lines.push(`job = sampler.run([qc], shots=${shotCount})`)
  lines.push('print(job.job_id())')
  lines.push('result = job.result()')
  return lines.join('\n')
})

const bqmCode = computed(() => {
  const { num_qubits, linear_terms, quadratic_terms } = props.extras
  const lines: string[] = []
  lines.push('import dimod')
  lines.push('')
  lines.push('# The BQM the QAOA circuit encodes (Ising Hamiltonian, minimize).')
  lines.push(`# ${num_qubits} qubit${num_qubits === 1 ? '' : 's'} · ${linear_terms.length} h · ${quadratic_terms.length} J`)
  lines.push('linear = {')
  for (const [qi, hi] of linear_terms) {
    lines.push(`    "x_${qi}": ${fmt(hi)},`)
  }
  lines.push('}')
  lines.push('quadratic = {')
  for (const [qi, qj, jij] of quadratic_terms) {
    lines.push(`    ("x_${qi}", "x_${qj}"): ${fmt(jij)},`)
  }
  lines.push('}')
  lines.push('bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)')
  lines.push('')
  lines.push('# Sample locally (statevector QAOA)')
  lines.push('# The platform\'s qaoa_sim wraps this with parameter training.')
  return lines.join('\n')
})

const activeCode = computed(() => {
  if (tab.value === 'pyqpanda') return pyqpandaCode.value
  if (tab.value === 'qiskit') return qiskitCode.value
  return bqmCode.value
})

function copyActive() {
  if (!navigator.clipboard) return
  void navigator.clipboard.writeText(activeCode.value).then(() => {
    justCopied.value = true
    setTimeout(() => (justCopied.value = false), 1400)
  })
}
</script>

<template>
  <div class="code-preview">
    <div class="d-flex align-center mb-2">
      <v-tabs
        v-model="tab"
        density="compact"
        color="primary"
        class="flex-grow-1"
        style="min-height: 32px"
      >
        <v-tab value="pyqpanda">pyqpanda3</v-tab>
        <v-tab value="qiskit">Qiskit</v-tab>
        <v-tab value="bqm">BQM (dimod)</v-tab>
      </v-tabs>
      <v-btn
        size="x-small"
        variant="text"
        :prepend-icon="justCopied ? 'mdi-check' : 'mdi-content-copy'"
        @click="copyActive"
      >
        {{ justCopied ? 'Copied' : 'Copy' }}
      </v-btn>
    </div>
    <pre class="code-block">{{ activeCode }}</pre>
    <div class="text-caption text-medium-emphasis mt-1">
      Paste into a fresh Python file, replace <code>YOUR_API_KEY</code>,
      and you'll reproduce the exact submission that reached the cloud.
    </div>
  </div>
</template>

<style scoped>
.code-block {
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, ui-monospace, monospace;
  font-size: 0.78rem;
  line-height: 1.4;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 0.6rem 0.75rem;
  white-space: pre;
  overflow: auto;
  max-height: 320px;
  margin: 0;
}
code {
  font-family: inherit;
  background: rgba(255, 255, 255, 0.08);
  padding: 1px 4px;
  border-radius: 3px;
}
</style>
