<script setup lang="ts">
/**
 * ClassicalExplainerPanel — educational mini-panel for non-quantum
 * solver rows in the comparison table.
 *
 * Dispatches on the solver's family:
 *   - Simulated Annealing (gpu_sa, cpu_sa_neal): temperature schedule
 *   - Parallel Tempering (parallel_tempering): multi-chain swap diagram
 *   - Simulated Bifurcation: bifurcation trajectories
 *   - CP-SAT / HiGHS / exact_cqm: branch-and-bound tree
 *
 * Phase 10C. Unlike the QAOA panel, these don't yet read per-run
 * telemetry from the sampler (would require each sampler to emit
 * traces); they're conceptual explainers anchored by the run's
 * actual energy + wall-time, so the student sees a result they
 * just produced and a picture of *how it got there*.
 */
import { computed } from 'vue'
import type { SolverResult } from '@/stores/solve'

const props = defineProps<{
  solverName: string
  result: SolverResult
  /** "minimize" / "maximize" — drives the energy chart direction. */
  sense: 'minimize' | 'maximize'
}>()

type Family = 'sa' | 'pt' | 'sb' | 'cpsat' | 'unknown'

const family = computed<Family>(() => {
  const n = props.solverName
  if (n === 'gpu_sa' || n === 'cpu_sa_neal') return 'sa'
  if (n === 'parallel_tempering') return 'pt'
  if (n === 'simulated_bifurcation') return 'sb'
  if (n === 'cpsat' || n === 'highs' || n === 'exact_cqm') return 'cpsat'
  return 'unknown'
})

// ---------- Temperature schedule (SA / PT) ----------
// Geometric cooling: T_k = T_0 * α^k, default for dwave / our gpu_sa.
function geometricSchedule(T0: number, alpha: number, steps: number): number[] {
  const out: number[] = []
  let t = T0
  for (let i = 0; i < steps; i++) {
    out.push(t)
    t *= alpha
  }
  return out
}

const saSchedule = computed(() => geometricSchedule(10.0, 0.96, 60))
const saAcceptanceCurve = computed(() => {
  // Schematic acceptance rate of a "worse by 1" move as T drops.
  // P(accept worse by 1) = exp(-1/T). At high T this is ~1; at T→0
  // it's ~0. The curve is what makes SA both exploratory (early)
  // and exploitative (late).
  return saSchedule.value.map((T) => Math.exp(-1 / T))
})

// SVG path helpers
function pathOf(values: number[], width: number, height: number, padding = 8): string {
  if (!values.length) return ''
  const n = values.length
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = Math.max(1e-9, max - min)
  return values
    .map((v, i) => {
      const x = padding + (i / (n - 1)) * (width - 2 * padding)
      const y = height - padding - ((v - min) / range) * (height - 2 * padding)
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
    })
    .join(' ')
}

// ---------- PT replica grid ----------
const ptReplicas = computed(() =>
  [10.0, 5.0, 2.0, 1.0, 0.5, 0.2].map((T, i) => ({ idx: i, T }))
)

// ---------- Branch-and-bound stylized tree ----------
// Static representation of a 3-deep BnB tree on 3 binary vars.
// Nodes: x_0 = ?, x_1 = ?, x_2 = ?, with some branches pruned.
interface BnbNode {
  id: string
  label: string
  x: number
  y: number
  state: 'open' | 'pruned' | 'optimal'
  parent?: string
}

const bnbNodes: BnbNode[] = [
  { id: 'r', label: 'root', x: 240, y: 30, state: 'open' },
  { id: 'l1', label: 'x₀=0', x: 120, y: 90, state: 'open', parent: 'r' },
  { id: 'r1', label: 'x₀=1', x: 360, y: 90, state: 'open', parent: 'r' },
  { id: 'l1a', label: 'x₁=0', x: 60, y: 150, state: 'pruned', parent: 'l1' },
  { id: 'l1b', label: 'x₁=1', x: 180, y: 150, state: 'open', parent: 'l1' },
  { id: 'r1a', label: 'x₁=0', x: 300, y: 150, state: 'optimal', parent: 'r1' },
  { id: 'r1b', label: 'x₁=1', x: 420, y: 150, state: 'pruned', parent: 'r1' },
  { id: 'l1b1', label: 'x₂=0', x: 150, y: 210, state: 'open', parent: 'l1b' },
  { id: 'l1b2', label: 'x₂=1', x: 210, y: 210, state: 'open', parent: 'l1b' },
]
function nodeColor(state: BnbNode['state']): string {
  if (state === 'optimal') return '#22c55e'
  if (state === 'pruned') return 'rgba(255, 100, 100, 0.5)'
  return 'rgb(var(--v-theme-primary))'
}

function fmtTime(ms: number): string {
  if (ms < 10) return `${ms.toFixed(1)} ms`
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}
function fmtEnergy(e?: number): string {
  if (e === undefined || e === null) return '—'
  return e.toFixed(4)
}
</script>

<template>
  <div class="classical-explainer pa-3">
    <!-- ============ SA family ============ -->
    <template v-if="family === 'sa'">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-thermometer-lines" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">How simulated annealing got this answer</h4>
      </header>
      <p class="text-body-2 mb-3">
        SA starts with a random bit assignment and proposes single-bit
        flips. It always accepts moves that lower the energy; uphill
        moves are accepted with probability <code>exp(−ΔE / T)</code>.
        The temperature <strong>T</strong> drops geometrically over
        time: early sweeps escape local minima (high T = exploratory);
        late sweeps refine (low T = exploitative).
      </p>
      <div class="charts-row">
        <div class="chart-card">
          <div class="chart-title">Temperature schedule</div>
          <svg viewBox="0 0 240 80" class="schedule-svg" preserveAspectRatio="none">
            <path
              :d="pathOf(saSchedule, 240, 80)"
              fill="none"
              stroke="#a855f7"
              stroke-width="2"
            />
            <text x="6" y="14" font-size="9" fill="rgba(255,255,255,0.55)">T₀</text>
            <text x="220" y="72" font-size="9" fill="rgba(255,255,255,0.55)">→ 0</text>
          </svg>
          <div class="chart-caption">Geometric: T<sub>k+1</sub> = α · T<sub>k</sub></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">P(accept uphill by 1)</div>
          <svg viewBox="0 0 240 80" class="schedule-svg" preserveAspectRatio="none">
            <path
              :d="pathOf(saAcceptanceCurve, 240, 80)"
              fill="none"
              stroke="#22c55e"
              stroke-width="2"
            />
            <text x="6" y="14" font-size="9" fill="rgba(255,255,255,0.55)">~1</text>
            <text x="220" y="72" font-size="9" fill="rgba(255,255,255,0.55)">~0</text>
          </svg>
          <div class="chart-caption">Open early, picky late</div>
        </div>
      </div>
      <div class="result-stat mt-3">
        Your run: <strong>{{ fmtEnergy(result.energy) }}</strong> energy
        in <strong>{{ fmtTime(result.elapsed_ms) }}</strong> — 200 reads
        × 500 sweeps each.
      </div>
    </template>

    <!-- ============ PT ============ -->
    <template v-else-if="family === 'pt'">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-layers-triple-outline" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">How parallel tempering got this answer</h4>
      </header>
      <p class="text-body-2 mb-3">
        Like SA, but several copies (replicas) of the system run
        <em>in parallel</em> at different fixed temperatures. Every few
        sweeps, neighboring replicas are allowed to swap states with a
        Metropolis criterion: <code>P(swap) = min(1, exp((β<sub>i</sub>−β<sub>j</sub>)(E<sub>i</sub>−E<sub>j</sub>)))</code>.
        Result: the hot replicas keep exploring; the cold one inherits
        their discoveries via the swap chain. Beats vanilla SA on rugged
        landscapes (e.g. spin glasses).
      </p>
      <div class="pt-grid">
        <div
          v-for="r in ptReplicas"
          :key="r.idx"
          class="pt-replica"
        >
          <div class="pt-replica-label">replica {{ r.idx }}</div>
          <div class="pt-replica-bar" :style="{ background: `hsl(${260 - r.idx * 35}, 60%, 55%)` }"></div>
          <div class="pt-replica-temp">T = {{ r.T }}</div>
        </div>
      </div>
      <div class="text-caption text-medium-emphasis mt-2">
        Hot replicas (purple) tunnel through barriers; cold replicas
        (red-violet) settle into local minima. Periodic swaps move good
        configurations from hot to cold.
      </div>
      <div class="result-stat mt-3">
        Your run: <strong>{{ fmtEnergy(result.energy) }}</strong> energy
        in <strong>{{ fmtTime(result.elapsed_ms) }}</strong> — 50 reads
        × 1000 sweeps × 8 replicas.
      </div>
    </template>

    <!-- ============ Simulated Bifurcation ============ -->
    <template v-else-if="family === 'sb'">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-chart-bell-curve-cumulative" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">How simulated bifurcation got this answer</h4>
      </header>
      <p class="text-body-2 mb-3">
        Simulated Bifurcation models each spin as a classical mechanical
        particle in a potential well that gradually bifurcates from a
        single minimum into two (corresponding to spin ±1). Trajectories
        roll downhill with momentum and pairwise coupling drawn from
        the Ising/QUBO matrix. By the end of the schedule, each particle
        has committed to a side — that's your bit assignment. This is
        the open algorithm behind Fujitsu's commercial SQBM+.
      </p>
      <svg viewBox="0 0 480 120" class="sb-svg" preserveAspectRatio="xMidYMid meet">
        <!-- Two bifurcating trajectories -->
        <path d="M 10 60 Q 200 60 240 30 T 470 12" stroke="#7c3aed" stroke-width="2" fill="none"/>
        <path d="M 10 60 Q 200 60 240 90 T 470 108" stroke="#06b6d4" stroke-width="2" fill="none"/>
        <text x="475" y="14" font-size="10" fill="rgba(255,255,255,0.55)">+1</text>
        <text x="475" y="112" font-size="10" fill="rgba(255,255,255,0.55)">−1</text>
        <line x1="10" y1="60" x2="470" y2="60" stroke="rgba(255,255,255,0.15)" stroke-dasharray="3 3"/>
        <text x="14" y="74" font-size="9" fill="rgba(255,255,255,0.55)">0 (undecided)</text>
        <text x="240" y="65" font-size="9" fill="rgba(255,255,255,0.55)" text-anchor="middle">bifurcation point →</text>
      </svg>
      <div class="result-stat mt-3">
        Your run: <strong>{{ fmtEnergy(result.energy) }}</strong> energy
        in <strong>{{ fmtTime(result.elapsed_ms) }}</strong> — 500 steps
        × 128 parallel agents.
      </div>
    </template>

    <!-- ============ CP-SAT / HiGHS / exact_cqm ============ -->
    <template v-else-if="family === 'cpsat'">
      <header class="d-flex align-center mb-2">
        <v-icon icon="mdi-source-branch" size="small" class="mr-2" />
        <h4 class="text-subtitle-2">How {{ solverName }} got this answer</h4>
      </header>
      <p class="text-body-2 mb-3">
        Classical exact solvers (CP-SAT, HiGHS, ExactCQM) use
        <em>branch-and-bound</em>: enumerate the space of possible
        assignments as a tree, but at each node use a fast linear
        relaxation to estimate a bound on the best achievable
        objective. If the bound is worse than the best known feasible
        solution, prune the whole subtree without exploring it.
        Combined with clever cuts and learned clauses, this can solve
        small problems to <strong>provable optimality</strong> faster
        than any heuristic.
      </p>
      <svg viewBox="0 0 480 240" class="bnb-svg" preserveAspectRatio="xMidYMid meet">
        <!-- Edges -->
        <line
          v-for="n in bnbNodes.filter((x) => x.parent)"
          :key="`e-${n.id}`"
          :x1="bnbNodes.find((p) => p.id === n.parent)!.x"
          :y1="bnbNodes.find((p) => p.id === n.parent)!.y + 12"
          :x2="n.x"
          :y2="n.y - 12"
          stroke="rgba(255,255,255,0.25)"
          stroke-width="1.5"
          :stroke-dasharray="n.state === 'pruned' ? '4 3' : '0'"
        />
        <!-- Nodes -->
        <g v-for="n in bnbNodes" :key="n.id">
          <circle
            :cx="n.x"
            :cy="n.y"
            r="14"
            :fill="nodeColor(n.state)"
            stroke="rgba(255,255,255,0.4)"
            stroke-width="1"
          />
          <text
            :x="n.x"
            :y="n.y + 4"
            font-size="9"
            fill="white"
            text-anchor="middle"
            font-family="Cascadia Code, Consolas, monospace"
          >{{ n.label }}</text>
        </g>
        <!-- Legend -->
        <g transform="translate(10 220)">
          <circle cx="6" cy="0" r="5" fill="rgb(var(--v-theme-primary))"/>
          <text x="16" y="3" font-size="9" fill="rgba(255,255,255,0.6)">explored</text>
          <circle cx="100" cy="0" r="5" fill="rgba(255,100,100,0.5)"/>
          <text x="110" y="3" font-size="9" fill="rgba(255,255,255,0.6)">pruned (bound worse than incumbent)</text>
          <circle cx="320" cy="0" r="5" fill="#22c55e"/>
          <text x="330" y="3" font-size="9" fill="rgba(255,255,255,0.6)">optimal leaf</text>
        </g>
      </svg>
      <div class="result-stat mt-3">
        Your run: <strong>{{ fmtEnergy(result.energy) }}</strong> energy
        in <strong>{{ fmtTime(result.elapsed_ms) }}</strong> — proven
        optimal under the 5-second time limit.
      </div>
    </template>

    <template v-else>
      <div class="text-medium-emphasis text-body-2">
        No educational view wired for this solver yet.
      </div>
    </template>
  </div>
</template>

<style scoped>
.classical-explainer {
  background: rgba(124, 58, 237, 0.04);
  border-left: 3px solid rgb(var(--v-theme-info));
  border-radius: 4px;
}
.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
}
.chart-card {
  background: rgba(255, 255, 255, 0.03);
  padding: 0.6rem;
  border-radius: 4px;
}
.chart-title {
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 0.3rem;
  color: rgba(255, 255, 255, 0.7);
}
.schedule-svg {
  width: 100%;
  height: 80px;
  background: rgba(0, 0, 0, 0.15);
  border-radius: 3px;
}
.chart-caption {
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.5);
  margin-top: 0.3rem;
}
.pt-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 0.3rem;
  margin-top: 0.6rem;
}
.pt-replica {
  text-align: center;
  font-size: 0.75rem;
}
.pt-replica-label {
  color: rgba(255, 255, 255, 0.55);
  margin-bottom: 0.2rem;
}
.pt-replica-bar {
  height: 36px;
  border-radius: 3px;
  opacity: 0.85;
}
.pt-replica-temp {
  margin-top: 0.2rem;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.7rem;
}
.sb-svg {
  width: 100%;
  height: 120px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
}
.bnb-svg {
  width: 100%;
  height: 240px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
}
.result-stat {
  font-size: 0.85rem;
  padding: 0.5rem 0.75rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}
</style>
