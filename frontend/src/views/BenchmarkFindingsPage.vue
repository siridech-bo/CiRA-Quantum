<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useBenchmarksStore, type FindingsCell, type SolverSummary } from '@/stores/benchmarks'
import PendingCloudJobsPanel from '@/components/PendingCloudJobsPanel.vue'

const router = useRouter()
const benchmarks = useBenchmarksStore()

onMounted(() => {
  void benchmarks.loadFindings()
})

// Group cells by instance for the headline grid.
const cellsByInstance = computed(() => {
  const out: Record<string, FindingsCell[]> = {}
  for (const c of benchmarks.findings?.cells || []) {
    if (!out[c.instance_id]) out[c.instance_id] = []
    out[c.instance_id].push(c)
  }
  return out
})

// Unique solver names across the experiment, sorted.
const solverNames = computed<string[]>(() => {
  const set = new Set<string>()
  for (const c of benchmarks.findings?.cells || []) {
    set.add(c.solver_name)
  }
  return Array.from(set).sort()
})

// Per-instance: best result across all solvers (for highlighting).
function bestSolverOnInstance(instanceId: string): string | null {
  const cells = cellsByInstance.value[instanceId] || []
  let best: FindingsCell | null = null
  for (const c of cells) {
    if (c.best_user_energy === null) continue
    if (best === null || c.best_user_energy < best.best_user_energy!) {
      best = c
    }
  }
  return best?.solver_name || null
}

function cellFor(instanceId: string, solverName: string): FindingsCell | undefined {
  return (cellsByInstance.value[instanceId] || []).find(c => c.solver_name === solverName)
}

function fmtEnergy(e: number | null | undefined): string {
  if (e == null) return '—'
  return Number.isInteger(e) ? e.toString() : e.toFixed(3)
}

function fmtTime(ms: number | null | undefined): string {
  if (ms == null) return '—'
  if (ms < 10) return ms.toFixed(2) + ' ms'
  if (ms < 1000) return Math.round(ms) + ' ms'
  return (ms / 1000).toFixed(2) + ' s'
}

function fmtPct(p: number | null | undefined): string {
  if (p == null) return '—'
  return `${Math.round(p * 100)}%`
}

function matchRateForSummary(s: SolverSummary): number {
  if (s.instances_attempted === 0) return 0
  return s.instances_with_match / s.instances_attempted
}
</script>

<template>
  <v-app-bar color="surface" flat>
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/benchmarks')" />
    <v-app-bar-title class="text-primary font-weight-bold">
      Benchmarks · Findings
    </v-app-bar-title>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-skeleton-loader v-if="benchmarks.loading && !benchmarks.findings" type="article, table" />
      <v-alert v-else-if="benchmarks.error" type="error" variant="tonal">
        {{ benchmarks.error }}
      </v-alert>

      <template v-else-if="benchmarks.findings">
        <!-- Pending cloud-jobs panel — polls every 30s, auto-materializes
             jobs when their probability counts become available. -->
        <PendingCloudJobsPanel />

        <!-- Headline tonal card -->
        <v-card class="pa-6 mb-4" variant="tonal" color="primary">
          <v-card-title class="pa-0 text-h5 mb-1">
            <v-icon icon="mdi-chart-box" start /> Benchmark Experiment #1
          </v-card-title>
          <v-card-subtitle class="pa-0 text-body-2">
            Controlled sweep across all six registered solver tiers,
            with 5-seed sweeps for stochastic solvers and consistent
            per-class hyperparameters. Cells aggregate every archived
            run that matches the (solver, instance) pair — best /
            mean / worst energy, mean wall time, and convergence rate
            against the documented optimum.
            See <code>BENCHMARK_REPORT_001.md</code> at the repo root for the full prose writeup with five headline findings and recommendations forward.
          </v-card-subtitle>
        </v-card>

        <!-- Per-solver summary cards -->
        <div class="text-h6 mb-2">Per-solver headline stats</div>
        <v-row class="mb-4">
          <v-col
            v-for="s in benchmarks.findings.solver_summaries"
            :key="s.solver_name"
            cols="12"
            sm="6"
            md="4"
            lg="2"
          >
            <v-card
              class="pa-3 h-100"
              hover
              variant="outlined"
              @click="router.push(`/benchmarks/solvers/${s.solver_name}`)"
            >
              <div class="text-subtitle-2 font-weight-bold mb-1">
                {{ s.solver_name }}
              </div>
              <div class="d-flex flex-column ga-1">
                <div>
                  <span class="text-body-2 text-medium-emphasis">match rate:</span>
                  <strong class="ml-1"
                    :class="matchRateForSummary(s) >= 0.8 ? 'text-success' :
                            matchRateForSummary(s) >= 0.5 ? 'text-warning' : 'text-error'">
                    {{ fmtPct(matchRateForSummary(s)) }}
                  </strong>
                </div>
                <div class="text-caption text-medium-emphasis">
                  {{ s.instances_with_match }} / {{ s.instances_attempted }} instances hit the optimum
                </div>
                <div class="text-caption text-medium-emphasis">
                  {{ s.total_runs }} total run{{ s.total_runs !== 1 ? 's' : '' }} archived
                </div>
              </div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Aggregated grid: rows=instances, cols=solvers, cells=stats -->
        <div class="text-h6 mb-2">Per-(solver, instance) aggregated results</div>
        <v-card>
          <v-card-subtitle class="pa-4 pt-3 text-medium-emphasis">
            Each cell shows <strong>best</strong> energy (lowest across seeds),
            <strong>mean</strong> energy when more than one run, and
            <strong>(N)</strong> the number of archived runs. The
            <strong>best result on each row</strong> is highlighted green.
            <code>✓</code> means every run hit the documented optimum;
            <code>!</code> means none did; <code>~</code> means partial.
          </v-card-subtitle>

          <div class="table-scroll">
            <v-table density="compact">
              <thead>
                <tr>
                  <th class="sticky-col instance-col text-left">Instance</th>
                  <th class="text-right">Expected</th>
                  <th
                    v-for="solver in solverNames"
                    :key="solver"
                    class="text-right"
                  >
                    {{ solver }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(cells, instanceId) in cellsByInstance" :key="instanceId">
                  <td class="sticky-col instance-col">
                    <a class="instance-link" @click="router.push(`/benchmarks/instances/${instanceId}`)">
                      {{ instanceId.split('/').pop() }}
                    </a>
                    <div class="text-caption text-medium-emphasis">
                      {{ instanceId.split('/').slice(0, -1).join('/') }}
                    </div>
                  </td>
                  <td class="text-right text-no-wrap">
                    <span v-if="cells[0]?.expected_optimum !== null">
                      {{ fmtEnergy(cells[0].expected_optimum) }}
                    </span>
                    <span v-else class="text-medium-emphasis">—</span>
                  </td>
                  <td
                    v-for="solver in solverNames"
                    :key="solver"
                    class="text-right"
                    :class="{ 'cell-best': bestSolverOnInstance(instanceId) === solver && cellFor(instanceId, solver)?.best_user_energy !== null }"
                  >
                    <template v-if="cellFor(instanceId, solver)">
                      <div class="d-flex align-center justify-end ga-1">
                        <v-chip
                          v-if="cellFor(instanceId, solver)!.convergence_rate === 1"
                          size="x-small"
                          color="success"
                          variant="tonal"
                          title="100% of runs hit the documented optimum"
                        >
                          ✓
                        </v-chip>
                        <v-chip
                          v-else-if="cellFor(instanceId, solver)!.convergence_rate === 0"
                          size="x-small"
                          color="error"
                          variant="tonal"
                          title="0% of runs hit the documented optimum"
                        >
                          !
                        </v-chip>
                        <v-chip
                          v-else-if="cellFor(instanceId, solver)!.convergence_rate !== null"
                          size="x-small"
                          color="warning"
                          variant="tonal"
                          :title="`${Math.round(cellFor(instanceId, solver)!.convergence_rate! * 100)}% of runs hit the documented optimum`"
                        >
                          ~
                        </v-chip>
                        <strong>{{ fmtEnergy(cellFor(instanceId, solver)!.best_user_energy) }}</strong>
                      </div>
                      <div
                        v-if="cellFor(instanceId, solver)!.n_runs > 1"
                        class="text-caption text-medium-emphasis"
                      >
                        mean {{ fmtEnergy(cellFor(instanceId, solver)!.mean_user_energy) }}
                        (N={{ cellFor(instanceId, solver)!.n_runs }})
                      </div>
                      <div class="text-caption text-medium-emphasis">
                        best {{ fmtTime(cellFor(instanceId, solver)!.best_elapsed_ms) }}
                      </div>
                    </template>
                    <span v-else class="text-medium-emphasis">—</span>
                  </td>
                </tr>
              </tbody>
            </v-table>
          </div>
        </v-card>

        <!-- Methodology footnote -->
        <v-card class="pa-4 mt-4" variant="outlined">
          <div class="text-subtitle-2 font-weight-bold mb-2">
            <v-icon icon="mdi-information-outline" start />
            Methodology notes
          </div>
          <ul class="text-body-2 text-medium-emphasis">
            <li>
              <strong>Deterministic solvers</strong>
              (<code>exact_cqm</code>, <code>cpsat</code>, <code>highs</code>):
              1 run per instance, <code>seed=42</code>,
              <code>time_limit=60s</code>.
            </li>
            <li>
              <strong>Stochastic solvers</strong>
              (<code>gpu_sa</code>, <code>cpu_sa_neal</code>,
              <code>qaoa_sim</code>): 5 seeds per instance with
              identical per-class hyperparameters
              (<code>num_reads=500, num_sweeps=1000</code> for SA-class;
              <code>layer=3, optimizer=SLSQP</code> for QAOA).
            </li>
            <li>
              Cells annotated <code>—</code> mean the solver was not
              attempted on that instance (typically: qubit-budget on
              <code>qaoa_sim</code>, quadratic-objective on
              <code>highs</code>, problem-size-blowup on <code>exact_cqm</code>).
            </li>
            <li>
              All numbers in this view aggregate every archived record
              that matches the (solver, instance) pair, including any
              older Phase-1 development runs in addition to Experiment
              #1's controlled sweep. Best-per-cell logic surfaces the
              best across all runs.
            </li>
          </ul>
        </v-card>
      </template>
    </v-container>
  </v-main>
</template>

<style scoped>
.table-scroll {
  overflow-x: auto;
}
.sticky-col {
  position: sticky;
  left: 0;
  background: var(--v-theme-surface);
  z-index: 1;
}
.instance-col {
  min-width: 220px;
}
.instance-link {
  cursor: pointer;
  border-bottom: 1px dotted currentColor;
}
.instance-link:hover {
  color: rgb(var(--v-theme-primary));
}
.cell-best {
  background: rgba(80, 200, 120, 0.10);
}
</style>
