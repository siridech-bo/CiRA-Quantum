<script setup lang="ts">
import { computed } from 'vue'
import type { InstanceMeta, RecordSummary } from '@/stores/benchmarks'
import CitationButton from '@/components/CitationButton.vue'

/**
 * Phase 5C — the dashboard's defining shot.
 *
 * Rows = instances in the suite. Columns = solvers seen on this suite
 * in the archive. Each cell shows the *best* run of that (instance,
 * solver) pair, with the elapsed time and a small chip indicating
 * whether the solver matched the documented optimum.
 *
 * "Best" means lowest ``best_user_energy`` (minimize-encoded; the
 * orchestrator already sign-flipped maximize problems before solving).
 * Ties broken by shortest ``elapsed_ms``. An empty cell means that
 * solver has never been run on that instance — a transparent gap, not
 * a 0.
 */
const props = defineProps<{
  suiteId: string
  instances: InstanceMeta[]
  records: RecordSummary[]
}>()

const emit = defineEmits<{
  (e: 'open-instance', instanceId: string): void
  (e: 'open-solver', solverName: string): void
}>()

// Unique solvers actually present in this suite's records.
const solverNames = computed<string[]>(() => {
  const set = new Set<string>()
  for (const r of props.records) set.add(r.solver_name)
  return Array.from(set).sort()
})

interface Cell {
  best: RecordSummary | null
  numRuns: number
}

const grid = computed<Map<string, Cell>>(() => {
  const cells = new Map<string, Cell>()
  for (const inst of props.instances) {
    for (const solver of solverNames.value) {
      cells.set(key(inst.instance_id, solver), { best: null, numRuns: 0 })
    }
  }
  for (const r of props.records) {
    const k = key(r.instance_id, r.solver_name)
    const cell = cells.get(k)
    if (!cell) continue
    cell.numRuns += 1
    const cur = cell.best
    if (
      cur === null ||
      ((r.best_user_energy ?? Infinity) < (cur.best_user_energy ?? Infinity)) ||
      ((r.best_user_energy ?? Infinity) === (cur.best_user_energy ?? Infinity) &&
        (r.elapsed_ms ?? Infinity) < (cur.elapsed_ms ?? Infinity))
    ) {
      cell.best = r
    }
  }
  return cells
})

function key(instanceId: string, solverName: string): string {
  return `${instanceId}__${solverName}`
}

function cellFor(instanceId: string, solverName: string): Cell | undefined {
  return grid.value.get(key(instanceId, solverName))
}

function fmtTime(ms: number | null | undefined): string {
  if (ms == null) return '—'
  if (ms < 10) return ms.toFixed(2) + ' ms'
  if (ms < 1000) return Math.round(ms) + ' ms'
  return (ms / 1000).toFixed(2) + ' s'
}

function fmtEnergy(e: number | null | undefined): string {
  if (e == null) return '—'
  // Strip trailing .0 for the common integer-valued case.
  return Number.isInteger(e) ? e.toString() : e.toFixed(3)
}

function bestOnRow(instanceId: string): RecordSummary | null {
  let best: RecordSummary | null = null
  for (const solver of solverNames.value) {
    const cell = cellFor(instanceId, solver)
    if (!cell?.best) continue
    if (
      best === null ||
      (cell.best.best_user_energy ?? Infinity) < (best.best_user_energy ?? Infinity)
    ) {
      best = cell.best
    }
  }
  return best
}

function isBestOnRow(instanceId: string, solverName: string): boolean {
  const best = bestOnRow(instanceId)
  return !!best && best.solver_name === solverName
}

function matchState(r: RecordSummary | null): 'match' | 'mismatch' | 'unknown' {
  if (!r) return 'unknown'
  if (r.converged_to_expected === true) return 'match'
  if (r.converged_to_expected === false) return 'mismatch'
  return 'unknown'
}
</script>

<template>
  <v-card>
    <v-card-title class="d-flex align-center">
      <span>{{ suiteId }}</span>
      <v-spacer />
      <span class="text-body-2 text-medium-emphasis">
        {{ instances.length }} instances · {{ solverNames.length }} solvers ·
        {{ records.length }} records
      </span>
    </v-card-title>
    <v-card-subtitle class="pa-4 pt-0 text-medium-emphasis">
      Each cell shows the best run of that (instance, solver) pair.
      <strong>Best</strong> energy on a row is highlighted; <code>✓</code> means
      the solver matched the documented optimum, <code>!</code> means it didn't.
      Empty cells mean that solver has never been run on the instance.
    </v-card-subtitle>

    <div class="table-scroll">
      <v-table density="compact">
        <thead>
          <tr>
            <th class="sticky-col instance-col text-left">Instance</th>
            <th class="text-left">Expected</th>
            <th
              v-for="s in solverNames"
              :key="s"
              class="solver-col text-right"
            >
              <a class="solver-link" @click="emit('open-solver', s)">
                {{ s }}
              </a>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="inst in instances" :key="inst.instance_id">
            <td class="sticky-col instance-col">
              <a class="instance-link" @click="emit('open-instance', inst.instance_id)">
                {{ inst.instance_id.split('/').pop() }}
              </a>
              <div class="text-caption text-medium-emphasis">{{ inst.problem_class }}</div>
            </td>
            <td class="text-no-wrap">
              <span v-if="inst.expected_optimum !== null">
                {{ fmtEnergy(inst.expected_optimum) }}
                <span class="text-caption text-medium-emphasis ml-1">
                  ({{ inst.expected_optimum_kind }})
                </span>
              </span>
              <span v-else class="text-medium-emphasis">—</span>
            </td>
            <td
              v-for="s in solverNames"
              :key="s"
              class="text-right"
              :class="{ 'cell-best': isBestOnRow(inst.instance_id, s) }"
            >
              <template v-if="cellFor(inst.instance_id, s)?.best">
                <div class="d-flex align-center justify-end ga-1">
                  <v-chip
                    v-if="matchState(cellFor(inst.instance_id, s)!.best) === 'match'"
                    size="x-small"
                    color="success"
                    variant="tonal"
                    title="Matches the documented optimum"
                  >
                    ✓
                  </v-chip>
                  <v-chip
                    v-else-if="matchState(cellFor(inst.instance_id, s)!.best) === 'mismatch'"
                    size="x-small"
                    color="warning"
                    variant="tonal"
                    title="Does not match the documented optimum"
                  >
                    !
                  </v-chip>
                  <span class="font-weight-medium">
                    {{ fmtEnergy(cellFor(inst.instance_id, s)!.best!.best_user_energy) }}
                  </span>
                </div>
                <div class="text-caption text-medium-emphasis">
                  {{ fmtTime(cellFor(inst.instance_id, s)!.best!.elapsed_ms) }}
                  <span v-if="(cellFor(inst.instance_id, s)?.numRuns || 0) > 1" class="ml-1">
                    (best of {{ cellFor(inst.instance_id, s)?.numRuns }})
                  </span>
                </div>
              </template>
              <span v-else class="text-medium-emphasis">—</span>
            </td>
          </tr>
        </tbody>
      </v-table>
    </div>
  </v-card>
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
.solver-col {
  min-width: 130px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
}
.solver-link,
.instance-link {
  cursor: pointer;
  border-bottom: 1px dotted currentColor;
}
.solver-link:hover,
.instance-link:hover {
  color: rgb(var(--v-theme-primary));
}
.cell-best {
  background: rgba(80, 200, 120, 0.08);
}
</style>
