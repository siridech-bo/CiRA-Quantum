<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted } from 'vue'
import { useSolveStore, type JobStatus } from '@/stores/solve'

const solve = useSolveStore()

const solverList = computed<string[]>(
  () => solve.currentJob?.solvers_requested ?? [],
)
const solvingHint = computed(() => {
  if (solverList.value.length > 1) {
    return `Running ${solverList.value.length} solvers sequentially`
  }
  if (solverList.value.length === 1) {
    return `Running ${solverList.value[0]}`
  }
  return 'Sampling the BQM lowering'
})

interface SolverProgressRow {
  name: string
  status: 'pending' | 'running' | 'complete' | 'error'
  energy?: number
  elapsed_ms?: number
  error?: string
}

const solverProgressRows = computed<SolverProgressRow[]>(() => {
  const job = solve.currentJob
  if (!job) return []
  const requested = job.solvers_requested ?? []
  const live = job.solver_results?.solvers ?? {}
  // Preserve the original request order so the user sees solvers in
  // the order they'll actually be run, not alphabetical.
  return requested.map((name) => {
    const r = live[name]
    return {
      name,
      status: (r?.status ?? 'pending') as SolverProgressRow['status'],
      energy: r?.energy ?? undefined,
      elapsed_ms: r?.elapsed_ms ?? undefined,
      error: r?.error,
    }
  })
})

function fmtTime(ms: number | undefined): string {
  if (ms === undefined) return ''
  if (ms < 10) return `${ms.toFixed(1)} ms`
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}
function fmtEnergy(e: number | undefined): string {
  if (e === undefined || e === null) return '—'
  if (Math.abs(e) >= 1000) return e.toExponential(2)
  return e.toFixed(3)
}

const STAGES = computed<{ key: JobStatus; label: string; hint: string }[]>(() => [
  { key: 'formulating', label: 'Formulating', hint: 'LLM is writing the CQM' },
  { key: 'compiling',   label: 'Compiling',   hint: 'JSON → dimod.CQM' },
  { key: 'validating',  label: 'Validating',  hint: 'Oracle + constraint checks' },
  { key: 'solving',     label: 'Solving',     hint: solvingHint.value },
  { key: 'complete',    label: 'Done',        hint: 'Interpreting the result' },
])

const STAGE_INDEX: Record<JobStatus, number> = {
  queued: -1,
  formulating: 0,
  compiling: 1,
  validating: 2,
  solving: 3,
  complete: 4,
  error: 4, // surfaced separately via the error alert below
}

const status = computed<JobStatus>(() => solve.currentJob?.status ?? 'queued')
const isError = computed(() => status.value === 'error')
const currentIndex = computed(() => STAGE_INDEX[status.value])

const startedAt = ref<number | null>(null)
const now = ref(Date.now())
let tick: number | null = null

function stageState(idx: number): 'done' | 'active' | 'pending' | 'error' {
  if (isError.value && idx === currentIndex.value) return 'error'
  if (status.value === 'complete' && idx === 4) return 'done'
  if (idx < currentIndex.value) return 'done'
  if (idx === currentIndex.value) return 'active'
  return 'pending'
}

const elapsed = computed(() => {
  if (!startedAt.value) return '0.0'
  return ((now.value - startedAt.value) / 1000).toFixed(1)
})

onMounted(() => {
  startedAt.value = solve.currentJob?.created_at
    ? new Date(solve.currentJob.created_at).getTime()
    : Date.now()
  tick = window.setInterval(() => {
    now.value = Date.now()
  }, 200)
})

onUnmounted(() => {
  if (tick !== null) window.clearInterval(tick)
})
</script>

<template>
  <v-card class="pa-5">
    <div class="d-flex align-center mb-3">
      <v-card-title class="pa-0 text-h6 flex-grow-1">
        Solve in progress
      </v-card-title>
      <v-chip v-if="!isError" variant="tonal" color="info">
        <v-icon icon="mdi-timer-outline" start />
        {{ elapsed }}s
      </v-chip>
      <v-btn
        variant="text"
        density="compact"
        class="ml-2"
        prepend-icon="mdi-close"
        disabled
      >
        Cancel (Phase 6)
      </v-btn>
    </div>

    <v-timeline align="start" side="end" density="compact">
      <v-timeline-item
        v-for="(stage, idx) in STAGES"
        :key="stage.key"
        :dot-color="
          stageState(idx) === 'done'   ? 'success' :
          stageState(idx) === 'active' ? 'primary' :
          stageState(idx) === 'error'  ? 'error'   : 'grey-darken-2'
        "
        :icon="
          stageState(idx) === 'done'   ? 'mdi-check' :
          stageState(idx) === 'error'  ? 'mdi-alert' :
          undefined
        "
        size="small"
      >
        <template v-if="stageState(idx) === 'active' && !isError" #icon>
          <v-progress-circular indeterminate size="14" width="2" color="white" />
        </template>
        <div :class="['stage-row', `state-${stageState(idx)}`]">
          <span class="text-body-1">{{ stage.label }}</span>
          <span class="text-caption text-medium-emphasis ml-3">{{ stage.hint }}</span>
        </div>
        <!-- Per-solver checklist, only inside the Solving stage. -->
        <div
          v-if="stage.key === 'solving' && stageState(idx) === 'active' && solverProgressRows.length"
          class="solver-list mt-2"
        >
          <div
            v-for="row in solverProgressRows"
            :key="row.name"
            class="solver-list-row"
            :class="`solver-row--${row.status}`"
          >
            <span class="solver-icon">
              <v-icon
                v-if="row.status === 'complete'"
                icon="mdi-check-circle"
                color="success"
                size="x-small"
              />
              <v-icon
                v-else-if="row.status === 'error'"
                icon="mdi-alert-circle"
                color="error"
                size="x-small"
              />
              <v-progress-circular
                v-else-if="row.status === 'running'"
                indeterminate
                size="12"
                width="2"
                color="primary"
              />
              <v-icon
                v-else
                icon="mdi-circle-outline"
                size="x-small"
                color="grey"
              />
            </span>
            <code class="solver-name">{{ row.name }}</code>
            <span class="solver-detail">
              <template v-if="row.status === 'complete'">
                E = {{ fmtEnergy(row.energy) }} · {{ fmtTime(row.elapsed_ms) }}
              </template>
              <template v-else-if="row.status === 'error'">
                <v-tooltip location="top">
                  <template #activator="{ props: tipProps }">
                    <span v-bind="tipProps" class="text-error">errored</span>
                  </template>
                  <span>{{ row.error || 'Unknown error' }}</span>
                </v-tooltip>
              </template>
              <template v-else-if="row.status === 'running'">
                running…
              </template>
              <template v-else>
                waiting
              </template>
            </span>
          </div>
        </div>
      </v-timeline-item>
    </v-timeline>

    <v-alert
      v-if="isError"
      type="error"
      variant="tonal"
      class="mt-4"
      title="Pipeline failed"
    >
      {{ solve.currentJob?.error || 'Unknown error' }}
    </v-alert>

    <div v-if="solve.currentJob?.num_variables" class="mt-4 text-caption text-medium-emphasis">
      Encoded as {{ solve.currentJob.num_variables }} variables,
      {{ solve.currentJob.num_constraints }} constraints
    </div>
  </v-card>
</template>

<style scoped>
.stage-row {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
}
.state-pending {
  opacity: 0.45;
}
.state-active {
  font-weight: 600;
}

.solver-list {
  display: grid;
  gap: 0.2rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 4px;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;
}
.solver-list-row {
  display: grid;
  grid-template-columns: 20px 1fr auto;
  align-items: center;
  gap: 0.5rem;
}
.solver-row--pending { opacity: 0.45; }
.solver-row--running { font-weight: 600; }
.solver-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.solver-name {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.82rem;
}
.solver-detail {
  font-size: 0.78rem;
  color: rgba(255, 255, 255, 0.55);
  font-family: 'Cascadia Code', 'Consolas', monospace;
}
</style>
