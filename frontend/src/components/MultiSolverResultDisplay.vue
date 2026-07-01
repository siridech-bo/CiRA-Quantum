<script setup lang="ts">
/**
 * MultiSolverResultDisplay — per-solver comparison panel.
 *
 * Renders one row per solver that ran on this job, with energy,
 * elapsed time, feasibility, and a "winner" badge on the row that
 * drove the final ``interpreted_solution``. Rendered above the
 * standard ``ResultDisplay`` (which keeps showing the winning solver's
 * solution / cqm / validation tabs).
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Job, SolverResult, QaoaExtras } from '@/stores/solve'
import { useSolveStore } from '@/stores/solve'
import QaoaExplainerPanel from '@/components/QaoaExplainerPanel.vue'
import ClassicalExplainerPanel from '@/components/ClassicalExplainerPanel.vue'

// Solvers that ClassicalExplainerPanel knows how to render. Kept in
// sync with that component's family-dispatch logic.
const CLASSICAL_EXPLAINER_SOLVERS = new Set([
  'gpu_sa',
  'cpu_sa_neal',
  'parallel_tempering',
  'simulated_bifurcation',
  'cpsat',
  'highs',
  'exact_cqm',
])

const props = defineProps<{ job: Job }>()
const solve = useSolveStore()

const expanded = ref<Set<string>>(new Set())

// Row-level click only OPENS. Once open, a stray click from anywhere
// on the row (Vuetify internals, a poll-driven re-render, a click in
// the code-preview tabs whose event happens to bubble further than
// expected) can't sneak the panel closed. To close, the user clicks
// the chevron in the leftmost cell — that hits ``toggleExpand`` which
// flips the state either way. This matches how the browser's own
// details/summary widget behaves: click anywhere to reveal, click the
// disclosure control to hide.
function expandRow(name: string) {
  if (expanded.value.has(name)) return
  const next = new Set(expanded.value)
  next.add(name)
  expanded.value = next
}
function toggleExpand(name: string) {
  const next = new Set(expanded.value)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  expanded.value = next
}

function getExtras(name: string): QaoaExtras | undefined {
  return props.job.solver_results?.solvers?.[name]?.qaoa_extras
}

const isMaximize = computed(
  () => props.job.solver_results?.sense === 'maximize',
)

const rows = computed(() => {
  const sr = props.job.solver_results
  if (!sr || !sr.solvers) return []
  const primary = sr.primary
  const out = Object.entries(sr.solvers).map(([name, r]: [string, SolverResult]) => {
    const meta = solve.solvers.find((s) => s.name === name)
    return {
      name,
      meta,
      tier_label: meta?.tier_label || 'Solver',
      tier_color: meta?.tier_color || 'default',
      status: r.status,
      energy: r.energy,
      feasible: r.feasible ?? null,
      elapsed_ms: r.elapsed_ms,
      version: r.version || meta?.version,
      hardware: r.hardware ?? meta?.hardware ?? null,
      error: r.error,
      is_primary: name === primary,
      has_explainer: !!r.qaoa_extras || CLASSICAL_EXPLAINER_SOLVERS.has(name),
      explainer_kind: r.qaoa_extras ? 'qaoa' : (CLASSICAL_EXPLAINER_SOLVERS.has(name) ? 'classical' : null),
      solver_result: r,
      cloud_job_id: r.cloud_job_id,
      backend_name: r.backend_name ?? null,
    }
  })
  // Sort: completed-feasible by energy asc, then completed-but-infeasible,
  // then skipped (capacity-refused), then errors at the bottom. Keeps the
  // winner visually at the top and pushes non-winning outcomes down in
  // increasing order of "how informative is this row's energy" — error
  // rows have nothing to compare, skipped rows are deliberate refusals,
  // infeasible-complete rows still carry a number worth seeing.
  const rank = (r: typeof out[number]): number => {
    if (r.status === 'error') return 4
    if (r.status === 'skipped') return 3
    if (!r.feasible) return 2
    return 1
  }
  return out.sort((a, b) => {
    const ra = rank(a)
    const rb = rank(b)
    if (ra !== rb) return ra - rb
    // Treat null / undefined / NaN energies as "no answer" — push to bottom.
    const aBad = a.energy == null || Number.isNaN(a.energy)
    const bBad = b.energy == null || Number.isNaN(b.energy)
    if (aBad && bBad) return 0
    if (aBad) return 1
    if (bBad) return -1
    // For maximize problems, higher energy is better — sort descending
    // so the optimum row floats to the top alongside its trophy.
    return isMaximize.value
      ? (b.energy as number) - (a.energy as number)
      : (a.energy as number) - (b.energy as number)
  })
})

const bestEnergy = computed(() => {
  const completed = rows.value.filter(
    (r) =>
      r.status === 'complete' &&
      r.feasible &&
      r.energy != null &&
      !Number.isNaN(r.energy),
  )
  if (!completed.length) return null
  const energies = completed.map((r) => r.energy as number)
  return isMaximize.value ? Math.max(...energies) : Math.min(...energies)
})

const fastestMs = computed(() => {
  const completed = rows.value.filter((r) => r.status === 'complete')
  if (!completed.length) return null
  return Math.min(...completed.map((r) => r.elapsed_ms))
})

function fmtEnergy(e: number | undefined): string {
  if (e === undefined || e === null) return '—'
  if (Math.abs(e) >= 1000 || (Math.abs(e) < 0.01 && e !== 0)) {
    return e.toExponential(3)
  }
  return e.toFixed(4)
}

function fmtTime(ms: number): string {
  if (ms < 10) return `${ms.toFixed(1)} ms`
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

const completedCount = computed(
  () => rows.value.filter((r) => r.status === 'complete').length,
)
const errorCount = computed(
  () => rows.value.filter((r) => r.status === 'error').length,
)
const queuedCount = computed(
  () => rows.value.filter((r) => r.status === 'queued').length,
)
const skippedCount = computed(
  () => rows.value.filter((r) => r.status === 'skipped').length,
)
// QAOA tiers refuse instances whose lowered BQM exceeds their qubit
// cap. We surface that as a soft "skipped" status (not a red error)
// and show a banner explaining the qubit count and the affected tiers
// so the user understands why the row is gray instead of green.
const skippedRows = computed(() =>
  rows.value.filter((r) => r.status === 'skipped'),
)

// Phase 11 — auto-poll the pending-jobs endpoint while any row is
// queued. Hitting that endpoint triggers materialization server-side
// as a side effect, so the queued row will fill in without the user
// having to manually refresh. Stops the loop as soon as no queued
// rows remain.
//
// Origin's full_amplitude simulator finishes in 1-2 s; IBM's real-QPU
// queue can hold a job for minutes to hours. The old flat 30 s poll
// was tuned for IBM and left Origin users staring at a "queued" chip
// for 30 s after Origin had already finished (2026-07-01 demo). Ramp:
// fast at first (catch quick simulator jobs), then back off (spare
// the SDK from useless polls while a long IBM queue drains).
const POLL_RAMP_MS = [1500, 2500, 4000, 7000, 12000, 20000, 30000]
const lastPolledAt = ref<number | null>(null)
const isRefreshing = ref(false)
let pollTimer: number | null = null
let pollTick = 0

async function manualRefresh() {
  if (isRefreshing.value) return
  isRefreshing.value = true
  try {
    await solve.refreshCloudPoll()
    lastPolledAt.value = Date.now()
  } finally {
    isRefreshing.value = false
  }
}

function nextPollDelayMs(): number {
  const idx = Math.min(pollTick, POLL_RAMP_MS.length - 1)
  return POLL_RAMP_MS[idx]
}

function scheduleNextPoll() {
  if (pollTimer !== null) return
  if (queuedCount.value === 0) return
  const delay = nextPollDelayMs()
  pollTimer = window.setTimeout(async () => {
    pollTimer = null
    if (queuedCount.value === 0) return
    await manualRefresh()
    pollTick += 1
    scheduleNextPoll()
  }, delay)
}

function startPollingIfNeeded() {
  if (queuedCount.value === 0) return
  pollTick = 0
  scheduleNextPoll()
}

function stopPolling() {
  if (pollTimer !== null) {
    // scheduleNextPoll uses setTimeout, not setInterval, so we need the
    // matching clearTimeout to avoid the delayed callback still firing.
    window.clearTimeout(pollTimer)
    pollTimer = null
  }
  pollTick = 0
}

onMounted(() => {
  startPollingIfNeeded()
})
onBeforeUnmount(() => {
  stopPolling()
})
watch(queuedCount, (n) => {
  if (n > 0) {
    startPollingIfNeeded()
  } else {
    stopPolling()
  }
})

function fmtAgo(ms: number | null): string {
  if (ms === null) return 'not yet'
  const dt = Math.floor((Date.now() - ms) / 1000)
  if (dt < 60) return `${dt}s ago`
  return `${Math.floor(dt / 60)} min ago`
}
</script>

<template>
  <v-card class="pa-5 mb-3" v-if="rows.length">
    <div class="d-flex align-center mb-3">
      <v-icon icon="mdi-chart-box-multiple" class="mr-2" />
      <div class="flex-grow-1">
        <div class="text-h6">Solver comparison</div>
        <div class="text-caption text-medium-emphasis">
          {{ rows.length }} solver{{ rows.length === 1 ? '' : 's' }} ·
          {{ completedCount }} completed<span v-if="skippedCount > 0">, {{ skippedCount }} skipped</span><span v-if="errorCount > 0">, {{ errorCount }} errored</span><span v-if="queuedCount > 0">, {{ queuedCount }} queued</span>
        </div>
      </div>
      <v-btn
        v-if="queuedCount > 0"
        size="small"
        variant="tonal"
        color="info"
        :loading="isRefreshing"
        prepend-icon="mdi-refresh"
        class="mr-2"
        :title="`Last checked ${fmtAgo(lastPolledAt)}. Auto-polls on a 1.5→30 s ramp while queued.`"
        @click="manualRefresh"
      >
        Refresh
      </v-btn>
      <v-chip
        v-if="bestEnergy !== null"
        size="small"
        color="success"
        variant="tonal"
        prepend-icon="mdi-trophy"
      >
        Best energy {{ fmtEnergy(bestEnergy) }}
      </v-chip>
    </div>

    <v-alert
      v-if="skippedRows.length > 0"
      type="info"
      variant="tonal"
      density="compact"
      class="mb-3"
      icon="mdi-debug-step-over"
    >
      <div class="text-body-2">
        <strong>{{ skippedRows.length }} solver{{ skippedRows.length === 1 ? '' : 's' }} skipped — instance too large for the tier.</strong>
        QAOA tiers refuse a CQM whose lowered BQM exceeds their qubit cap
        ({{ skippedRows.map((r) => r.name).join(', ') }}).
        This isn't an error — the rows are deliberately left out of the
        comparison. To exercise QAOA, pick a smaller instance or raise
        the relevant cap.
      </div>
    </v-alert>

    <v-table density="compact" class="comparison-table">
      <thead>
        <tr>
          <th class="text-left" style="width: 32px"></th>
          <th class="text-left" style="width: 28%">Solver</th>
          <th class="text-left" style="width: 20%">Tier</th>
          <th class="text-right" style="width: 12%">Energy</th>
          <th class="text-center" style="width: 9%">Feasible</th>
          <th class="text-right" style="width: 12%">Time</th>
          <th class="text-left" style="width: 19%">Status</th>
        </tr>
      </thead>
      <tbody>
        <template v-for="r in rows" :key="r.name">
        <tr
          :class="{
            'row-best': r.is_primary,
            'row-error': r.status === 'error',
            'row-skipped': r.status === 'skipped',
            'row-clickable': r.has_explainer,
          }"
          @click="r.has_explainer && expandRow(r.name)"
        >
          <td class="expand-cell" @click.stop="r.has_explainer && toggleExpand(r.name)">
            <v-icon
              v-if="r.has_explainer"
              :icon="expanded.has(r.name) ? 'mdi-chevron-down' : 'mdi-chevron-right'"
              size="small"
              color="primary"
              :title="expanded.has(r.name) ? 'Collapse' : 'Expand to see the circuit + code'"
              style="cursor: pointer"
            />
          </td>
          <td>
            <div class="d-flex align-center">
              <v-icon
                v-if="r.is_primary"
                icon="mdi-trophy"
                size="x-small"
                color="success"
                class="mr-1"
              />
              <code class="solver-name">{{ r.name }}</code>
            </div>
            <div class="text-caption text-medium-emphasis">
              <span v-if="r.version">v{{ r.version }}</span>
              <span v-if="r.hardware"> · {{ r.hardware }}</span>
            </div>
          </td>
          <td>
            <v-chip size="x-small" :color="r.tier_color" variant="tonal">
              {{ r.tier_label }}
            </v-chip>
          </td>
          <td class="text-right">
            <span
              v-if="r.status === 'complete'"
              :class="{ 'best-energy': r.energy === bestEnergy }"
            >
              {{ fmtEnergy(r.energy) }}
            </span>
            <span v-else class="text-medium-emphasis">—</span>
          </td>
          <td class="text-center">
            <v-icon
              v-if="r.status === 'complete' && r.feasible"
              icon="mdi-check-circle"
              color="success"
              size="small"
            />
            <v-icon
              v-else-if="r.status === 'complete' && !r.feasible"
              icon="mdi-alert-circle"
              color="warning"
              size="small"
            />
            <span v-else class="text-medium-emphasis">—</span>
          </td>
          <td class="text-right">
            <span
              v-if="r.elapsed_ms !== undefined && r.status === 'complete'"
              :class="{ 'best-time': r.elapsed_ms === fastestMs }"
            >
              {{ fmtTime(r.elapsed_ms) }}
            </span>
            <span v-else class="text-medium-emphasis">—</span>
          </td>
          <td>
            <v-chip
              v-if="r.status === 'complete'"
              size="x-small"
              color="success"
              variant="tonal"
            >
              complete
            </v-chip>
            <v-tooltip v-else-if="r.status === 'queued'" location="top">
              <template #activator="{ props: tipProps }">
                <v-chip
                  v-bind="tipProps"
                  size="x-small"
                  color="info"
                  variant="tonal"
                  prepend-icon="mdi-cloud-clock-outline"
                >
                  queued
                </v-chip>
              </template>
              <span>
                Submitted to {{ r.backend_name || 'cloud QPU' }}.
                We poll every 1-30 s (ramping up while we wait); this
                row fills in as soon as the cloud finishes.
              </span>
            </v-tooltip>
            <v-tooltip v-else-if="r.status === 'skipped'" location="top">
              <template #activator="{ props: tipProps }">
                <v-chip
                  v-bind="tipProps"
                  size="x-small"
                  color="grey"
                  variant="tonal"
                  prepend-icon="mdi-debug-step-over"
                >
                  skipped
                </v-chip>
              </template>
              <span>{{ r.error || 'Skipped: instance too large for this tier' }}</span>
            </v-tooltip>
            <v-tooltip v-else location="top">
              <template #activator="{ props: tipProps }">
                <v-chip
                  v-bind="tipProps"
                  size="x-small"
                  color="error"
                  variant="tonal"
                >
                  error
                </v-chip>
              </template>
              <span>{{ r.error || 'Unknown error' }}</span>
            </v-tooltip>
          </td>
        </tr>
        <tr
          v-if="r.has_explainer && expanded.has(r.name)"
          class="explainer-row"
          @click.stop
        >
          <td :colspan="7" class="pa-0" @click.stop>
            <QaoaExplainerPanel
              v-if="r.explainer_kind === 'qaoa'"
              :job="job"
              :extras="getExtras(r.name)!"
              :tier-color="r.tier_color"
              :sense="isMaximize ? 'maximize' : 'minimize'"
              :solver-status="r.status"
              :solver-name="r.name"
            />
            <ClassicalExplainerPanel
              v-else-if="r.explainer_kind === 'classical'"
              :solver-name="r.name"
              :result="r.solver_result"
              :sense="isMaximize ? 'maximize' : 'minimize'"
            />
          </td>
        </tr>
        </template>
      </tbody>
    </v-table>

    <div v-if="completedCount > 0" class="text-caption text-medium-emphasis mt-2">
      Best feasible energy drives the interpreted solution shown below.
      Hover the status chips for details.
    </div>
  </v-card>
</template>

<style scoped>
/* Vuetify's <v-table> renders a wrapper div, NOT a <table>, so the
 * `class="comparison-table"` ends up on the wrapper. We need :deep()
 * to reach the actual <table> for table-layout to take effect. */
.comparison-table :deep(table) {
  table-layout: fixed;
  width: 100%;
}
.comparison-table :deep(td),
.comparison-table :deep(th) {
  padding: 6px 8px;
  /* Allow long solver names / hardware strings to wrap rather than
   * forcing the column wide. */
  word-break: break-word;
}
.solver-name {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
}
.row-best {
  background: rgba(76, 175, 80, 0.06);
}
.row-error :deep(td),
.row-skipped :deep(td) {
  opacity: 0.7;
}
.row-clickable {
  cursor: pointer;
}
.row-clickable:hover {
  background: rgba(124, 58, 237, 0.05);
}
.expand-cell {
  width: 32px;
  padding-right: 0 !important;
}
.explainer-row :deep(td) {
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  padding: 0;
  /* table-layout:fixed on the parent table already prevents this td
   * from blowing past its column allocation, so wide children like
   * the gate-level QAOA circuit SVG can scroll horizontally inside
   * .circuit-scroll without dragging the whole row off-screen. */
}
.best-energy {
  color: rgb(var(--v-theme-success));
  font-weight: 600;
}
.best-time {
  color: rgb(var(--v-theme-info));
  font-weight: 500;
}
</style>
