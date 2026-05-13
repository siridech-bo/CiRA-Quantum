<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useBenchmarksStore } from '@/stores/benchmarks'

/**
 * Phase 9B+ — pending cloud-jobs panel.
 *
 * Polls `/api/benchmarks/cloud-jobs/pending` every 30 seconds. For
 * each job:
 *   - Shows current live cloud status (Computing / Completed / Error)
 *   - Auto-fires the materialize endpoint when `has_probs` flips
 *     true (which means the QPU returned a measurement distribution
 *     and we can build the RunRecord)
 *   - Surfaces cluster-side errors and offers a manual "Drop" action
 *
 * The panel is intentionally compact — it lives at the top of the
 * Findings page so users notice when QPU records show up without
 * needing to refresh anything.
 */

const benchmarks = useBenchmarksStore()

const POLL_INTERVAL_MS = 30_000

let timer: number | null = null
const materializing = ref<Set<string>>(new Set())
const recentlyMaterialized = ref<{ job_id: string; record_id: string; at: number }[]>([])
const recentErrors = ref<{ job_id: string; message: string; at: number }[]>([])

async function refreshOnce() {
  const before = new Map<string, string>()
  for (const j of benchmarks.pendingCloudJobs) {
    before.set(j.job_id, `${j.live_status}|${j.has_probs}`)
  }

  const pending = await benchmarks.loadPendingCloudJobs()

  // Auto-materialize any job whose has_probs just turned true.
  for (const j of pending) {
    const prev = before.get(j.job_id)
    const justFinished = j.has_probs && prev !== `${j.live_status}|true`
    if (justFinished && !materializing.value.has(j.job_id)) {
      void autoMaterialize(j.job_id)
    }
  }
}

async function autoMaterialize(jobId: string) {
  materializing.value.add(jobId)
  try {
    const result = await benchmarks.materializePendingJob(jobId)
    if (result.record_id) {
      recentlyMaterialized.value = [
        { job_id: jobId, record_id: result.record_id, at: Date.now() },
        ...recentlyMaterialized.value,
      ].slice(0, 5)
    } else if (result.error) {
      recentErrors.value = [
        { job_id: jobId, message: result.error, at: Date.now() },
        ...recentErrors.value,
      ].slice(0, 5)
    }
  } finally {
    materializing.value.delete(jobId)
  }
}

async function manualMaterialize(jobId: string) {
  await autoMaterialize(jobId)
}

async function manualDrop(jobId: string) {
  if (!confirm(`Drop pending job ${jobId.slice(0, 16)}...? This does not delete the cloud-side job, only the local pending entry.`)) {
    return
  }
  await benchmarks.dropPendingJob(jobId)
}

onMounted(() => {
  void refreshOnce()
  timer = window.setInterval(refreshOnce, POLL_INTERVAL_MS)
})

onBeforeUnmount(() => {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
})

function fmtAge(iso: string): string {
  const submitted = new Date(iso).getTime()
  const ageMs = Date.now() - submitted
  const min = Math.floor(ageMs / 60_000)
  if (min < 60) return `${min} min ago`
  const hr = Math.floor(min / 60)
  return `${hr}h ${min % 60}m ago`
}

function statusColor(status: string): string {
  const s = status.toUpperCase()
  if (s === 'COMPLETED') return 'success'
  if (s === 'COMPUTING') return 'primary'
  if (s.includes('ERROR') || s === 'FAILED') return 'error'
  return 'grey'
}

const isEmpty = computed(() =>
  !benchmarks.pendingCloudJobs.length &&
  !recentlyMaterialized.value.length &&
  !recentErrors.value.length &&
  !benchmarks.pendingCloudError
)
</script>

<template>
  <v-card
    v-if="!isEmpty"
    class="pa-4 mb-4"
    variant="tonal"
    :color="benchmarks.pendingCloudError ? 'warning' : 'primary'"
  >
    <div class="d-flex align-center mb-2">
      <v-icon icon="mdi-cloud-sync-outline" start />
      <span class="text-subtitle-1 font-weight-medium">
        Pending cloud jobs
      </span>
      <v-chip
        v-if="benchmarks.pendingCloudJobs.length"
        size="x-small"
        class="ml-2"
      >
        {{ benchmarks.pendingCloudJobs.length }}
      </v-chip>
      <v-spacer />
      <span class="text-caption text-medium-emphasis">
        auto-polling every {{ POLL_INTERVAL_MS / 1000 }} s
      </span>
      <v-btn icon="mdi-refresh" size="small" variant="text" class="ml-2" @click="refreshOnce" />
    </div>

    <!-- Soft error: e.g. no credential configured -->
    <v-alert
      v-if="benchmarks.pendingCloudError"
      type="warning"
      variant="tonal"
      density="compact"
      class="mb-2"
    >
      {{ benchmarks.pendingCloudError }}
    </v-alert>

    <!-- Pending list -->
    <v-table v-if="benchmarks.pendingCloudJobs.length" density="compact" class="pending-table">
      <thead>
        <tr>
          <th class="text-left">Job</th>
          <th class="text-left">Instance / solver</th>
          <th class="text-left">Submitted</th>
          <th class="text-left">Status</th>
          <th class="text-right">Actions</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="j in benchmarks.pendingCloudJobs" :key="j.job_id">
          <td>
            <code class="job-id">{{ j.job_id.slice(0, 16) }}…</code>
            <div v-if="j.notes" class="text-caption text-medium-emphasis" :title="j.notes">
              {{ j.notes.slice(0, 60) }}{{ j.notes.length > 60 ? '…' : '' }}
            </div>
          </td>
          <td>
            <code>{{ j.instance_id.split('/').pop() }}</code>
            <div class="text-caption text-medium-emphasis">{{ j.solver_name }}</div>
          </td>
          <td class="text-caption">{{ fmtAge(j.submitted_at) }}</td>
          <td>
            <v-chip
              :color="statusColor(j.live_status)"
              size="small"
              variant="tonal"
            >
              {{ j.live_status }}
            </v-chip>
            <div v-if="j.live_error" class="text-caption text-error" style="max-width: 280px">
              {{ j.live_error.slice(0, 90) }}{{ j.live_error.length > 90 ? '…' : '' }}
            </div>
            <div v-else-if="materializing.has(j.job_id)" class="text-caption text-primary">
              <v-icon icon="mdi-progress-clock" size="x-small" /> materializing…
            </div>
            <div v-else-if="j.has_probs" class="text-caption text-success">
              <v-icon icon="mdi-check" size="x-small" /> ready to materialize
            </div>
          </td>
          <td class="text-right text-no-wrap">
            <v-btn
              v-if="j.has_probs"
              size="x-small"
              color="success"
              variant="flat"
              :loading="materializing.has(j.job_id)"
              @click="manualMaterialize(j.job_id)"
            >
              Archive now
            </v-btn>
            <v-btn
              size="x-small"
              color="error"
              variant="text"
              @click="manualDrop(j.job_id)"
            >
              Drop
            </v-btn>
          </td>
        </tr>
      </tbody>
    </v-table>

    <!-- Recent successes -->
    <div v-if="recentlyMaterialized.length" class="mt-3">
      <div class="text-caption text-medium-emphasis mb-1">Recently archived:</div>
      <v-chip
        v-for="r in recentlyMaterialized"
        :key="r.job_id"
        size="small"
        color="success"
        variant="tonal"
        class="mr-1 mb-1"
        prepend-icon="mdi-check-circle"
      >
        {{ r.record_id.slice(-12) }}
      </v-chip>
    </div>

    <!-- Recent failures -->
    <div v-if="recentErrors.length" class="mt-2">
      <div class="text-caption text-medium-emphasis mb-1">Recent materialize failures:</div>
      <v-chip
        v-for="e in recentErrors"
        :key="e.job_id"
        size="small"
        color="error"
        variant="tonal"
        class="mr-1 mb-1"
        :title="e.message"
      >
        {{ e.job_id.slice(0, 12) }}…
      </v-chip>
    </div>
  </v-card>
</template>

<style scoped>
.pending-table {
  background: transparent !important;
}
.job-id {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
}
</style>
