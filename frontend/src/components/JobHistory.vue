<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSolveStore, type Job } from '@/stores/solve'

const solve = useSolveStore()
const router = useRouter()

const filterStatus = ref<'all' | 'complete' | 'error' | 'running'>('all')
const confirmDeleteId = ref<string | null>(null)
const refreshing = ref(false)

async function refresh() {
  refreshing.value = true
  try {
    await solve.loadHistory(1, 50)
  } finally {
    refreshing.value = false
  }
}

const isRunning = (s: string) =>
  !['complete', 'error'].includes(s)

function matches(j: Job) {
  if (filterStatus.value === 'all') return true
  if (filterStatus.value === 'running') return isRunning(j.status)
  return j.status === filterStatus.value
}

async function handleDelete(id: string) {
  await solve.deleteJob(id)
  confirmDeleteId.value = null
}

function open(job: Job) {
  router.push(`/jobs/${job.id}`)
}

function shortStatement(s: string): string {
  return s.length > 80 ? s.slice(0, 77) + '…' : s
}
function fmtDate(s: string): string {
  return new Date(s).toLocaleString()
}
function fmtSolveTime(ms: number | null | undefined): string {
  if (!ms) return '—'
  return `${(ms / 1000).toFixed(1)}s`
}

const STATUS_COLORS: Record<string, string> = {
  complete: 'success',
  error: 'error',
  queued: 'grey',
  formulating: 'primary',
  compiling: 'primary',
  validating: 'primary',
  solving: 'primary',
}

onMounted(() => {
  void refresh()
})

// Refresh whenever a solve finishes so the history table picks it up.
watch(
  () => solve.currentJob?.status,
  (s) => {
    if (s && ['complete', 'error'].includes(s)) void refresh()
  },
)
</script>

<template>
  <v-card class="pa-5">
    <div class="d-flex align-center mb-3">
      <v-card-title class="pa-0 text-h6 flex-grow-1">
        Your solve history
      </v-card-title>
      <v-btn
        size="small"
        variant="tonal"
        prepend-icon="mdi-refresh"
        :loading="refreshing"
        @click="refresh"
      >
        Refresh
      </v-btn>
    </div>

    <v-chip-group
      v-model="filterStatus"
      mandatory
      selected-class="bg-primary text-white"
      class="mb-2"
    >
      <v-chip value="all" size="small">All</v-chip>
      <v-chip value="running" size="small">Running</v-chip>
      <v-chip value="complete" size="small">Complete</v-chip>
      <v-chip value="error" size="small">Error</v-chip>
    </v-chip-group>

    <v-table density="comfortable">
      <thead>
        <tr>
          <th>Created</th>
          <th>Problem</th>
          <th>Provider</th>
          <th>Status</th>
          <th class="text-right">Vars</th>
          <th class="text-right">Time</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-if="!solve.history.length">
          <td colspan="7" class="text-center text-medium-emphasis py-6">
            No jobs yet — submit one from the Solve tab.
          </td>
        </tr>
        <tr
          v-for="job in solve.history.filter(matches)"
          :key="job.id"
          class="job-row"
          @click="open(job)"
        >
          <td>{{ fmtDate(job.created_at) }}</td>
          <td>{{ shortStatement(job.problem_statement) }}</td>
          <td>{{ job.provider }}</td>
          <td>
            <v-chip
              size="x-small"
              :color="STATUS_COLORS[job.status] || 'grey'"
              variant="tonal"
            >
              {{ job.status }}
            </v-chip>
          </td>
          <td class="text-right">{{ job.num_variables ?? '—' }}</td>
          <td class="text-right">{{ fmtSolveTime(job.solve_time_ms) }}</td>
          <td>
            <v-btn
              icon="mdi-delete-outline"
              size="x-small"
              variant="text"
              color="error"
              @click.stop="confirmDeleteId = job.id"
              :aria-label="`Delete job ${job.id}`"
            />
          </td>
        </tr>
      </tbody>
    </v-table>

    <v-dialog
      :model-value="!!confirmDeleteId"
      max-width="400"
      @update:model-value="(v: boolean) => { if (!v) confirmDeleteId = null }"
    >
      <v-card>
        <v-card-title>Delete this job?</v-card-title>
        <v-card-text>
          The job and its solver output will be removed. This can't be undone.
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn @click="confirmDeleteId = null">Cancel</v-btn>
          <v-btn
            color="error"
            variant="flat"
            @click="confirmDeleteId && handleDelete(confirmDeleteId)"
          >
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-card>
</template>

<style scoped>
.job-row {
  cursor: pointer;
}
.job-row:hover {
  background: rgba(255, 255, 255, 0.03);
}
</style>
