<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted } from 'vue'
import { useSolveStore, type JobStatus } from '@/stores/solve'

const solve = useSolveStore()

const STAGES: { key: JobStatus; label: string; hint: string }[] = [
  { key: 'formulating', label: 'Formulating', hint: 'LLM is writing the CQM' },
  { key: 'compiling',   label: 'Compiling',   hint: 'JSON → dimod.CQM' },
  { key: 'validating',  label: 'Validating',  hint: 'Oracle + constraint checks' },
  { key: 'solving',     label: 'Solving',     hint: 'GPU SA on the BQM lowering' },
  { key: 'complete',    label: 'Done',        hint: 'Interpreting the result' },
]

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
      </v-timeline-item>
    </v-timeline>

    <v-alert
      v-if="isError"
      type="error"
      variant="tonal"
      class="mt-4"
      :title="`Pipeline failed during ${STAGES[currentIndex]?.label ?? 'unknown'}`"
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
</style>
