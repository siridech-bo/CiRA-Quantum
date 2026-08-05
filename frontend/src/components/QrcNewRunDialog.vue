<script setup lang="ts">
/**
 * QrcNewRunDialog — "New run" control from §0.D: pick a task + key config
 * fields, POST /api/qrc/runs. The backend enforces the actual config
 * allow-list (§3) and the single-active-job lock (409); this dialog just
 * needs to produce valid JSON in a shape a human can reason about, so it
 * exposes a per-task template in a JSON editor rather than guessing at
 * the exact allow-listed field names (unspecified in §3 beyond
 * `{task, config}`).
 */
import { computed, ref, watch } from 'vue'
import { useQrcStore, type QrcTask } from '@/stores/qrc'

const props = defineProps<{
  modelValue: boolean
}>()
const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'created', id: string): void
}>()

const qrc = useQrcStore()

const open = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v),
})

const TASKS: { value: QrcTask; title: string; hint: string }[] = [
  { value: 'trace-gen', title: 'trace-gen', hint: 'Generate + cache raw-FID traces (long-running, GPU).' },
  { value: 'narma', title: 'narma', hint: 'NARMA benchmark from a cached (or fresh) trace.' },
  { value: 'weather', title: 'weather', hint: 'Weather forecast benchmark from a cached (or fresh) trace.' },
  { value: 'phase1', title: 'phase1', hint: 'Phase-1 feature-method sweep on a cached trace (fast — no re-evolution).' },
]

// Keys MUST match the backend launcher allow-list (app/qrc/launcher.py _TASKS);
// any extra key is rejected with HTTP 400. `subtask` chooses the trace-gen
// target; `trace` (phase1) is a cached-trace NAME → artifacts/traces/<name>.npz.
// Legacy quick-launch dialog (the full /qrc/new page is now the canonical setup).
// Only the four original tasks are offered here; Partial<> because QrcTask has
// since grown to include the schema-driven tasks (memory/learnable/phase2/memcap).
const DEFAULT_CONFIGS: Partial<Record<QrcTask, Record<string, any>>> = {
  'trace-gen': {
    subtask: 'weather',
    fid_points: 2048,
    n_peaks: 653,
    tau: 0.03,
    n_virtual: 25,
    splits: [374, 600, 500],
    horizons: [1, 10, 20, 30, 45],
    seed: 42,
  },
  narma: {
    fid_points: 2048,
    n_peaks: 653,
    orders: [2, 5, 10, 15, 20],
    seed: 42,
  },
  weather: {
    fid_points: 2048,
    n_peaks: 653,
    washout: 374,
    n_train: 600,
    n_test: 500,
    horizons: [1, 5, 10, 15, 20, 30, 45],
    seed: 42,
  },
  phase1: {
    trace: 'weather',
    n_peaks: 653,
    select: 'first',
    key_horizon: 30,
  },
}

const task = ref<QrcTask>('phase1')
const configText = ref(JSON.stringify(DEFAULT_CONFIGS[task.value], null, 2))
const jsonError = ref<string | null>(null)
const submitting = ref(false)
const submitError = ref<string | null>(null)

watch(task, (t) => {
  configText.value = JSON.stringify(DEFAULT_CONFIGS[t], null, 2)
  jsonError.value = null
})

watch(open, (v) => {
  if (v) {
    submitError.value = null
    jsonError.value = null
  }
})

function validateJson(): Record<string, any> | null {
  try {
    const parsed = JSON.parse(configText.value)
    jsonError.value = null
    return parsed
  } catch (e: any) {
    jsonError.value = `Invalid JSON: ${e?.message || 'parse error'}`
    return null
  }
}

async function submit() {
  const config = validateJson()
  if (!config) return
  submitting.value = true
  submitError.value = null
  try {
    const { id } = await qrc.createRun(task.value, config)
    open.value = false
    emit('created', id)
  } catch (e: any) {
    submitError.value = e?.message || 'Failed to launch run'
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <v-dialog v-model="open" max-width="560">
    <v-card>
      <v-card-title>
        <v-icon icon="mdi-play-circle" class="mr-2" />
        New QRC run
      </v-card-title>
      <v-card-text>
        <v-select
          v-model="task"
          :items="TASKS"
          item-title="title"
          item-value="value"
          label="Task"
          class="mb-1"
        />
        <div class="text-caption text-medium-emphasis mb-3">
          {{ TASKS.find((t) => t.value === task)?.hint }}
        </div>

        <v-textarea
          v-model="configText"
          label="Config (JSON)"
          rows="9"
          spellcheck="false"
          class="config-editor"
          hide-details="auto"
          :error="!!jsonError"
          :error-messages="jsonError || undefined"
          @blur="validateJson()"
        />

        <v-alert
          type="warning"
          variant="tonal"
          density="compact"
          class="mt-3"
          icon="mdi-lock-alert-outline"
        >
          Only one heavy QRC job may run at a time. The server rejects
          this launch with a 409 if a run is already active — stop it
          first from the dashboard.
        </v-alert>

        <v-alert
          v-if="submitError"
          type="error"
          variant="tonal"
          density="compact"
          class="mt-3"
        >{{ submitError }}</v-alert>
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" :disabled="submitting" @click="open = false">Cancel</v-btn>
        <v-btn
          color="primary"
          variant="flat"
          :loading="submitting"
          prepend-icon="mdi-rocket-launch"
          @click="submit"
        >Launch run</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.config-editor :deep(textarea) {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.8rem;
}
</style>
