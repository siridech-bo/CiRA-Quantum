<script setup lang="ts">
/**
 * QpuRunPanel — "Run on real hardware" surface, now multi-provider.
 *
 * Two cloud paths share a panel because the pedagogy is the same:
 * locally-trained weights → real QPU → see shot noise. Each provider's
 * differences (IBM batches all test points, Origin evaluates one) are
 * surfaced in copy + result rendering, not hidden behind an
 * abstraction.
 *
 * IBM (QML-5): batch evaluation over the full test set, returns a
 *   single test-accuracy + confusion matrix. Compared row-by-row
 *   against the simulator.
 * Origin (QML-6): one representative test point per submission,
 *   returns the predicted class probability + correctness for that
 *   point. Multiple Origin runs build up an empirical accuracy.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useQmlStore, type QmlQpuRun } from '@/stores/qml'

const props = defineProps<{
  parentJobId: string
  /** Whether the parent has a 2D test split (required for the QML-5/6 cut). */
  hasTestSplit: boolean
  /** Optional: VQC simulator accuracy, for the "compare vs simulator" row. */
  simulatorTestAccuracy?: number | null
}>()

const router = useRouter()
const auth = useAuthStore()
const qml = useQmlStore()

const provider = ref<'ibmq' | 'originqc'>('ibmq')
const submitting = ref(false)
const submitError = ref<string | null>(null)
/** Tracks which provider produced the current submitError. When the
 *  user switches tabs we drop a stale error from the other provider so
 *  it doesn't look like the new path failed. */
const submitErrorProvider = ref<'ibmq' | 'originqc' | null>(null)

watch(provider, () => {
  // Drop any error from the previous tab. The user is starting fresh
  // on the new provider — showing a leftover red alert would be
  // misleading (see QML-6 follow-up bug report).
  submitError.value = null
  submitErrorProvider.value = null
})
const form = ref({
  ibmShots: 1024,
  ibmBackend: '',
  originShots: 2048,
  originBackend: 'full_amplitude',
  originSampleIndex: null as number | null,
})

const ibmReady = computed(() => qml.capabilities?.qiskit_ibm_runtime === true)
// pyqpanda3 is the same dep that powers qaoa_originqc on the optimization
// side — if the optimization app has the [quantum] extra installed, this
// path is available too. The capabilities endpoint does not surface it
// yet (it pre-dated QML-6), so default to "assume yes if IBM is too" and
// rely on the backend's 503 to flag misconfiguration.
const originReady = computed(() => true)

const runs = computed(() => qml.qpuRuns)
const anyRunning = computed(() =>
  runs.value.some((r) => r.status !== 'complete' && r.status !== 'error'),
)

onMounted(async () => {
  await qml.loadQpuRuns(props.parentJobId)
  if (anyRunning.value) qml.startQpuPolling()
})

onBeforeUnmount(() => {
  qml.stopQpuPolling()
})

async function submit() {
  submitting.value = true
  submitError.value = null
  submitErrorProvider.value = null
  const submittedProvider = provider.value
  try {
    if (submittedProvider === 'ibmq') {
      await qml.submitQpuRun(props.parentJobId, {
        provider: 'ibmq',
        shots: form.value.ibmShots,
        backend_name: form.value.ibmBackend || null,
      })
    } else {
      await qml.submitQpuRun(props.parentJobId, {
        provider: 'originqc',
        shots: form.value.originShots,
        backend_name: form.value.originBackend || null,
        sample_index: form.value.originSampleIndex,
      })
    }
    qml.startQpuPolling()
  } catch (e: any) {
    submitError.value =
      e?.response?.data?.error || e?.message || 'Submission failed'
    submitErrorProvider.value = submittedProvider
  } finally {
    submitting.value = false
  }
}

function statusColor(s: QmlQpuRun['status']) {
  switch (s) {
    case 'complete':   return 'success'
    case 'error':      return 'error'
    case 'running':    return 'primary'
    case 'submitted':  return 'info'
    case 'submitting': return 'info'
    case 'queued':     return 'grey'
    default:           return 'grey'
  }
}

function parseMetrics(run: QmlQpuRun) {
  if (!run.metrics) return null
  try { return JSON.parse(run.metrics) } catch { return null }
}

function parseCtx(run: QmlQpuRun) {
  if (!run.submission_context) return null
  try { return JSON.parse(run.submission_context) } catch { return null }
}

function accuracyDelta(qpuAcc: number) {
  if (props.simulatorTestAccuracy == null) return null
  return qpuAcc - props.simulatorTestAccuracy
}

function providerLabel(p: string) {
  return p === 'originqc' ? 'Origin Quantum' : 'IBM Quantum'
}

function providerColor(p: string) {
  return p === 'originqc' ? 'warning' : 'error'
}
</script>

<template>
  <v-card variant="outlined" class="pa-4 qpu-panel">
    <!-- Header -->
    <div class="d-flex align-center mb-2 flex-wrap">
      <v-icon icon="mdi-atom" color="error" class="mr-2" />
      <div class="text-subtitle-1 flex-grow-1">
        Run on real hardware
      </div>
      <v-chip
        v-if="anyRunning"
        size="small"
        color="primary"
        variant="flat"
      >polling…</v-chip>
    </div>

    <div class="text-body-2 text-medium-emphasis mb-3">
      Two cloud paths share this panel — IBM Quantum and Origin Quantum.
      Both bake your trained parameters into a circuit on the chosen
      backend and submit. Real QPUs return shot-noisy probabilities;
      the cloud simulators (Origin's <code>full_amplitude</code>) return
      exact distributions but exercise the same dispatch path.
    </div>

    <!-- Provider switcher -->
    <v-tabs v-model="provider" color="primary" align-tabs="start" class="mb-2">
      <v-tab value="ibmq">
        <v-icon icon="mdi-atom-variant" start color="error" />
        IBM Quantum
      </v-tab>
      <v-tab value="originqc">
        <v-icon icon="mdi-atom-variant" start color="warning" />
        Origin Quantum
      </v-tab>
    </v-tabs>

    <!-- 2D-split gate (applies to both providers) -->
    <v-alert
      v-if="!hasTestSplit"
      type="info"
      variant="tonal"
      density="compact"
      class="mb-3"
      icon="mdi-information-outline"
    >
      This job doesn't have a persisted test split, so real-QPU
      inference can't reconstruct the test data. This applies to
      jobs trained before 2026-07-02 (only the 2D scatter for the
      boundary plot was saved back then). Re-train to enable the
      real-hardware path — the new pipeline persists the full
      N-dimensional test split for any qubit count up to Wukong's 12.
    </v-alert>

    <v-window v-else v-model="provider">
      <!-- ============ IBM Quantum ============ -->
      <v-window-item value="ibmq">
        <v-alert
          v-if="!ibmReady"
          type="warning"
          variant="tonal"
          density="compact"
          class="mb-3"
        >
          <code>qiskit-ibm-runtime</code> is not installed on the server.
          Install <code>pip install ".[ibm-quantum]"</code>.
        </v-alert>
        <template v-else>
          <v-alert
            type="info"
            variant="tonal"
            density="compact"
            class="mb-3"
            icon="mdi-key"
          >
            Uses your <strong>ibm_quantum</strong> BYOK key.
            <strong>Batch mode:</strong> all test points are submitted
            as one PUB and the run reports a full test accuracy.
          </v-alert>
          <v-row dense>
            <v-col cols="12" sm="4">
              <v-text-field
                v-model.number="form.ibmShots"
                label="Shots"
                type="number"
                :min="1" :max="4096"
                density="comfortable"
                hide-details="auto"
                hint="more shots = lower noise; 1024 is a good default"
                persistent-hint
              />
            </v-col>
            <v-col cols="12" sm="8">
              <v-text-field
                v-model="form.ibmBackend"
                label="Backend (leave blank for least-busy)"
                placeholder="e.g. ibm_brisbane, ibm_kingston"
                density="comfortable"
                hide-details="auto"
                hint="must be accessible from your IBM Quantum account"
                persistent-hint
              />
            </v-col>
          </v-row>
        </template>
      </v-window-item>

      <!-- ============ Origin Quantum ============ -->
      <v-window-item value="originqc">
        <v-alert
          type="info"
          variant="tonal"
          density="compact"
          class="mb-3"
          icon="mdi-key"
        >
          Uses your <strong>originqc</strong> BYOK key — the same one
          <code>qaoa_originqc</code> uses.
          <strong>Single-point mode:</strong> Origin's cloud API
          submits one program per call, so each run evaluates one
          representative test point. Launch multiple runs to build
          intuition for the shot-noise distribution.
        </v-alert>
        <v-row dense>
          <v-col cols="12" sm="4">
            <v-text-field
              v-model.number="form.originShots"
              label="Shots"
              type="number"
              :min="1" :max="8192"
              density="comfortable"
              hide-details="auto"
              hint="2048 is the typical default on full_amplitude"
              persistent-hint
            />
          </v-col>
          <v-col cols="12" sm="4">
            <v-select
              v-model="form.originBackend"
              label="Backend"
              :items="[
                { title: 'full_amplitude (cloud simulator, fast)', value: 'full_amplitude' },
                { title: 'partial_amplitude (cloud simulator)', value: 'partial_amplitude' },
                { title: 'WK_C180 (real QPU — Wukong)', value: 'WK_C180' },
              ]"
              density="comfortable"
              hide-details="auto"
            />
          </v-col>
          <v-col cols="12" sm="4">
            <v-text-field
              v-model.number="form.originSampleIndex"
              label="Sample index (blank = random)"
              type="number"
              :min="0"
              density="comfortable"
              hide-details="auto"
              hint="pick a specific test point or leave blank"
              persistent-hint
            />
          </v-col>
        </v-row>
        <v-alert
          v-if="form.originBackend === 'WK_C180'"
          type="warning"
          variant="tonal"
          density="compact"
          class="mt-3"
          icon="mdi-shield-alert"
        >
          <code>WK_C180</code> is a real superconducting QPU. The
          server must have <code>ENABLE_ORIGIN_REAL_HARDWARE=1</code>
          set for the submission to go through. Queue times can run
          to tens of minutes.
        </v-alert>
      </v-window-item>
    </v-window>

    <v-alert
      v-if="submitError"
      type="error"
      variant="tonal"
      density="compact"
      class="mt-3"
    >
      <div class="text-subtitle-2 mb-1" v-if="submitErrorProvider">
        {{ providerLabel(submitErrorProvider) }} returned an error:
      </div>
      <div>{{ submitError }}</div>
      <div
        v-if="
          submitErrorProvider === 'ibmq'
          && /no backend matches/i.test(submitError)
        "
        class="text-caption mt-2 text-medium-emphasis"
      >
        Your IBM Quantum instance has no real QPU available right now
        (open-plan access varies by time of day + region). Try again in
        a minute, or specify a backend name explicitly
        (e.g. <code>ibm_brisbane</code>, <code>ibm_kingston</code>)
        if you know one your account can access.
      </div>
    </v-alert>

    <div v-if="hasTestSplit" class="d-flex align-center mt-3">
      <v-spacer />
      <v-btn
        :color="providerColor(provider)"
        variant="flat"
        :loading="submitting"
        :disabled="!auth.user || (provider === 'ibmq' && !ibmReady)"
        prepend-icon="mdi-atom"
        @click="submit"
      >
        {{ runs.length ? 'Submit another run' : `Submit to ${providerLabel(provider)}` }}
      </v-btn>
    </div>

    <!-- Run cards (any provider) -->
    <template v-if="runs.length">
      <v-divider class="my-4" />
      <div class="text-overline text-medium-emphasis mb-2">
        Runs ({{ runs.length }})
      </div>
      <div v-for="run in runs" :key="run.id" class="qpu-run mb-3">
        <div class="d-flex align-center mb-1 flex-wrap">
          <v-chip
            size="small"
            :color="providerColor(run.provider)"
            variant="flat"
            class="mr-2"
          >{{ providerLabel(run.provider) }}</v-chip>
          <v-chip
            size="small"
            :color="statusColor(run.status)"
            variant="flat"
            class="mr-2"
          >{{ run.status }}</v-chip>
          <span class="text-body-2 text-medium-emphasis">
            {{ run.backend_name || '(picking…)' }} · {{ run.shots }} shots
          </span>
          <v-spacer />
          <span class="text-caption text-medium-emphasis">
            id {{ run.id.slice(0, 8) }}
          </span>
        </div>

        <!-- Non-terminal: queue status -->
        <template v-if="run.status !== 'complete' && run.status !== 'error'">
          <v-progress-linear
            indeterminate
            color="primary"
            height="4"
            rounded
            class="mb-2"
          />
          <div class="text-caption text-medium-emphasis">
            <span v-if="run.live_status">
              cloud status: <code>{{ run.live_status }}</code>
            </span>
            <span v-if="run.queue_position !== null" class="ml-2">
              · queue position: <strong>{{ run.queue_position }}</strong>
            </span>
            <span v-if="run.cloud_job_id" class="ml-2">
              · job <code>{{ run.cloud_job_id.slice(0, 12) }}…</code>
            </span>
            <span v-if="!run.live_status && !run.cloud_job_id">
              Submitting…
            </span>
          </div>
        </template>

        <!-- Error -->
        <v-alert
          v-else-if="run.status === 'error'"
          type="error"
          variant="tonal"
          density="compact"
          class="mt-2"
        >{{ run.error || 'Unknown cloud error' }}</v-alert>

        <!-- Complete -->
        <template v-else-if="run.status === 'complete'">
          <div v-if="parseMetrics(run)" class="run-complete">
            <!-- IBM batch result -->
            <template v-if="parseMetrics(run).mode === 'batch'">
              <v-row dense>
                <v-col cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">QPU test accuracy</div>
                  <div class="text-h6 text-error">
                    {{ (parseMetrics(run).test_accuracy * 100).toFixed(1) }}%
                  </div>
                </v-col>
                <v-col v-if="simulatorTestAccuracy != null" cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">vs simulator</div>
                  <div class="text-h6">
                    <span
                      :class="
                        accuracyDelta(parseMetrics(run).test_accuracy)! >= 0
                          ? 'text-success' : 'text-warning'
                      "
                    >
                      {{ accuracyDelta(parseMetrics(run).test_accuracy)! >= 0 ? '+' : '' }}
                      {{ (accuracyDelta(parseMetrics(run).test_accuracy)! * 100).toFixed(1) }}%
                    </span>
                  </div>
                </v-col>
                <v-col cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">wall time</div>
                  <div class="text-subtitle-1">
                    {{ run.wall_time_ms != null
                      ? ((run.wall_time_ms / 1000).toFixed(1) + ' s')
                      : '?' }}
                  </div>
                </v-col>
                <v-col cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">shots/circuit</div>
                  <div class="text-subtitle-1">{{ parseMetrics(run).shots }}</div>
                </v-col>
              </v-row>
              <div class="text-caption text-medium-emphasis mt-2">
                Simulator above is exact (statevector). The QPU number
                carries shot noise + device error — typically a 1–10%
                accuracy drop on a 2-qubit circuit at 1024 shots; real
                Heron devices can be worse.
              </div>
            </template>

            <!-- Origin single-point result -->
            <template v-else-if="parseMetrics(run).mode === 'single_point'">
              <v-row dense>
                <v-col cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">test point</div>
                  <div class="text-subtitle-1">
                    #{{ parseMetrics(run).sample_index }}
                    <v-chip
                      size="x-small"
                      class="ml-1"
                      variant="tonal"
                    >true: class {{ parseMetrics(run).sample_true_label }}</v-chip>
                  </div>
                </v-col>
                <v-col cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">predicted</div>
                  <div class="text-h6">
                    class {{ parseMetrics(run).predicted_label }}
                    <v-icon
                      :icon="parseMetrics(run).correct ? 'mdi-check-circle' : 'mdi-close-circle'"
                      :color="parseMetrics(run).correct ? 'success' : 'error'"
                      size="small"
                      class="ml-1"
                    />
                  </div>
                </v-col>
                <v-col cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">P(class 1)</div>
                  <div class="text-subtitle-1">
                    {{ (parseMetrics(run).prob_class1 * 100).toFixed(1) }}%
                  </div>
                </v-col>
                <v-col cols="6" sm="3">
                  <div class="text-caption text-medium-emphasis">⟨Z⟩ on q₀</div>
                  <div class="text-subtitle-1">
                    {{ parseMetrics(run).expect_z.toFixed(3) }}
                  </div>
                </v-col>
              </v-row>
              <div class="text-caption text-medium-emphasis mt-2">
                Single shot-noisy evaluation of test point
                #{{ parseMetrics(run).sample_index }}. To build an
                empirical accuracy, run several Origin submissions on
                different sample indices.
              </div>
            </template>
          </div>
        </template>
      </div>
    </template>
  </v-card>
</template>

<style scoped>
.qpu-panel {
  border-color: rgba(239, 68, 68, 0.4);
}
.qpu-run {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 6px;
  padding: 12px;
}
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
</style>
