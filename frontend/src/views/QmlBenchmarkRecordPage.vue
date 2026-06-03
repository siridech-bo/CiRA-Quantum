<script setup lang="ts">
/**
 * QmlBenchmarkRecordPage — full detail of one archived TrainRecord.
 *
 * Reuses the same charts as the live job-detail page (loss curve,
 * baseline comparison) so a student walking the archive sees the
 * same shape they'd see while training. Adds the citation block + a
 * "copy BibTeX" button.
 */
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'
import { useAuthStore } from '@/stores/auth'
import CiraLogo from '@/components/CiraLogo.vue'
import TrainingLossChart from '@/components/TrainingLossChart.vue'
import BaselineComparison from '@/components/BaselineComparison.vue'
import type { QmlBaseline } from '@/stores/qml'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const api = axios.create({ withCredentials: true })

interface QpuRunInRecord {
  id: string
  provider: string
  backend_name: string | null
  shots: number | null
  status: string
  cloud_job_id: string | null
  wall_time_ms: number | null
  metrics: any
}

interface FullRecord {
  record_id: string
  code_version: string
  repro_hash: string
  model: string
  dataset_id: string
  hyperparameters: Record<string, any>
  hardware_id: string
  started_at: string
  completed_at: string
  metrics: Record<string, any>
  baselines: QmlBaseline[]
  training_history: { epoch: number; loss: number; train_accuracy: number; test_accuracy: number }[]
  qpu_runs: QpuRunInRecord[]
  contributor_display_name: string
  notes: string
  warnings: string[]
}

const recordId = computed(() => route.params.id as string)
const record = ref<FullRecord | null>(null)
const bibtex = ref<string | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)
const copyState = ref<'idle' | 'copied' | 'failed'>('idle')
const deleting = ref(false)
const deleteError = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    const [r1, r2] = await Promise.all([
      api.get<FullRecord>(`/api/qml/benchmarks/${recordId.value}`),
      api.get(`/api/qml/benchmarks/${recordId.value}/cite`, { responseType: 'text' }),
    ])
    record.value = r1.data
    bibtex.value = typeof r2.data === 'string' ? r2.data : String(r2.data)
  } catch (e: any) {
    error.value =
      e?.response?.status === 404
        ? 'Record not found in the archive.'
        : e?.response?.data?.error || e?.message || 'Failed to load record'
  } finally {
    loading.value = false
  }
}

onMounted(load)

async function copyCite() {
  if (!bibtex.value) return
  try {
    await navigator.clipboard.writeText(bibtex.value)
    copyState.value = 'copied'
    setTimeout(() => { copyState.value = 'idle' }, 1800)
  } catch {
    copyState.value = 'failed'
    setTimeout(() => { copyState.value = 'idle' }, 1800)
  }
}

async function deleteRecord() {
  if (!record.value) return
  if (!confirm(
    'Delete this archived record? The on-disk JSON will be removed. '
    + 'This is meant for bad submissions / leaked tokens, not routine cleanup.'
  )) return
  deleting.value = true
  deleteError.value = null
  try {
    await api.delete(`/api/qml/benchmarks/${record.value.record_id}`)
    router.push('/qml/benchmarks')
  } catch (e: any) {
    deleteError.value =
      e?.response?.data?.error || e?.message || 'Delete failed'
  } finally {
    deleting.value = false
  }
}

const vqcMetricsForCompare = computed(() => {
  if (!record.value) return null
  return {
    final_train_accuracy: record.value.metrics.final_train_accuracy ?? 0,
    final_test_accuracy: record.value.metrics.final_test_accuracy ?? 0,
    final_loss: record.value.metrics.final_loss ?? 0,
    confusion_matrix: record.value.metrics.confusion_matrix ?? [[0, 0], [0, 0]],
    weights: [],
    bias: 0,
    n_qubits: record.value.metrics.n_qubits ?? 0,
    pca_applied: record.value.metrics.pca_applied ?? false,
    classes: record.value.metrics.classes ?? ['class_0', 'class_1'],
    feature_names: record.value.metrics.feature_names ?? [],
    notes: record.value.metrics.notes ?? [],
    train_time_ms: record.value.metrics.train_time_ms ?? 0,
    circuit_info: record.value.metrics.circuit_info,
  }
})

function fmtAcc(p: any) {
  if (typeof p !== 'number') return '—'
  return (p * 100).toFixed(1) + '%'
}
function fmtDate(iso: string) {
  try { return new Date(iso).toLocaleString() } catch { return iso }
}
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QML benchmark record">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— archive</span>
    </div>
    <v-spacer />
    <v-btn variant="text" @click="router.push('/qml/benchmarks')">All records</v-btn>
    <v-btn variant="text" @click="router.push('/qml')">Datasets</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />
      <v-alert v-else-if="error" type="error" variant="tonal">{{ error }}</v-alert>

      <template v-else-if="record">
        <!-- Header -->
        <div class="d-flex align-center flex-wrap ga-3 mb-3">
          <v-btn
            icon="mdi-arrow-left"
            variant="text"
            @click="router.push('/qml/benchmarks')"
            aria-label="Back to archive"
          />
          <div class="flex-grow-1">
            <div class="text-overline text-medium-emphasis">
              archive record
            </div>
            <div class="text-h5">
              {{ record.model }} on
              <code>{{ record.dataset_id }}</code>
            </div>
          </div>
          <v-chip
            color="success"
            variant="flat"
            prepend-icon="mdi-check-decagram"
          >archived</v-chip>
        </div>

        <!-- Identity strip -->
        <v-card variant="tonal" class="pa-3 mb-4">
          <v-row dense>
            <v-col cols="12" sm="4">
              <div class="text-caption text-medium-emphasis">contributor</div>
              <div class="text-subtitle-2">{{ record.contributor_display_name }}</div>
            </v-col>
            <v-col cols="12" sm="4">
              <div class="text-caption text-medium-emphasis">completed</div>
              <div class="text-subtitle-2">{{ fmtDate(record.completed_at) }}</div>
            </v-col>
            <v-col cols="12" sm="4">
              <div class="text-caption text-medium-emphasis">repro hash</div>
              <code class="text-body-2">{{ record.repro_hash }}</code>
            </v-col>
            <v-col cols="12" sm="4">
              <div class="text-caption text-medium-emphasis">code version</div>
              <code class="text-caption">{{ record.code_version.slice(0, 16) }}…</code>
            </v-col>
            <v-col cols="12" sm="8">
              <div class="text-caption text-medium-emphasis">classical host</div>
              <div class="text-caption text-medium-emphasis">{{ record.hardware_id }}</div>
            </v-col>
          </v-row>
        </v-card>

        <!-- Headline metrics -->
        <v-row dense class="mb-2">
          <v-col cols="6" sm="3">
            <v-card variant="outlined" class="pa-3 text-center">
              <div class="text-caption text-medium-emphasis">test accuracy</div>
              <div class="text-h5 text-success">
                {{ fmtAcc(record.metrics.final_test_accuracy) }}
              </div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="outlined" class="pa-3 text-center">
              <div class="text-caption text-medium-emphasis">train accuracy</div>
              <div class="text-h5">
                {{ fmtAcc(record.metrics.final_train_accuracy) }}
              </div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="outlined" class="pa-3 text-center">
              <div class="text-caption text-medium-emphasis">final loss</div>
              <div class="text-h5">{{ (record.metrics.final_loss ?? 0).toFixed(4) }}</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="outlined" class="pa-3 text-center">
              <div class="text-caption text-medium-emphasis">qubits / layers</div>
              <div class="text-h5">
                {{ record.metrics.n_qubits ?? '?' }} /
                {{ record.hyperparameters.n_layers ?? '?' }}
              </div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Training history -->
        <v-card variant="outlined" class="pa-4 mb-4">
          <div class="text-subtitle-1 mb-2">
            <v-icon icon="mdi-chart-line" class="mr-1" />
            Training history
          </div>
          <TrainingLossChart
            v-if="record.training_history.length"
            :history="record.training_history"
            :total-epochs="record.hyperparameters.n_epochs ?? null"
          />
          <v-alert
            v-else
            type="info"
            variant="tonal"
            density="compact"
          >No training history recorded.</v-alert>
        </v-card>

        <!-- Baselines comparison -->
        <BaselineComparison
          v-if="record.baselines.length && vqcMetricsForCompare"
          :vqc="vqcMetricsForCompare as any"
          :baselines="record.baselines"
          class="mb-4"
        />

        <!-- QPU runs -->
        <v-card
          v-if="record.qpu_runs.length"
          variant="outlined"
          class="pa-4 mb-4"
        >
          <div class="text-subtitle-1 mb-2">
            <v-icon icon="mdi-atom" class="mr-1" color="error" />
            Real-QPU evaluations ({{ record.qpu_runs.length }})
          </div>
          <div
            v-for="q in record.qpu_runs"
            :key="q.id"
            class="qpu-row mb-2"
          >
            <v-chip
              size="small"
              :color="q.provider === 'originqc' ? 'warning' : 'error'"
              variant="flat"
              class="mr-2"
            >{{ q.provider }}</v-chip>
            <span class="text-body-2">
              {{ q.backend_name || '—' }} · {{ q.shots ?? '?' }} shots
            </span>
            <span
              v-if="q.metrics && q.metrics.test_accuracy != null"
              class="ml-3 text-body-2"
            >
              test acc <strong>{{ fmtAcc(q.metrics.test_accuracy) }}</strong>
            </span>
            <span
              v-else-if="q.metrics && q.metrics.predicted_label != null"
              class="ml-3 text-body-2"
            >
              point #{{ q.metrics.sample_index }} →
              class <strong>{{ q.metrics.predicted_label }}</strong>
              ({{ q.metrics.correct ? '✓' : '✗' }})
            </span>
          </div>
        </v-card>

        <!-- Hyperparameters dump -->
        <v-card variant="outlined" class="pa-4 mb-4">
          <div class="text-subtitle-1 mb-2">
            <v-icon icon="mdi-tune-variant" class="mr-1" />
            Hyperparameters
          </div>
          <pre class="hp-dump">{{ JSON.stringify(record.hyperparameters, null, 2) }}</pre>
        </v-card>

        <!-- Notes (if any) -->
        <v-alert
          v-if="record.notes"
          type="info"
          variant="tonal"
          icon="mdi-note-text"
          class="mb-4"
        >
          <div class="text-subtitle-2 mb-1">Notes from the archiver</div>
          <div class="text-body-2">{{ record.notes }}</div>
        </v-alert>

        <!-- Citation -->
        <v-card variant="outlined" class="pa-4 mb-4">
          <div class="d-flex align-center mb-2">
            <v-icon icon="mdi-format-quote-close" class="mr-2" />
            <div class="text-subtitle-1 flex-grow-1">Cite this record</div>
            <v-btn
              :color="copyState === 'copied' ? 'success' : 'primary'"
              variant="flat"
              size="small"
              :prepend-icon="
                copyState === 'copied' ? 'mdi-check' :
                copyState === 'failed' ? 'mdi-alert' : 'mdi-content-copy'
              "
              @click="copyCite"
            >
              {{
                copyState === 'copied'
                  ? 'Copied!'
                  : copyState === 'failed'
                    ? 'Copy failed'
                    : 'Copy BibTeX'
              }}
            </v-btn>
          </div>
          <pre class="bibtex">{{ bibtex }}</pre>
        </v-card>

        <!-- Admin-only delete -->
        <v-card
          v-if="auth.user && auth.user.role === 'admin'"
          variant="outlined"
          class="pa-4 mb-8"
          border="error"
        >
          <div class="text-subtitle-2 mb-2">
            <v-icon icon="mdi-shield-account" size="small" class="mr-1" />
            Admin
          </div>
          <div class="text-body-2 text-medium-emphasis mb-2">
            Records are append-only by convention. Delete only for bad
            submissions or leaked tokens.
          </div>
          <v-alert
            v-if="deleteError"
            type="error"
            variant="tonal"
            density="compact"
            class="mb-2"
          >{{ deleteError }}</v-alert>
          <v-btn
            color="error"
            variant="outlined"
            prepend-icon="mdi-delete"
            :loading="deleting"
            @click="deleteRecord"
          >Delete this record</v-btn>
        </v-card>
      </template>
    </v-container>
  </v-main>
</template>

<style scoped>
.logo-link {
  cursor: pointer;
  transition: opacity 0.15s ease-in-out;
}
.logo-link:hover {
  opacity: 0.8;
}
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
.bibtex {
  background: rgba(255, 255, 255, 0.05);
  padding: 12px;
  border-radius: 4px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.82rem;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-x: auto;
  margin: 0;
}
.hp-dump {
  background: rgba(255, 255, 255, 0.04);
  padding: 10px;
  border-radius: 4px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.82rem;
  margin: 0;
  white-space: pre-wrap;
  overflow-x: auto;
}
.qpu-row {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 4px;
  padding: 8px 12px;
}
</style>
