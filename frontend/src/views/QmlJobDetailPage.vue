<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'
import { useAuthStore } from '@/stores/auth'
import { useQmlStore, type QmlCircuitInfo, type QmlBaseline } from '@/stores/qml'
import CiraLogo from '@/components/CiraLogo.vue'
import TrainingLossChart from '@/components/TrainingLossChart.vue'
import VqcCircuitExplainer from '@/components/VqcCircuitExplainer.vue'
import BaselineComparison from '@/components/BaselineComparison.vue'
import DecisionBoundaryPlot from '@/components/DecisionBoundaryPlot.vue'
import QpuRunPanel from '@/components/QpuRunPanel.vue'

const auth = useAuthStore()
const api = axios.create({ withCredentials: true })

const route = useRoute()
const router = useRouter()
const qml = useQmlStore()

const loading = ref(true)
const fatal = ref<string | null>(null)

const jobId = computed(() => route.params.id as string)

const hyperparametersParsed = computed(() => {
  if (!qml.currentJob?.hyperparameters) return null
  try {
    return JSON.parse(qml.currentJob.hyperparameters)
  } catch {
    return null
  }
})

const dataset = computed(() =>
  qml.datasets.find((d) => d.id === qml.currentJob?.dataset_id),
)

const statusColor = computed(() => {
  switch (qml.currentJob?.status) {
    case 'complete':           return 'success'
    case 'error':              return 'error'
    case 'training':           return 'primary'
    case 'baselines_training': return 'accent'
    case 'loading':            return 'info'
    case 'queued':             return 'grey'
    default:                   return 'grey'
  }
})

const totalEpochs = computed(
  () => qml.trainingMeta?.n_epochs ?? hyperparametersParsed.value?.n_epochs ?? null,
)

const circuitInfo = computed<QmlCircuitInfo | null>(
  () =>
    qml.finalMetrics?.circuit_info
    || qml.trainingMeta?.circuit_info
    || hyperparametersParsed.value?.circuit_info
    || null,
)

const featureNames = computed<string[]>(
  () =>
    qml.finalMetrics?.feature_names
    || qml.trainingMeta?.feature_names
    || hyperparametersParsed.value?.feature_names
    || [],
)

const classes = computed<string[]>(
  () =>
    qml.finalMetrics?.classes
    || qml.trainingMeta?.classes
    || hyperparametersParsed.value?.classes
    || [],
)

const trainedWeights = computed<number[][] | undefined>(
  () => qml.finalMetrics?.weights,
)

// Decision boundary — live during training, falls back to the
// persisted final-epoch grid on completion (or page refresh).
const boundaryGrid = computed(
  () => qml.liveDecisionGrid || qml.finalMetrics?.decision_grid || null,
)
const boundaryEpochLabel = computed(() => {
  if (qml.isTraining && qml.liveDecisionGridEpoch !== null) {
    return `epoch ${qml.liveDecisionGridEpoch}/${totalEpochs.value || '?'}`
  }
  if (!qml.isTraining && qml.finalMetrics?.decision_grid) {
    return 'final'
  }
  return null
})
const scatterPoints = computed(
  () => qml.finalMetrics?.scatter_points || null,
)

// QML-5: completed QPU runs surface in the baseline comparison as
// synthetic "baseline" rows so the student sees vqc_ibmq next to
// LogReg / SVM / RF / MLP on the same scoreboard.
//
// Only IBM's batch-mode runs land in the comparison — Origin's
// single-point runs aren't full test accuracies and would mislead the
// scoreboard. They show up only in the QpuRunPanel where their
// single-point shape is rendered honestly.
const baselinesWithQpu = computed<QmlBaseline[]>(() => {
  const base = qml.finalMetrics?.baselines ? [...qml.finalMetrics.baselines] : []
  for (const run of qml.qpuRuns) {
    if (run.status !== 'complete' || !run.metrics) continue
    let m: any = null
    try { m = JSON.parse(run.metrics) } catch { continue }
    if (!m || m.mode !== 'batch') continue
    base.push({
      name: `vqc_${run.provider}_${run.id.slice(0, 6)}`,
      title: `VQC on ${m.backend_name || run.provider} (${m.shots} shots)`,
      library: run.provider === 'originqc' ? 'pyqpanda3' : 'qiskit-ibm-runtime',
      version: '',
      // "neural" family chip so QPU rows are visually distinct from
      // the four sklearn baselines but sit in the same table. The
      // trophy logic in BaselineComparison treats all rows uniformly.
      family: 'neural',
      train_accuracy: qml.finalMetrics?.final_train_accuracy ?? 0,
      test_accuracy: m.test_accuracy,
      train_time_ms: run.wall_time_ms || 0,
      confusion_matrix: m.confusion_matrix,
      notes: 'Real superconducting QPU — same trained weights as the simulator '
        + 'row, re-evaluated with shot noise.',
    })
  }
  return base
})

// 2-qubit jobs are the only ones with a 2D test split persisted, which
// is what the QPU panel needs. Check it the cheap way: scatter_points
// existing AND every point having exactly 2 coordinates is implied by
// the trainer's invariant, so a non-null scatter is enough.
const has2dSplit = computed(
  () => Array.isArray(scatterPoints.value) && scatterPoints.value.length > 0,
)

// QML-7: admin-only "archive this run to public benchmarks" action.
const archiving = ref(false)
const archiveError = ref<string | null>(null)
const archivedRecordId = ref<string | null>(null)
const archiveNotes = ref('')
const archiveDialog = ref(false)

async function archiveToBenchmarks() {
  if (!qml.currentJob) return
  archiving.value = true
  archiveError.value = null
  try {
    const r = await api.post(
      `/api/qml/benchmarks/archive/${qml.currentJob.id}`,
      { notes: archiveNotes.value.trim() || undefined },
    )
    archivedRecordId.value = r.data.record_id
    archiveDialog.value = false
  } catch (e: any) {
    archiveError.value =
      e?.response?.data?.error || e?.message || 'Archive failed'
  } finally {
    archiving.value = false
  }
}

const progressPct = computed(() => {
  if (!totalEpochs.value || !qml.liveHistory.length) return 0
  return Math.min(100, (qml.liveHistory.length / totalEpochs.value) * 100)
})

onMounted(async () => {
  try {
    // Loading datasets is best-effort — we only need it to render the
    // dataset card. Failing here shouldn't break the page.
    await Promise.all([
      qml.loadDatasets().catch(() => {}),
      qml.loadJob(jobId.value),
    ])
    // If the job is still running, open the SSE stream. If it already
    // completed before we got here, the persisted training_history was
    // already hydrated into liveHistory in loadJob.
    if (qml.isTraining) {
      qml.subscribeStream(jobId.value)
    }
  } catch (e: any) {
    fatal.value = e?.response?.data?.error || e?.message || 'Failed to load job'
  } finally {
    loading.value = false
  }
})

onBeforeUnmount(() => {
  qml.closeStream()
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QML job detail">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QML job</span>
    </div>
    <v-spacer />
    <v-btn variant="text" prepend-icon="mdi-school" @click="router.push('/qml/learn')">
      Primer
    </v-btn>
    <v-btn variant="text" @click="router.push('/qml')">Datasets</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />
      <v-alert v-else-if="fatal" type="error" variant="tonal">{{ fatal }}</v-alert>

      <template v-else-if="qml.currentJob">
        <!-- Header -->
        <div class="d-flex align-center flex-wrap ga-3 mb-2">
          <v-btn
            icon="mdi-arrow-left"
            variant="text"
            @click="router.push('/qml')"
            aria-label="Back to dataset gallery"
          />
          <div class="flex-grow-1">
            <div class="text-overline text-medium-emphasis">
              VQC training — {{ dataset?.title || qml.currentJob.dataset_id }}
            </div>
            <div class="text-h5">Job {{ qml.currentJob.id.slice(0, 8) }}</div>
          </div>
          <v-chip :color="statusColor" variant="flat">
            {{ qml.currentJob.status }}
          </v-chip>
        </div>

        <!-- Hyperparameters strip -->
        <v-card variant="tonal" class="pa-3 mb-4">
          <div class="d-flex flex-wrap ga-4">
            <div v-if="hyperparametersParsed">
              <div class="text-caption text-medium-emphasis">qubits</div>
              <div class="text-subtitle-2">
                {{ qml.trainingMeta?.n_qubits ?? hyperparametersParsed.n_qubits }}
              </div>
            </div>
            <div v-if="hyperparametersParsed">
              <div class="text-caption text-medium-emphasis">layers</div>
              <div class="text-subtitle-2">{{ hyperparametersParsed.n_layers }}</div>
            </div>
            <div v-if="totalEpochs">
              <div class="text-caption text-medium-emphasis">epochs</div>
              <div class="text-subtitle-2">{{ totalEpochs }}</div>
            </div>
            <div v-if="hyperparametersParsed">
              <div class="text-caption text-medium-emphasis">batch</div>
              <div class="text-subtitle-2">{{ hyperparametersParsed.batch_size }}</div>
            </div>
            <div v-if="hyperparametersParsed">
              <div class="text-caption text-medium-emphasis">lr</div>
              <div class="text-subtitle-2">{{ hyperparametersParsed.learning_rate }}</div>
            </div>
            <div v-if="qml.trainingMeta?.n_samples_train">
              <div class="text-caption text-medium-emphasis">train samples</div>
              <div class="text-subtitle-2">{{ qml.trainingMeta.n_samples_train }}</div>
            </div>
            <div v-if="qml.trainingMeta?.n_samples_test">
              <div class="text-caption text-medium-emphasis">test samples</div>
              <div class="text-subtitle-2">{{ qml.trainingMeta.n_samples_test }}</div>
            </div>
          </div>
          <div
            v-if="qml.trainingMeta?.pca_applied"
            class="text-caption mt-2"
          >
            <v-icon icon="mdi-information-outline" size="x-small" />
            PCA projection applied: {{ qml.trainingMeta?.notes?.find((n: string) => n.includes('PCA')) }}
          </div>
        </v-card>

        <!-- Circuit + backend transparency. Visible from the moment the
             trainer emits the 'training' event so students see WHAT is
             being trained, not just the loss numbers. -->
        <VqcCircuitExplainer
          v-if="circuitInfo"
          :info="circuitInfo"
          :feature-names="featureNames"
          :classes="classes"
          :weights="trainedWeights"
          class="mb-4"
        />

        <!-- Live training -->
        <v-card v-if="qml.isTraining" class="pa-4 mb-4">
          <div class="d-flex align-center mb-2">
            <v-progress-circular indeterminate size="20" width="2" class="mr-2" />
            <span class="text-subtitle-1">
              {{
                qml.currentJob.status === 'baselines_training'
                  ? 'Training classical baselines…'
                  : 'Training in progress'
              }}
            </span>
            <v-spacer />
            <span class="text-caption text-medium-emphasis">
              {{ qml.liveHistory.length }} / {{ totalEpochs || '?' }} epochs
            </span>
          </div>
          <v-progress-linear
            :model-value="qml.currentJob.status === 'baselines_training' ? 100 : progressPct"
            :color="qml.currentJob.status === 'baselines_training' ? 'accent' : 'primary'"
            :indeterminate="qml.currentJob.status === 'baselines_training'"
            height="8"
            rounded
            class="mb-3"
          />
          <v-row>
            <v-col :cols="boundaryGrid ? 12 : 12" :md="boundaryGrid ? 7 : 12">
              <TrainingLossChart
                v-if="qml.liveHistory.length"
                :history="qml.liveHistory"
                :total-epochs="totalEpochs"
              />
              <div v-else class="text-body-2 text-medium-emphasis">
                Waiting for the first epoch…
              </div>
            </v-col>
            <v-col v-if="boundaryGrid" cols="12" md="5">
              <div class="text-caption text-medium-emphasis mb-1">
                <v-icon icon="mdi-vector-curve" size="x-small" />
                Decision boundary — what the model has carved up so far.
                Watch it bend as the rotations train.
              </div>
              <DecisionBoundaryPlot
                :grid="boundaryGrid"
                :points="scatterPoints || undefined"
                :classes="classes"
                :feature-names="featureNames"
                :epoch-label="boundaryEpochLabel"
                show-split
              />
            </v-col>
          </v-row>
          <v-alert
            v-if="qml.currentJob.status === 'baselines_training'"
            type="info"
            variant="tonal"
            density="compact"
            icon="mdi-podium"
            class="mt-3"
          >
            VQC finished. Now running four classical baselines
            (LogReg, SVM-RBF, RandomForest, MLP) on the same split so you
            can compare them honestly. This takes a couple of seconds.
          </v-alert>
        </v-card>

        <!-- Completed -->
        <template v-else-if="qml.currentJob.status === 'complete' && qml.finalMetrics">
          <v-row>
            <v-col cols="12" md="7">
              <v-card class="pa-4 h-100">
                <div class="text-subtitle-1 mb-2">Training history</div>
                <TrainingLossChart
                  :history="qml.liveHistory"
                  :total-epochs="totalEpochs"
                />
              </v-card>
            </v-col>
            <v-col cols="12" md="5">
              <v-card class="pa-4 h-100">
                <div class="text-subtitle-1 mb-3">Final metrics</div>
                <v-row dense>
                  <v-col cols="6">
                    <div class="text-caption text-medium-emphasis">test accuracy</div>
                    <div class="text-h5 text-success">
                      {{ (qml.finalMetrics.final_test_accuracy * 100).toFixed(1) }}%
                    </div>
                  </v-col>
                  <v-col cols="6">
                    <div class="text-caption text-medium-emphasis">train accuracy</div>
                    <div class="text-h5">
                      {{ (qml.finalMetrics.final_train_accuracy * 100).toFixed(1) }}%
                    </div>
                  </v-col>
                  <v-col cols="6">
                    <div class="text-caption text-medium-emphasis">final loss (BCE)</div>
                    <div class="text-subtitle-1">
                      {{ qml.finalMetrics.final_loss.toFixed(4) }}
                    </div>
                  </v-col>
                  <v-col cols="6">
                    <div class="text-caption text-medium-emphasis">wall time</div>
                    <div class="text-subtitle-1">
                      {{ ((qml.finalMetrics.train_time_ms || 0) / 1000).toFixed(1) }} s
                    </div>
                  </v-col>
                </v-row>

                <v-divider class="my-3" />
                <div class="text-subtitle-2 mb-2">Confusion matrix</div>
                <table class="confusion-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th>pred {{ qml.finalMetrics.classes[0] }}</th>
                      <th>pred {{ qml.finalMetrics.classes[1] }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>true {{ qml.finalMetrics.classes[0] }}</th>
                      <td class="diag">{{ qml.finalMetrics.confusion_matrix[0][0] }}</td>
                      <td>{{ qml.finalMetrics.confusion_matrix[0][1] }}</td>
                    </tr>
                    <tr>
                      <th>true {{ qml.finalMetrics.classes[1] }}</th>
                      <td>{{ qml.finalMetrics.confusion_matrix[1][0] }}</td>
                      <td class="diag">{{ qml.finalMetrics.confusion_matrix[1][1] }}</td>
                    </tr>
                  </tbody>
                </table>
              </v-card>
            </v-col>
          </v-row>

          <!-- QML-4 final decision boundary. Only meaningful for 2-qubit
               jobs where the input space is 2D — for higher-dim runs we
               skip the section silently (the boundary would have to be
               projected and that's deferred to a later phase). -->
          <v-card
            v-if="boundaryGrid && scatterPoints"
            variant="outlined"
            class="pa-4 mt-4"
          >
            <div class="d-flex align-center mb-2">
              <v-icon icon="mdi-vector-curve" class="mr-2" />
              <div class="text-subtitle-1 flex-grow-1">
                What the VQC learned — final decision boundary
              </div>
              <v-chip size="x-small" variant="tonal">final</v-chip>
            </div>
            <div class="text-body-2 text-medium-emphasis mb-3">
              The heatmap shows the model's predicted probability across
              the input plane. Cells where it predicts class
              <code>{{ classes[0] }}</code> are blue, class
              <code>{{ classes[1] }}</code> are orange, and the white
              ribbon is the <code>0.5</code> decision boundary. Training
              points are overlaid — a point in a region of its own
              color means the VQC classifies it correctly.
            </div>
            <DecisionBoundaryPlot
              :grid="boundaryGrid"
              :points="scatterPoints"
              :classes="classes"
              :feature-names="featureNames"
              :epoch-label="'final'"
              show-split
            />
          </v-card>

          <!-- QML-5: real-QPU inference panel. Lives between the final
               decision boundary and the comparison table so a student
               first sees what the model learned, then can ask "what
               would this look like on real hardware?" before reading
               the scoreboard. -->
          <QpuRunPanel
            :parent-job-id="qml.currentJob.id"
            :has2d-split="has2dSplit"
            :simulator-test-accuracy="qml.finalMetrics.final_test_accuracy"
            class="mt-4"
          />

          <!-- QML-3: side-by-side comparison with four classical baselines.
               Same train/test split, same standard scaling — like-for-like.
               QML-5 extends this with completed QPU runs as extra rows. -->
          <BaselineComparison
            v-if="baselinesWithQpu.length"
            :vqc="qml.finalMetrics"
            :baselines="baselinesWithQpu"
            class="mt-4"
          />

          <!-- QML-7: admin-only "archive to public benchmarks" panel.
               Hidden for regular users to avoid noise; an admin curates
               which runs are worth citing. -->
          <v-card
            v-if="auth.user && auth.user.role === 'admin'"
            variant="outlined"
            class="pa-4 mt-4"
            border="success"
          >
            <div class="d-flex align-center mb-2">
              <v-icon icon="mdi-archive-arrow-up" color="success" class="mr-2" />
              <div class="text-subtitle-1 flex-grow-1">
                Archive to public benchmarks
              </div>
              <v-chip size="small" color="success" variant="tonal">
                admin
              </v-chip>
            </div>
            <div class="text-body-2 text-medium-emphasis mb-3">
              Snapshot this run into the public
              <code>qml_benchmarks/archive</code> directory. The record
              keeps the contributor's display name + every metric on
              this page, with the trained weights stripped (those are
              model state, not a citation).
            </div>

            <v-alert
              v-if="archivedRecordId"
              type="success"
              variant="tonal"
              density="compact"
              class="mb-2"
              icon="mdi-check-decagram"
            >
              Archived as
              <code>{{ archivedRecordId.slice(0, 26) }}…</code>.
              <a
                href="#"
                class="ml-1"
                @click.prevent="router.push(`/qml/benchmarks/${archivedRecordId}`)"
              >View record →</a>
            </v-alert>

            <v-btn
              v-if="!archivedRecordId"
              color="success"
              variant="flat"
              prepend-icon="mdi-archive-arrow-up"
              @click="archiveDialog = true"
            >Archive this run</v-btn>
          </v-card>

          <!-- Archive confirmation dialog -->
          <v-dialog v-model="archiveDialog" max-width="500">
            <v-card>
              <v-card-title>
                <v-icon icon="mdi-archive-arrow-up" class="mr-2" />
                Archive run to public benchmarks
              </v-card-title>
              <v-card-text>
                <div class="text-body-2 mb-3">
                  The record will be visible on the public
                  <code>/qml/benchmarks</code> dashboard and citable by
                  BibTeX. The contributor credit goes to the original
                  runner of this job, not you.
                </div>
                <v-textarea
                  v-model="archiveNotes"
                  label="Notes (optional)"
                  placeholder="e.g. reference run for the QML-2 primer; baseline for course week 3"
                  rows="3"
                  density="comfortable"
                  hide-details="auto"
                  counter="300"
                  maxlength="300"
                />
                <v-alert
                  v-if="archiveError"
                  type="error"
                  variant="tonal"
                  density="compact"
                  class="mt-3"
                >{{ archiveError }}</v-alert>
              </v-card-text>
              <v-card-actions>
                <v-spacer />
                <v-btn
                  variant="text"
                  :disabled="archiving"
                  @click="archiveDialog = false"
                >Cancel</v-btn>
                <v-btn
                  color="success"
                  variant="flat"
                  :loading="archiving"
                  prepend-icon="mdi-archive-arrow-up"
                  @click="archiveToBenchmarks"
                >Archive</v-btn>
              </v-card-actions>
            </v-card>
          </v-dialog>
        </template>

        <!-- Error -->
        <v-alert
          v-else-if="qml.currentJob.status === 'error'"
          type="error"
          variant="tonal"
          icon="mdi-alert-circle"
        >
          <div class="text-subtitle-1 font-weight-medium">Training failed</div>
          <div class="text-body-2 mt-1">
            {{ qml.currentJob.error || 'No error message recorded.' }}
          </div>
        </v-alert>

        <!-- Queued (transient — should flip to 'loading' almost immediately) -->
        <v-card v-else class="pa-4 text-center">
          <v-progress-circular indeterminate class="mb-2" />
          <div class="text-body-2 text-medium-emphasis">
            Queued — the trainer is spinning up.
          </div>
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
.confusion-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.confusion-table th,
.confusion-table td {
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 6px 10px;
  text-align: center;
}
.confusion-table th {
  font-weight: 500;
  background: rgba(255, 255, 255, 0.04);
}
.confusion-table td.diag {
  background: rgba(34, 197, 94, 0.12);
  font-weight: 600;
}
</style>
