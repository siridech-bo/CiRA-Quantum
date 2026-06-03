<script setup lang="ts">
/**
 * QmlDatasetDetailPage — the "study before training" surface.
 *
 * The educational flow is:
 *   1. Read what the dataset is (summary, source, sample count).
 *   2. See it as a scatter plot — 2D natively, PCA-2 if high-dim.
 *   3. Inspect the VQC circuit we'll train (gates + parameter count).
 *   4. See which classical baselines we'll run side-by-side.
 *   5. Set hyperparameters with caps + sensible defaults.
 *   6. Hit Launch — and the page hands off to /qml/jobs/:id.
 */
import { computed, onMounted, ref } from 'vue'
import axios from 'axios'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import {
  useQmlStore,
  type QmlCircuitInfo,
  type QmlDataset,
  type QmlHyperparameters,
} from '@/stores/qml'
import CiraLogo from '@/components/CiraLogo.vue'
import DatasetScatterPlot from '@/components/DatasetScatterPlot.vue'
import VqcCircuitExplainer from '@/components/VqcCircuitExplainer.vue'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const qml = useQmlStore()
const api = axios.create({ withCredentials: true })

const datasetId = computed(() => route.params.id as string)

const dataset = ref<QmlDataset | null>(null)
const preview = ref<{
  classes: string[]
  feature_names: string[]
  pca_applied: boolean
  n_points: number
  points: { x: number; y: number; label: number }[]
  notes: string[]
} | null>(null)

const loading = ref(true)
const previewLoading = ref(true)
const error = ref<string | null>(null)
const previewError = ref<string | null>(null)

const submitting = ref(false)
const submitError = ref<string | null>(null)

const hyperparameters = ref<QmlHyperparameters>({
  n_qubits: 2,
  n_layers: 2,
  n_epochs: 30,
  batch_size: 16,
  learning_rate: 0.05,
  seed: 42,
})

// Derived circuit preview from the current hyperparameter form. This
// drives the gate diagram before any job exists — students see what
// the circuit looks like before they spend simulator time on it.
const circuitPreview = computed<QmlCircuitInfo>(() => ({
  backend_name: 'PennyLane default.qubit',
  backend_kind: 'statevector',
  is_real_hardware: false,
  n_qubits: Math.min(
    Math.max(1, hyperparameters.value.n_qubits || 2),
    10,
    dataset.value?.n_features || 10,
  ),
  n_layers: Math.max(1, hyperparameters.value.n_layers || 2),
  n_trainable_params:
    Math.max(1, hyperparameters.value.n_layers || 2)
    * Math.min(
      Math.max(1, hyperparameters.value.n_qubits || 2),
      10,
      dataset.value?.n_features || 10,
    )
    + 1,
  encoding: 'AngleEmbedding (RX rotations, one per input feature)',
  entangler: 'BasicEntanglerLayers (RY rotations + ring CNOT)',
  measurement: 'PauliZ expectation on qubit 0',
  shots: null,
}))

const pcaWarning = computed(() => {
  if (!dataset.value) return null
  const n = dataset.value.n_features
  const q = hyperparameters.value.n_qubits || n
  if (q < n) {
    return `Dataset has ${n} features; PCA will project to ${q} components `
      + 'before training. You can already see the same projection in the '
      + 'scatter plot above.'
  }
  return null
})

const featureNamesForCircuit = computed<string[]>(() => {
  // If the user picked fewer qubits than features, the trainer will PCA;
  // the circuit gets PC-named axes. Otherwise reuse the dataset's own
  // feature names if the preview returned native 2D (small datasets) —
  // we don't have the full feature list otherwise, so fall back to x_i.
  if (!dataset.value) return []
  const q = circuitPreview.value.n_qubits
  if (preview.value && !preview.value.pca_applied && preview.value.feature_names.length === q) {
    return preview.value.feature_names
  }
  if (q < dataset.value.n_features) {
    return Array.from({ length: q }, (_, i) => `PC${i + 1}`)
  }
  // Same-size match (e.g. Moons 2 features = 2 qubits).
  return preview.value?.feature_names || Array.from({ length: q }, (_, i) => `x${i}`)
})

const classesForCircuit = computed<string[]>(
  () => preview.value?.classes || [],
)

onMounted(async () => {
  // Pull the dataset record + the scatter preview in parallel — the
  // gallery may not have populated the store yet if a user deep-linked
  // straight into this page.
  try {
    const datasetReq = api.get<QmlDataset>(`/api/qml/datasets/${datasetId.value}`)
    const previewReq = api.get(`/api/qml/datasets/${datasetId.value}/preview`)
    const datasetRes = await datasetReq
    dataset.value = datasetRes.data
    // Seed hyperparameters from the dataset (the same rule the train
    // dialog used). n_qubits = min(n_features, 10).
    hyperparameters.value = {
      n_qubits: Math.min(datasetRes.data.n_features, 10),
      n_layers: datasetRes.data.difficulty === 'hard' ? 3 : 2,
      n_epochs: datasetRes.data.difficulty === 'easy' ? 30 : 50,
      batch_size: 16,
      learning_rate: 0.05,
      seed: 42,
    }
    try {
      const previewRes = await previewReq
      preview.value = previewRes.data
    } catch (e: any) {
      // Preview is non-fatal — page still works without the scatter plot.
      previewError.value =
        e?.response?.data?.error || e?.message || 'Failed to load preview'
    } finally {
      previewLoading.value = false
    }
    // Refresh capabilities for the train button gating.
    if (!qml.capabilities) {
      await qml.loadCapabilities()
    }
  } catch (e: any) {
    error.value =
      e?.response?.status === 404
        ? 'Dataset not found.'
        : e?.response?.data?.error || e?.message || 'Failed to load dataset'
  } finally {
    loading.value = false
  }
})

const trainingReady = computed(
  () => qml.capabilities?.pennylane && qml.capabilities?.sklearn,
)

async function launchTraining() {
  if (!dataset.value) return
  submitting.value = true
  submitError.value = null
  try {
    if (!auth.user) {
      router.push({
        path: '/login',
        query: { redirect: `/qml/datasets/${dataset.value.id}` },
      })
      return
    }
    const job = await qml.startTraining(dataset.value.id, hyperparameters.value)
    router.push(`/qml/jobs/${job.id}`)
  } catch (e: any) {
    submitError.value =
      e?.response?.data?.error || e?.message || 'Failed to start training'
  } finally {
    submitting.value = false
  }
}

// Tiny formatter for the per-difficulty / per-category chip colors.
function difficultyColor(d: 'easy' | 'medium' | 'hard') {
  return d === 'easy' ? 'success' : d === 'medium' ? 'warning' : 'error'
}
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QML dataset detail">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QML</span>
    </div>
    <v-spacer />
    <v-btn variant="text" prepend-icon="mdi-school" @click="router.push('/qml/learn')">
      Primer
    </v-btn>
    <v-btn variant="text" @click="router.push('/qml')">All datasets</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />
      <v-alert v-else-if="error" type="error" variant="tonal">{{ error }}</v-alert>

      <template v-else-if="dataset">
        <!-- Hero -->
        <div class="d-flex align-center flex-wrap ga-3 mb-2">
          <v-btn
            icon="mdi-arrow-left"
            variant="text"
            @click="router.push('/qml')"
            aria-label="Back to gallery"
          />
          <div class="flex-grow-1">
            <div class="text-overline text-medium-emphasis">
              QML dataset
            </div>
            <div class="text-h4 font-weight-bold">{{ dataset.title }}</div>
          </div>
          <v-chip
            :color="difficultyColor(dataset.difficulty)"
            variant="flat"
          >
            {{ dataset.difficulty }}
          </v-chip>
        </div>

        <div class="text-body-1 text-medium-emphasis mb-4" style="max-width: 800px">
          {{ dataset.summary }}
        </div>

        <div class="d-flex ga-2 flex-wrap mb-6">
          <v-chip size="small" variant="outlined">{{ dataset.category }}</v-chip>
          <v-chip size="small" variant="outlined">
            {{ dataset.n_features }} features
          </v-chip>
          <v-chip size="small" variant="outlined">
            {{ dataset.n_samples }} samples
          </v-chip>
          <v-chip size="small" variant="outlined">
            {{ dataset.n_classes }} classes
          </v-chip>
          <v-chip size="small" variant="outlined">
            <v-icon icon="mdi-source-branch" start size="small" />
            {{ dataset.source }}
          </v-chip>
        </div>

        <v-row>
          <!-- Scatter plot -->
          <v-col cols="12" md="6">
            <v-card variant="outlined" class="pa-4 h-100">
              <div class="d-flex align-center mb-2">
                <v-icon icon="mdi-chart-scatter-plot" class="mr-2" />
                <div class="text-subtitle-1 flex-grow-1">
                  Scatter preview
                </div>
                <v-chip
                  v-if="preview?.pca_applied"
                  size="x-small"
                  color="info"
                  variant="tonal"
                >
                  PCA-2D
                </v-chip>
              </div>
              <v-progress-circular
                v-if="previewLoading"
                indeterminate
                class="d-block mx-auto my-6"
              />
              <v-alert
                v-else-if="previewError"
                type="warning"
                variant="tonal"
                density="compact"
              >
                {{ previewError }}
              </v-alert>
              <DatasetScatterPlot
                v-else-if="preview"
                :points="preview.points"
                :classes="preview.classes"
                :feature-names="preview.feature_names"
                :pca-applied="preview.pca_applied"
              />
              <div
                v-if="preview?.pca_applied"
                class="text-caption text-medium-emphasis mt-2"
              >
                The dataset has {{ dataset.n_features }} features.
                We're projecting them onto the first two principal
                components so you can see the class structure at a
                glance. The trainer can use up to
                {{ Math.min(dataset.n_features, 10) }} components.
              </div>
            </v-card>
          </v-col>

          <!-- What we'll run -->
          <v-col cols="12" md="6">
            <v-card variant="outlined" class="pa-4 h-100">
              <div class="d-flex align-center mb-3">
                <v-icon icon="mdi-format-list-bulleted-square" class="mr-2" />
                <div class="text-subtitle-1">What gets trained</div>
              </div>
              <div class="text-body-2 mb-3">
                One training run produces <strong>five</strong> trained
                models on the same standard-scaled split, so the
                comparison is honest:
              </div>
              <v-list density="compact" class="pa-0">
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-atom-variant" color="accent" />
                  </template>
                  <v-list-item-title class="text-body-2">
                    <strong>Variational Quantum Classifier</strong>
                  </v-list-item-title>
                  <v-list-item-subtitle class="text-caption">
                    The model under study. Trained on a statevector simulator.
                  </v-list-item-subtitle>
                </v-list-item>
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-vector-line" color="grey" />
                  </template>
                  <v-list-item-title class="text-body-2">
                    Logistic Regression
                  </v-list-item-title>
                  <v-list-item-subtitle class="text-caption">
                    Linear baseline — the floor everyone has to beat.
                  </v-list-item-subtitle>
                </v-list-item>
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-blur-radial" color="info" />
                  </template>
                  <v-list-item-title class="text-body-2">
                    SVM (RBF kernel)
                  </v-list-item-title>
                  <v-list-item-subtitle class="text-caption">
                    Classical kernel method — what a quantum kernel has to beat.
                  </v-list-item-subtitle>
                </v-list-item>
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-forest" color="success" />
                  </template>
                  <v-list-item-title class="text-body-2">
                    Random Forest
                  </v-list-item-title>
                  <v-list-item-subtitle class="text-caption">
                    Tree ensemble — strong baseline on tabular data.
                  </v-list-item-subtitle>
                </v-list-item>
                <v-list-item class="px-0">
                  <template #prepend>
                    <v-icon icon="mdi-graph-outline" color="primary" />
                  </template>
                  <v-list-item-title class="text-body-2">
                    MLP (16-8 ReLU)
                  </v-list-item-title>
                  <v-list-item-subtitle class="text-caption">
                    Small neural net — fairest head-to-head against a VQC.
                  </v-list-item-subtitle>
                </v-list-item>
              </v-list>
              <v-divider class="my-3" />
              <div class="text-caption text-medium-emphasis">
                The detail page (next step) shows the loss curve live and
                a side-by-side comparison table once everything finishes.
              </div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Circuit preview — updates as the user touches the hyperparameter form -->
        <div class="text-h6 mt-6 mb-2">
          <v-icon icon="mdi-vector-line" class="mr-1" />
          What the VQC will look like
        </div>
        <div class="text-body-2 text-medium-emphasis mb-3">
          The circuit re-renders as you change the hyperparameters below.
          The trainable weights are shown as
          <code>θ<sub>l,q</sub></code> placeholders here; they get filled
          in with the real trained values once training completes.
        </div>
        <VqcCircuitExplainer
          :info="circuitPreview"
          :feature-names="featureNamesForCircuit"
          :classes="classesForCircuit"
          class="mb-6"
        />

        <!-- Hyperparameters + Launch CTA -->
        <v-card variant="outlined" class="pa-4 mb-4">
          <div class="d-flex align-center mb-3">
            <v-icon icon="mdi-tune-variant" class="mr-2" />
            <div class="text-subtitle-1 flex-grow-1">Hyperparameters</div>
          </div>

          <v-row dense>
            <v-col cols="6" sm="4" md="2">
              <v-text-field
                v-model.number="hyperparameters.n_qubits"
                label="Qubits"
                type="number"
                :min="1" :max="10"
                density="comfortable"
                hide-details="auto"
                hint="≤ dataset features, ≤ 10"
                persistent-hint
              />
            </v-col>
            <v-col cols="6" sm="4" md="2">
              <v-text-field
                v-model.number="hyperparameters.n_layers"
                label="Layers"
                type="number"
                :min="1" :max="6"
                density="comfortable"
                hide-details="auto"
                hint="trainable params = layers · qubits"
                persistent-hint
              />
            </v-col>
            <v-col cols="6" sm="4" md="2">
              <v-text-field
                v-model.number="hyperparameters.n_epochs"
                label="Epochs"
                type="number"
                :min="1" :max="200"
                density="comfortable"
                hide-details="auto"
              />
            </v-col>
            <v-col cols="6" sm="4" md="2">
              <v-text-field
                v-model.number="hyperparameters.batch_size"
                label="Batch size"
                type="number"
                :min="1" :max="64"
                density="comfortable"
                hide-details="auto"
              />
            </v-col>
            <v-col cols="6" sm="4" md="2">
              <v-text-field
                v-model.number="hyperparameters.learning_rate"
                label="Learning rate"
                type="number"
                step="0.01"
                :min="0.001" :max="1"
                density="comfortable"
                hide-details="auto"
              />
            </v-col>
            <v-col cols="6" sm="4" md="2">
              <v-text-field
                v-model.number="hyperparameters.seed"
                label="Seed"
                type="number"
                density="comfortable"
                hide-details="auto"
              />
            </v-col>
          </v-row>

          <v-alert
            v-if="pcaWarning"
            type="info"
            variant="tonal"
            density="compact"
            class="mt-3"
          >
            {{ pcaWarning }}
          </v-alert>

          <v-alert
            v-if="!trainingReady"
            type="warning"
            variant="tonal"
            density="compact"
            class="mt-3"
          >
            Training stack not installed on the backend. Run
            <code>pip install ".[qml]"</code> and reload this page.
          </v-alert>

          <v-alert
            v-if="submitError"
            type="error"
            variant="tonal"
            density="compact"
            class="mt-3"
          >
            {{ submitError }}
          </v-alert>

          <div class="d-flex align-center mt-4">
            <v-btn variant="text" @click="router.push('/qml')">
              Back to gallery
            </v-btn>
            <v-spacer />
            <v-btn
              color="primary"
              variant="flat"
              size="large"
              :loading="submitting"
              :disabled="!trainingReady"
              prepend-icon="mdi-play"
              @click="launchTraining"
            >
              Launch training
            </v-btn>
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
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
</style>
