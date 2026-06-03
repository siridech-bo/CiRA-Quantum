<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useQmlStore, type QmlDataset, type QmlHyperparameters } from '@/stores/qml'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
const auth = useAuthStore()
const qml = useQmlStore()

const loading = ref(true)
const error = ref<string | null>(null)

const trainingReady = computed(
  () => qml.capabilities?.pennylane && qml.capabilities?.sklearn,
)

const difficultyColor = (d: QmlDataset['difficulty']) =>
  d === 'easy' ? 'success' : d === 'medium' ? 'warning' : 'error'

onMounted(async () => {
  try {
    await Promise.all([qml.loadDatasets(), qml.loadCapabilities()])
  } catch (e: any) {
    error.value = e?.message || 'Failed to load QML datasets'
  } finally {
    loading.value = false
  }
})

// Train dialog state
const trainDialog = ref(false)
const trainDataset = ref<QmlDataset | null>(null)
const trainSubmitting = ref(false)
const trainError = ref<string | null>(null)

const hyperparameters = ref<QmlHyperparameters>({
  n_qubits: 2,
  n_layers: 2,
  n_epochs: 30,
  batch_size: 16,
  learning_rate: 0.05,
  seed: 42,
})

function openTrainDialog(d: QmlDataset) {
  trainDataset.value = d
  trainError.value = null
  // Pre-seed n_qubits from the dataset (capped at 10 — the trainer's
  // statevector ceiling). Datasets with > 10 features get PCA-projected
  // by the loader; this widget surfaces that so the student sees why.
  hyperparameters.value = {
    n_qubits: Math.min(d.n_features, 10),
    n_layers: d.difficulty === 'hard' ? 3 : 2,
    n_epochs: d.difficulty === 'easy' ? 30 : 50,
    batch_size: 16,
    learning_rate: 0.05,
    seed: 42,
  }
  trainDialog.value = true
}

async function submitTraining() {
  if (!trainDataset.value) return
  trainSubmitting.value = true
  trainError.value = null
  try {
    if (!auth.user) {
      router.push({
        path: '/login',
        query: { redirect: `/qml#train=${trainDataset.value.id}` },
      })
      return
    }
    const job = await qml.startTraining(trainDataset.value.id, hyperparameters.value)
    trainDialog.value = false
    router.push(`/qml/jobs/${job.id}`)
  } catch (e: any) {
    trainError.value =
      e?.response?.data?.error || e?.message || 'Failed to start training'
  } finally {
    trainSubmitting.value = false
  }
}

// PCA cap warning shown in the dialog when n_features > n_qubits.
const pcaWarning = computed(() => {
  if (!trainDataset.value) return null
  const n = trainDataset.value.n_features
  const q = hyperparameters.value.n_qubits || n
  if (q < n) {
    return `Dataset has ${n} features; PCA will project to ${q} components before training.`
  }
  return null
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QML app bar">
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
    <v-btn variant="text" prepend-icon="mdi-trophy-outline" @click="router.push('/qml/benchmarks')">
      Benchmarks
    </v-btn>
    <v-btn variant="text" @click="router.push('/')">Home</v-btn>
    <v-btn variant="text" @click="router.push('/solve')" v-if="auth.user">
      Optimization app
    </v-btn>
    <v-btn v-if="!auth.user" variant="outlined" class="ml-2" @click="router.push('/login')">
      Log in
    </v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <!-- Hero -->
      <div class="pt-8 pb-4">
        <div class="text-overline text-medium-emphasis">QML-2 — local VQC pipeline</div>
        <div class="text-h4 font-weight-bold mb-2">
          Quantum Machine Learning, on the same platform
        </div>
        <div class="text-body-1 text-medium-emphasis" style="max-width: 720px">
          Train a Variational Quantum Classifier on one of 6 curated
          datasets. PennyLane statevector simulator, live training-loss
          curve, confusion matrix + decision boundary on the results page.
        </div>
      </div>

      <!-- Capability strip -->
      <v-alert
        v-if="qml.capabilities && !trainingReady"
        type="warning"
        variant="tonal"
        class="my-4"
        icon="mdi-package-variant-closed"
      >
        <div class="text-subtitle-1 font-weight-medium">
          Training stack not installed
        </div>
        <div class="text-body-2">
          Install the optional extra on the backend host so the Train
          button can submit jobs:
        </div>
        <code class="d-block mt-2 pa-2 bg-grey-darken-4 rounded">
          pip install ".[qml]"
        </code>
      </v-alert>

      <v-alert
        v-else-if="qml.capabilities && trainingReady"
        type="info"
        variant="tonal"
        class="my-4"
        icon="mdi-flask-outline"
        density="compact"
      >
        Training stack ready —
        <v-chip size="x-small" color="success" class="mx-1">PennyLane</v-chip>
        <v-chip size="x-small" color="success" class="mx-1">scikit-learn</v-chip>
        <v-chip
          size="x-small"
          :color="qml.capabilities.qiskit_ibm_runtime ? 'success' : 'grey'"
          class="mx-1"
        >
          IBM cloud
          <span v-if="!qml.capabilities.qiskit_ibm_runtime"> (not installed)</span>
        </v-chip>
      </v-alert>

      <!-- Primer funnel — first-time visitors should hit the walkthrough
           before they spend simulator time figuring out what RX(x) means. -->
      <v-card
        variant="tonal"
        color="accent"
        class="pa-4 my-4 d-flex align-center flex-wrap ga-3 primer-card"
        hover
        @click="router.push('/qml/learn')"
      >
        <v-icon icon="mdi-school" size="32" />
        <div class="flex-grow-1">
          <div class="text-subtitle-1 font-weight-medium">
            New to quantum classifiers? Start here.
          </div>
          <div class="text-body-2 text-medium-emphasis">
            A five-step primer (qubit → encoding → entangler → measurement
            → learning) with an interactive Bloch sphere. ~5 min read.
          </div>
        </div>
        <v-btn
          variant="flat"
          color="accent"
          prepend-icon="mdi-arrow-right"
          @click.stop="router.push('/qml/learn')"
        >
          Open primer
        </v-btn>
      </v-card>

      <!-- Dataset gallery -->
      <div class="d-flex align-center mt-6 mb-2">
        <div class="text-h6">Dataset gallery</div>
        <v-spacer />
        <v-chip size="small" variant="tonal">{{ qml.datasets.length }} datasets</v-chip>
      </div>
      <div class="text-body-2 text-medium-emphasis mb-4">
        Every dataset has a 2-class form so it can drive a small VQC.
        Click a card (or <strong>Study</strong>) to see the data + the
        circuit before training, or <strong>Train</strong> to skip
        straight to a run.
      </div>

      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />
      <v-alert v-else-if="error" type="error" variant="tonal">{{ error }}</v-alert>

      <v-row v-else dense>
        <v-col
          v-for="d in qml.datasets"
          :key="d.id"
          cols="12"
          sm="6"
          md="4"
        >
          <v-card
            class="pa-4 h-100 dataset-card"
            variant="outlined"
            hover
            @click="router.push(`/qml/datasets/${d.id}`)"
          >
            <div class="d-flex align-center mb-2">
              <span class="text-subtitle-1 font-weight-medium flex-grow-1">
                {{ d.title }}
              </span>
              <v-chip
                size="x-small"
                :color="difficultyColor(d.difficulty)"
                variant="flat"
              >
                {{ d.difficulty }}
              </v-chip>
            </div>
            <div class="d-flex ga-1 mb-2 flex-wrap">
              <v-chip size="x-small" variant="outlined">{{ d.category }}</v-chip>
              <v-chip size="x-small" variant="outlined">
                {{ d.n_features }} features
              </v-chip>
              <v-chip size="x-small" variant="outlined">
                {{ d.n_samples }} samples
              </v-chip>
            </div>
            <div class="text-body-2 text-medium-emphasis">{{ d.summary }}</div>
            <v-divider class="my-3" />
            <div class="d-flex align-center ga-2">
              <code class="text-caption text-medium-emphasis flex-grow-1">
                {{ d.source }}
              </code>
              <!-- Detail: the educational entry point. Stops propagation
                   so it doesn't double-fire with the card click. -->
              <v-btn
                size="small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-magnify"
                @click.stop="router.push(`/qml/datasets/${d.id}`)"
              >
                Study
              </v-btn>
              <!-- Train: power-user shortcut that skips the study page. -->
              <v-btn
                size="small"
                variant="flat"
                color="primary"
                :disabled="!trainingReady"
                prepend-icon="mdi-play"
                @click.stop="openTrainDialog(d)"
              >
                Train
              </v-btn>
            </div>
          </v-card>
        </v-col>
      </v-row>

    </v-container>

    <!-- Train dialog -->
    <v-dialog v-model="trainDialog" max-width="540">
      <v-card v-if="trainDataset">
        <v-card-title>
          <v-icon icon="mdi-flask" class="mr-2" />
          Train VQC on {{ trainDataset.title }}
        </v-card-title>
        <v-card-text>
          <div class="text-body-2 text-medium-emphasis mb-3">
            {{ trainDataset.summary }}
          </div>

          <v-text-field
            v-model.number="hyperparameters.n_qubits"
            label="Qubits"
            type="number"
            :min="1" :max="10"
            hint="≤ dataset features; > 10 not supported (statevector blows up)"
            persistent-hint
            class="mb-2"
          />
          <v-text-field
            v-model.number="hyperparameters.n_layers"
            label="Entangler layers"
            type="number"
            :min="1" :max="6"
            hint="Trainable params = n_layers × n_qubits"
            persistent-hint
            class="mb-2"
          />
          <v-text-field
            v-model.number="hyperparameters.n_epochs"
            label="Epochs"
            type="number"
            :min="1" :max="200"
            class="mb-2"
          />
          <v-text-field
            v-model.number="hyperparameters.batch_size"
            label="Batch size"
            type="number"
            :min="1" :max="64"
            class="mb-2"
          />
          <v-text-field
            v-model.number="hyperparameters.learning_rate"
            label="Learning rate"
            type="number"
            step="0.01"
            :min="0.001" :max="1"
            class="mb-2"
          />
          <v-text-field
            v-model.number="hyperparameters.seed"
            label="Random seed"
            type="number"
            class="mb-2"
          />

          <v-alert
            v-if="pcaWarning"
            type="info"
            variant="tonal"
            density="compact"
            class="mt-2"
          >
            {{ pcaWarning }}
          </v-alert>
          <v-alert
            v-if="trainError"
            type="error"
            variant="tonal"
            density="compact"
            class="mt-2"
          >
            {{ trainError }}
          </v-alert>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" :disabled="trainSubmitting" @click="trainDialog = false">
            Cancel
          </v-btn>
          <v-btn
            color="primary"
            variant="flat"
            :loading="trainSubmitting"
            prepend-icon="mdi-play"
            @click="submitTraining"
          >
            Launch training
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
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
.logo-link:focus-visible {
  outline: 2px solid currentColor;
  outline-offset: 4px;
  border-radius: 4px;
}
.dataset-card {
  cursor: pointer;
}
.primer-card {
  cursor: pointer;
}
</style>
