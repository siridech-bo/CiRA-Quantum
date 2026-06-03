<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useQldpcStore, type QldpcCodeFamily } from '@/stores/qldpc'
import CiraLogo from '@/components/CiraLogo.vue'
import MatrixViewer from '@/components/MatrixViewer.vue'
import TannerGraphViewer from '@/components/TannerGraphViewer.vue'

const route = useRoute()
const router = useRouter()
const qldpc = useQldpcStore()

const loading = ref(true)
const error = ref<string | null>(null)

// Sprint 1 — per-action loading + error state for the matrix / distance
// buttons. Independent so two operations don't race each other's spinners.
const matrixLoading = ref(false)
const matrixError = ref<string | null>(null)
const distanceLoading = ref(false)
const distanceError = ref<string | null>(null)
const exactRequested = ref(false)

const familyId = computed(() => String(route.params.id ?? ''))
const family = computed<QldpcCodeFamily | null>(() => qldpc.currentCodeFamily)
const mathStackReady = computed(() => qldpc.capabilities?.qldpc_lib === true)

const categoryLabel = (c: QldpcCodeFamily['category']) => {
  switch (c) {
    case 'topological': return 'Topological'
    case 'css_classical': return 'CSS (classical)'
    case 'css_product': return 'CSS (product)'
    default: return c
  }
}

async function load(id: string) {
  loading.value = true
  error.value = null
  qldpc.clearDetailState()
  try {
    await Promise.all([qldpc.loadCodeFamily(id), qldpc.loadCapabilities()])
  } catch (e: any) {
    error.value =
      e?.response?.status === 404
        ? 'Unknown code family.'
        : (e?.message || 'Failed to load code family')
  } finally {
    loading.value = false
  }
}

async function onGenerateMatrix() {
  if (!family.value) return
  matrixLoading.value = true
  matrixError.value = null
  try {
    await qldpc.loadMatrix(family.value.id)
  } catch (e: any) {
    matrixError.value =
      e?.response?.data?.error || e?.message || 'Failed to generate matrices'
  } finally {
    matrixLoading.value = false
  }
}

async function onComputeDistance(exact: boolean) {
  if (!family.value) return
  distanceLoading.value = true
  distanceError.value = null
  exactRequested.value = exact
  try {
    await qldpc.loadDistance(family.value.id, { exact })
  } catch (e: any) {
    distanceError.value =
      e?.response?.data?.error || e?.message || 'Failed to compute distance'
  } finally {
    distanceLoading.value = false
  }
}

onMounted(() => load(familyId.value))
watch(familyId, (id) => { if (id) load(id) })
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA qLDPC family detail app bar">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— qLDPC</span>
    </div>
    <v-spacer />
    <v-btn variant="text" prepend-icon="mdi-format-list-bulleted" @click="router.push('/qldpc')">
      All code families
    </v-btn>
    <v-btn variant="text" prepend-icon="mdi-school" @click="router.push('/qldpc/learn')">
      Primer
    </v-btn>
    <v-btn variant="text" @click="router.push('/')">Home</v-btn>
  </v-app-bar>

  <v-main>
    <v-container style="max-width: 1080px">

      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />

      <v-alert v-else-if="error" type="error" variant="tonal" class="my-4">
        {{ error }}
        <v-btn class="ml-3" variant="text" @click="router.push('/qldpc')">
          Back to gallery
        </v-btn>
      </v-alert>

      <template v-else-if="family">
        <!-- Header -->
        <div class="pt-6 pb-2 d-flex align-start ga-3 flex-wrap">
          <div class="flex-grow-1">
            <div class="text-overline text-medium-emphasis">
              {{ categoryLabel(family.category) }} · {{ family.regime }}
            </div>
            <div class="text-h4 font-weight-bold">{{ family.title }}</div>
            <div class="text-caption text-medium-emphasis mt-1">
              Introduced by {{ family.discovered_by }}
            </div>
          </div>
          <v-chip color="primary" variant="flat" size="large">
            ⟦{{ family.n }}, {{ family.k }}, {{ family.d ?? '—' }}⟧
          </v-chip>
        </div>

        <!-- Capability strip: only when math stack is missing. -->
        <v-alert
          v-if="qldpc.capabilities && !mathStackReady"
          type="warning"
          variant="tonal"
          class="my-3"
          icon="mdi-package-variant-closed"
        >
          <div class="text-subtitle-1 font-weight-medium">
            Math stack not installed
          </div>
          <div class="text-body-2">
            Parity-check matrix generation, CSS verification, and distance
            computation require the <code>qldpc</code> Python package.
            Install on the backend host:
          </div>
          <code class="d-block mt-2 pa-2 bg-grey-darken-4 rounded">
            pip install ".[qldpc]"
          </code>
        </v-alert>

        <v-alert
          v-else-if="family.live"
          type="info"
          variant="tonal"
          density="compact"
          class="my-3"
          icon="mdi-flask-outline"
        >
          Live parameters from
          <v-chip size="x-small" color="success" class="mx-1">qldpc 0.2.9</v-chip>
          <span class="text-caption text-medium-emphasis ml-1">
            — n, k computed exactly; d is an upper bound from
            <code>get_distance(bound=True)</code> (use <strong>Compute exact distance</strong>
            for the slow exact path).
          </span>
        </v-alert>

        <!-- Parameter strip -->
        <v-row dense class="my-3">
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center">
              <div class="text-overline text-medium-emphasis">Physical qubits</div>
              <div class="text-h5 font-weight-bold">{{ family.n }}</div>
              <div class="text-caption text-medium-emphasis">n</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center">
              <div class="text-overline text-medium-emphasis">Logical qubits</div>
              <div class="text-h5 font-weight-bold">{{ family.k }}</div>
              <div class="text-caption text-medium-emphasis">k</div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center">
              <div class="text-overline text-medium-emphasis">Distance</div>
              <div class="text-h5 font-weight-bold">{{ family.d ?? '—' }}</div>
              <div class="text-caption text-medium-emphasis">
                d ({{ family.d_mode === 'exact' ? 'exact' : 'upper bound' }})
              </div>
            </v-card>
          </v-col>
          <v-col cols="6" sm="3">
            <v-card variant="tonal" class="pa-3 text-center">
              <div class="text-overline text-medium-emphasis">Threshold</div>
              <div class="text-h5 font-weight-bold">
                {{ family.best_known_threshold_pct }}%
              </div>
              <div class="text-caption text-medium-emphasis">best known</div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Summary -->
        <v-card variant="outlined" class="pa-4 my-4">
          <div class="text-subtitle-1 font-weight-medium mb-2">Overview</div>
          <p class="text-body-1">{{ family.summary }}</p>
          <div v-if="family.params" class="mt-3">
            <div class="text-overline text-medium-emphasis">Construction parameters</div>
            <code class="text-caption d-block mt-1">{{ JSON.stringify(family.params) }}</code>
          </div>
        </v-card>

        <!-- Key property + use case -->
        <v-row dense>
          <v-col cols="12" md="6">
            <v-card variant="outlined" class="pa-4 h-100">
              <div class="text-overline text-medium-emphasis">Key property</div>
              <div class="text-body-1 mt-1">{{ family.key_property }}</div>
            </v-card>
          </v-col>
          <v-col cols="12" md="6">
            <v-card variant="outlined" class="pa-4 h-100">
              <div class="text-overline text-medium-emphasis">Where it shines</div>
              <div class="text-body-1 mt-1">{{ family.use_case }}</div>
            </v-card>
          </v-col>
        </v-row>

        <!-- Sprint 1: Parity-check matrices + CSS verification -->
        <v-card variant="outlined" class="pa-4 mt-6">
          <div class="d-flex align-center flex-wrap mb-3 ga-2">
            <div class="text-overline text-medium-emphasis flex-grow-1">
              Sprint 1 — Parity-check matrices
            </div>
            <v-btn
              color="primary"
              variant="flat"
              :loading="matrixLoading"
              :disabled="!mathStackReady"
              prepend-icon="mdi-grid"
              @click="onGenerateMatrix"
            >
              {{ qldpc.currentMatrix ? 'Regenerate' : 'Generate parity-check matrices' }}
            </v-btn>
          </div>

          <v-alert v-if="matrixError" type="error" variant="tonal" density="compact" class="mb-3">
            {{ matrixError }}
          </v-alert>

          <div v-if="qldpc.currentMatrix">
            <!-- CSS commutativity verification -->
            <v-alert
              :type="qldpc.currentMatrix.css_check.commutes ? 'success' : 'error'"
              variant="tonal"
              class="mb-4"
              :icon="qldpc.currentMatrix.css_check.commutes ? 'mdi-check-circle' : 'mdi-alert'"
            >
              <div class="text-subtitle-1 font-weight-medium">
                CSS commutativity check:
                {{ qldpc.currentMatrix.css_check.commutes ? '✓ passes' : '✗ fails' }}
              </div>
              <div class="text-body-2">
                <code>H_X · H_Zᵀ ≡ 0 (mod 2)</code>
                — residual non-zero count:
                <strong>{{ qldpc.currentMatrix.css_check.residual_nonzero_count }}</strong>
                (product shape:
                {{ qldpc.currentMatrix.css_check.product_shape.join(' × ') }})
              </div>
            </v-alert>

            <v-row dense>
              <v-col cols="12" md="6">
                <MatrixViewer
                  :matrix="qldpc.currentMatrix.matrix.matrix_x"
                  title="H_X (X-stabilizer parity-check matrix)"
                  :subtitle="`${qldpc.currentMatrix.matrix.num_checks_x} checks × ${qldpc.currentMatrix.matrix.n} data qubits — detects Z (phase) errors`"
                  fill-color="#42a5f5"
                />
              </v-col>
              <v-col cols="12" md="6">
                <MatrixViewer
                  :matrix="qldpc.currentMatrix.matrix.matrix_z"
                  title="H_Z (Z-stabilizer parity-check matrix)"
                  :subtitle="`${qldpc.currentMatrix.matrix.num_checks_z} checks × ${qldpc.currentMatrix.matrix.n} data qubits — detects X (bit) errors`"
                  fill-color="#ef5350"
                />
              </v-col>
            </v-row>
          </div>

          <div v-else class="text-body-2 text-medium-emphasis">
            Click <strong>Generate parity-check matrices</strong> to compute
            <code>H_X</code> and <code>H_Z</code> from the live
            <code>qldpc</code> library and verify
            <code>H_X · H_Zᵀ = 0 (mod 2)</code> numerically.
          </div>
        </v-card>

        <!-- Sprint 1: Distance computation -->
        <v-card variant="outlined" class="pa-4 mt-4">
          <div class="d-flex align-center flex-wrap mb-3 ga-2">
            <div class="text-overline text-medium-emphasis flex-grow-1">
              Sprint 1 — Code distance
            </div>
            <v-btn
              variant="tonal"
              color="primary"
              :loading="distanceLoading && !exactRequested"
              :disabled="!mathStackReady"
              prepend-icon="mdi-speedometer"
              @click="onComputeDistance(false)"
            >
              Compute upper bound
            </v-btn>
            <v-btn
              variant="outlined"
              color="primary"
              :loading="distanceLoading && exactRequested"
              :disabled="!mathStackReady"
              prepend-icon="mdi-bullseye-arrow"
              @click="onComputeDistance(true)"
            >
              Compute exact (slow)
            </v-btn>
          </div>

          <v-alert v-if="distanceError" type="error" variant="tonal" density="compact" class="mb-3">
            {{ distanceError }}
          </v-alert>

          <div v-if="qldpc.currentDistance">
            <div class="text-h5 font-weight-bold mb-1">
              d = {{ qldpc.currentDistance.distance ?? 'no bound found' }}
            </div>
            <div class="text-body-2 text-medium-emphasis">
              Mode:
              <v-chip size="x-small" class="mx-1" :color="qldpc.currentDistance.mode === 'exact' ? 'success' : 'info'">
                {{ qldpc.currentDistance.mode === 'exact' ? 'exact (ILP-solved)' : 'upper bound (Monte Carlo)' }}
              </v-chip>
              · computed in {{ qldpc.currentDistance.time_ms }} ms
            </div>
            <div v-if="qldpc.currentDistance.mode === 'upper_bound'" class="text-caption text-medium-emphasis mt-2">
              The upper bound is computed from a heuristic search; the
              true distance is ≤ this value. For Surface and Toric codes
              the bound is tight; for HGP and Bicycle (BB) it may
              overshoot the analytic distance reported in the original
              papers.
            </div>
          </div>

          <div v-else class="text-body-2 text-medium-emphasis">
            Distance computation is NP-hard in general. The upper-bound
            path returns in milliseconds; the exact path uses ILP and
            can take seconds-to-minutes depending on the code family.
          </div>
        </v-card>

        <!-- Sprint 2: Tanner graph + routing analysis -->
        <v-card variant="outlined" class="pa-4 mt-4">
          <div class="d-flex align-center flex-wrap mb-3 ga-2">
            <div class="text-overline text-medium-emphasis flex-grow-1">
              Sprint 2 — Tanner graph &amp; routing analysis
            </div>
          </div>
          <div v-if="!mathStackReady" class="text-body-2 text-medium-emphasis">
            Tanner-graph layout + routing metrics require the
            <code>qldpc</code> Python package — see the install banner
            above.
          </div>
          <TannerGraphViewer v-else :family-id="family.id" />
        </v-card>

        <!-- Sprint 3/4 placeholders -->
        <v-card variant="tonal" color="surface-variant" class="pa-4 mt-4">
          <div class="text-overline text-medium-emphasis mb-2">Coming next</div>
          <v-row dense>
            <v-col cols="12" md="6">
              <div class="text-subtitle-2 font-weight-medium">Sprint 3 — Threshold benchmarks</div>
              <div class="text-caption text-medium-emphasis">
                <code>stim</code> Monte Carlo under depolarizing noise;
                physical-to-logical error curves; async via the existing
                RQ queue.
              </div>
            </v-col>
            <v-col cols="12" md="6">
              <div class="text-subtitle-2 font-weight-medium">Sprint 4 — IBMQ hardware</div>
              <div class="text-caption text-medium-emphasis">
                <code>qiskit-qec</code> syndrome circuit compilation +
                BYOK submission reusing the QML app's IBMQ pipeline.
              </div>
            </v-col>
          </v-row>
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
.logo-link:focus-visible {
  outline: 2px solid currentColor;
  outline-offset: 4px;
  border-radius: 4px;
}
</style>
