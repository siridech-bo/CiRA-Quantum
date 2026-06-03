<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useQldpcStore, type QldpcCodeFamily } from '@/stores/qldpc'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
const auth = useAuthStore()
const qldpc = useQldpcStore()

const loading = ref(true)
const error = ref<string | null>(null)

const mathStackReady = computed(
  () => qldpc.capabilities?.qldpc_lib && qldpc.capabilities?.networkx,
)

const categoryColor = (c: QldpcCodeFamily['category']) => {
  switch (c) {
    case 'topological': return 'info'
    case 'css_classical': return 'success'
    case 'css_product': return 'accent'
    default: return 'grey'
  }
}

const categoryLabel = (c: QldpcCodeFamily['category']) => {
  switch (c) {
    case 'topological': return 'Topological'
    case 'css_classical': return 'CSS (classical)'
    case 'css_product': return 'CSS (product)'
    default: return c
  }
}

onMounted(async () => {
  try {
    await Promise.all([qldpc.loadCodeFamilies(), qldpc.loadCapabilities()])
  } catch (e: any) {
    error.value = e?.message || 'Failed to load qLDPC code families'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA qLDPC app bar">
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
    <v-btn variant="text" prepend-icon="mdi-school" @click="router.push('/qldpc/learn')">
      Primer
    </v-btn>
    <v-btn variant="text" @click="router.push('/')">Home</v-btn>
    <v-btn variant="text" @click="router.push('/qml')">QML app</v-btn>
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
        <div class="text-overline text-medium-emphasis">qLDPC Sprint 0 — code-family scaffolding</div>
        <div class="text-h4 font-weight-bold mb-2">
          Quantum LDPC codes, end-to-end
        </div>
        <div class="text-body-1 text-medium-emphasis" style="max-width: 760px">
          Inspect, route, and execute quantum low-density parity-check
          codes from algebraic generation through Tanner-graph layout to
          syndrome extraction on real hardware. Pick a code family to
          start — Sprint 1 adds real matrix generation and distance
          verification, Sprint 4 adds IBMQ hardware submission.
        </div>
      </div>

      <!-- Capability strip — currently always grey in Sprint 0 -->
      <v-alert
        v-if="qldpc.capabilities && !mathStackReady"
        type="warning"
        variant="tonal"
        class="my-4"
        icon="mdi-package-variant-closed"
      >
        <div class="text-subtitle-1 font-weight-medium">
          Math stack not yet installed
        </div>
        <div class="text-body-2">
          Sprint 0 ships read-only scaffolding. To enable matrix
          generation, distance computation, and Tanner-graph export
          (Sprint 1), install the optional extra:
        </div>
        <code class="d-block mt-2 pa-2 bg-grey-darken-4 rounded">
          pip install ".[qldpc]"
        </code>
      </v-alert>

      <v-alert
        v-else-if="qldpc.capabilities && mathStackReady"
        type="info"
        variant="tonal"
        class="my-4"
        icon="mdi-flask-outline"
        density="compact"
      >
        Math stack ready —
        <v-chip size="x-small" color="success" class="mx-1">qldpc</v-chip>
        <v-chip size="x-small" color="success" class="mx-1">networkx</v-chip>
        <v-chip
          size="x-small"
          :color="qldpc.capabilities.stim ? 'success' : 'grey'"
          class="mx-1"
        >
          stim
          <span v-if="!qldpc.capabilities.stim"> (Sprint 3)</span>
        </v-chip>
        <v-chip
          size="x-small"
          :color="qldpc.capabilities.qiskit_qec ? 'success' : 'grey'"
          class="mx-1"
        >
          qiskit-qec
          <span v-if="!qldpc.capabilities.qiskit_qec"> (Sprint 4)</span>
        </v-chip>
      </v-alert>

      <!-- Primer funnel — first-time visitors hit the conceptual primer
           before they spend time staring at parity-check matrices. -->
      <v-card
        variant="tonal"
        color="accent"
        class="pa-4 my-4 d-flex align-center flex-wrap ga-3 primer-card"
        hover
        @click="router.push('/qldpc/learn')"
      >
        <v-icon icon="mdi-school" size="32" />
        <div class="flex-grow-1">
          <div class="text-subtitle-1 font-weight-medium">
            New to quantum error correction? Start here.
          </div>
          <div class="text-body-2 text-medium-emphasis">
            A five-step primer: stabilizer codes → CSS construction →
            Tanner graphs → distance &amp; threshold → why qLDPC beats
            surface codes. ~7 min read.
          </div>
        </div>
        <v-btn
          variant="flat"
          color="accent"
          prepend-icon="mdi-arrow-right"
          @click.stop="router.push('/qldpc/learn')"
        >
          Open primer
        </v-btn>
      </v-card>

      <!-- Code family gallery -->
      <div class="d-flex align-center mt-6 mb-2">
        <div class="text-h6">Code family gallery</div>
        <v-spacer />
        <v-chip size="small" variant="tonal">
          {{ qldpc.codeFamilies.length }} families
        </v-chip>
      </div>
      <div class="text-body-2 text-medium-emphasis mb-4">
        Four canonical qLDPC code families spanning the design space:
        zero-rate topological codes that win on chip simplicity, and
        finite-rate CSS codes that win on qubit overhead. Click a card
        to inspect parameters; Sprint 1 will fill in the parity-check
        matrices.
      </div>

      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />
      <v-alert v-else-if="error" type="error" variant="tonal">{{ error }}</v-alert>

      <v-row v-else dense>
        <v-col
          v-for="f in qldpc.codeFamilies"
          :key="f.id"
          cols="12"
          sm="6"
          md="6"
        >
          <v-card
            class="pa-4 h-100 family-card"
            variant="outlined"
            hover
            @click="router.push(`/qldpc/codes/${f.id}`)"
          >
            <div class="d-flex align-center mb-2">
              <span class="text-subtitle-1 font-weight-medium flex-grow-1">
                {{ f.title }}
              </span>
              <v-chip
                size="x-small"
                :color="categoryColor(f.category)"
                variant="flat"
              >
                {{ categoryLabel(f.category) }}
              </v-chip>
            </div>
            <div class="d-flex ga-1 mb-2 flex-wrap">
              <v-chip size="x-small" variant="outlined">{{ f.regime }}</v-chip>
              <v-chip size="x-small" variant="outlined">
                ⟦{{ f.n }}, {{ f.k }}, {{ f.d }}⟧
              </v-chip>
              <v-chip size="x-small" variant="outlined">
                threshold ≈ {{ f.best_known_threshold_pct }}%
              </v-chip>
            </div>
            <div class="text-body-2 text-medium-emphasis">{{ f.summary }}</div>
            <v-divider class="my-3" />
            <div class="d-flex align-center ga-2">
              <code class="text-caption text-medium-emphasis flex-grow-1">
                {{ f.discovered_by }}
              </code>
              <v-btn
                size="small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-magnify"
                @click.stop="router.push(`/qldpc/codes/${f.id}`)"
              >
                Inspect
              </v-btn>
            </div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Roadmap teaser — shows the customers what's coming -->
      <v-card variant="outlined" class="mt-8 pa-4">
        <div class="text-overline text-medium-emphasis mb-2">Roadmap</div>
        <div class="text-h6 mb-3">What's shipping next</div>
        <v-row dense>
          <v-col cols="12" sm="6" md="3">
            <div class="text-subtitle-2 font-weight-medium">Sprint 1</div>
            <div class="text-caption text-medium-emphasis">
              Real matrix generation via the <code>qldpc</code> Python lib;
              distance computation; <code>H_X · H_Zᵀ = 0</code> CSS check;
              Tanner-graph JSON export.
            </div>
          </v-col>
          <v-col cols="12" sm="6" md="3">
            <div class="text-subtitle-2 font-weight-medium">Sprint 2</div>
            <div class="text-caption text-medium-emphasis">
              Interactive Tanner-graph viewer; 2D/3D layout optimization;
              edge-length and crossing analysis for routing feasibility.
            </div>
          </v-col>
          <v-col cols="12" sm="6" md="3">
            <div class="text-subtitle-2 font-weight-medium">Sprint 3</div>
            <div class="text-caption text-medium-emphasis">
              <code>stim</code> Monte Carlo benchmarks under depolarizing
              noise; physical-to-logical threshold curves; async via the
              existing RQ queue.
            </div>
          </v-col>
          <v-col cols="12" sm="6" md="3">
            <div class="text-subtitle-2 font-weight-medium">Sprint 4</div>
            <div class="text-caption text-medium-emphasis">
              <code>qiskit-qec</code> syndrome-extraction circuit
              compilation; IBMQ hardware submission reusing the QML
              app's BYOK token flow.
            </div>
          </v-col>
        </v-row>
      </v-card>

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
.family-card {
  cursor: pointer;
}
.primer-card {
  cursor: pointer;
}
</style>
