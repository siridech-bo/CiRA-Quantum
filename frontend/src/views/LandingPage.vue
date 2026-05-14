<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
const auth = useAuthStore()

const ctaLabel = computed(() =>
  auth.user ? 'Open the platform' : 'Sign in to start solving',
)

function goSolve() {
  if (auth.user) {
    router.push('/solve')
  } else {
    router.push({ path: '/login', query: { redirect: '/solve' } })
  }
}

function goBenchmarks() {
  router.push('/benchmarks')
}

// Lightweight tier showcase — built from the registered-solver list
// the backend exposes. We don't make this dynamic on the landing page
// (we want the page to render fast and without auth), so the tier
// breakdown is hardcoded here and updated when new tiers ship.
const tiers = [
  { name: 'Classical SOTA', count: 3, solvers: ['exact_cqm', 'cpsat', 'highs'], color: 'success' },
  { name: 'QUBO heuristic', count: 2, solvers: ['gpu_sa', 'cpu_sa_neal'], color: 'primary' },
  { name: 'Quantum-inspired', count: 2, solvers: ['parallel_tempering', 'simulated_bifurcation'], color: 'info' },
  { name: 'Quantum (simulator)', count: 1, solvers: ['qaoa_sim'], color: 'warning' },
  { name: 'Quantum (real QPU)', count: 1, solvers: ['qaoa_originqc'], color: 'error' },
]
</script>

<template>
  <v-app-bar color="transparent" flat>
    <v-container class="d-flex align-center pa-0">
      <CiraLogo :size="36" />
      <v-spacer />
      <v-btn variant="text" @click="goBenchmarks">Benchmarks</v-btn>
      <v-btn variant="text" @click="router.push('/templates')">Examples</v-btn>
      <v-btn v-if="!auth.user" variant="outlined" class="ml-2" @click="router.push('/login')">Log in</v-btn>
      <v-btn v-else variant="outlined" class="ml-2" @click="goSolve">Open app</v-btn>
    </v-container>
  </v-app-bar>

  <v-main>
    <!-- Hero -->
    <v-container class="hero pt-12 pb-12">
      <v-row align="center" justify="center">
        <v-col cols="12" md="9" lg="8" class="text-center">
          <div class="logo-large mb-6">
            <CiraLogo :size="90" :with-wordmark="true" />
          </div>
          <div class="text-h3 font-weight-bold mb-3">
            An academic home for quantum experimentation
          </div>
          <div class="text-h6 text-medium-emphasis mb-6">
            Independent applications under one platform.
            Real hardware, real baselines, reproducible by design.
          </div>
          <div class="d-flex justify-center ga-3 flex-wrap">
            <v-btn
              color="primary"
              size="x-large"
              variant="flat"
              prepend-icon="mdi-rocket-launch"
              @click="goSolve"
            >
              {{ ctaLabel }}
            </v-btn>
            <v-btn
              size="x-large"
              variant="outlined"
              prepend-icon="mdi-chart-bar"
              @click="goBenchmarks"
            >
              View Benchmarks
            </v-btn>
          </div>
        </v-col>
      </v-row>
    </v-container>

    <!-- Three feature cards: Solve / Learn / Benchmark -->
    <v-container class="pt-8 pb-8">
      <div class="text-h5 mb-1 text-center">Three coordinated surfaces</div>
      <div class="text-body-2 text-medium-emphasis mb-6 text-center">
        Each one stands alone; together they're how the platform delivers
        the v2 "academic platform" promise.
      </div>
      <v-row>
        <v-col cols="12" md="4">
          <v-card class="pa-5 h-100" hover @click="goSolve">
            <v-icon icon="mdi-rocket-launch" size="x-large" color="primary" class="mb-3" />
            <div class="text-h6 mb-2">Solve</div>
            <div class="text-body-2 text-medium-emphasis">
              Paste a problem in plain English. An LLM formulates it as a CQM;
              the platform runs it through the solvers you pick and shows
              you the trained results, side by side.
            </div>
          </v-card>
        </v-col>
        <v-col cols="12" md="4">
          <v-card class="pa-5 h-100" hover @click="router.push('/templates')">
            <v-icon icon="mdi-school" size="x-large" color="accent" class="mb-3" />
            <div class="text-h6 mb-2">Learn</div>
            <div class="text-body-2 text-medium-emphasis">
              Curated example problems with documented optima. Click any of
              them, watch the pipeline run, and compare your solver's answer
              to the textbook value. Modules for structured curriculum use.
            </div>
          </v-card>
        </v-col>
        <v-col cols="12" md="4">
          <v-card class="pa-5 h-100" hover @click="goBenchmarks">
            <v-icon icon="mdi-chart-box" size="x-large" color="info" class="mb-3" />
            <div class="text-h6 mb-2">Benchmark</div>
            <div class="text-body-2 text-medium-emphasis">
              Public, append-only RunRecord archive citable by BibTeX entry.
              Compare every registered solver across every registered
              instance — including real Origin Quantum Wukong results.
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- Solver tier showcase -->
    <v-container class="pt-8 pb-8">
      <div class="text-h5 mb-1 text-center">Five solver-tier categories, populated end-to-end</div>
      <div class="text-body-2 text-medium-emphasis mb-6 text-center">
        Most platforms cover one or two of these. We cover all five — and the dashboard tells you honestly which tier wins on which problem class.
      </div>
      <v-row dense>
        <v-col
          v-for="t in tiers"
          :key="t.name"
          cols="12"
          sm="6"
          md="4"
          lg="auto"
          class="flex-grow-1"
        >
          <v-card
            class="pa-4 h-100"
            variant="tonal"
            :color="t.color"
          >
            <div class="d-flex align-center mb-2">
              <span class="text-subtitle-2 font-weight-bold flex-grow-1">{{ t.name }}</span>
              <v-chip size="x-small" variant="flat">{{ t.count }}</v-chip>
            </div>
            <div class="text-caption">
              <code v-for="(s, i) in t.solvers" :key="s" class="solver-tag">
                {{ s }}<span v-if="i < t.solvers.length - 1">, </span>
              </code>
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- Sister apps strip -->
    <v-container class="pt-8 pb-8">
      <div class="text-h5 mb-1 text-center">Other CiRA platforms</div>
      <div class="text-body-2 text-medium-emphasis mb-6 text-center">
        Independent applications under the same academic umbrella.
      </div>
      <v-row>
        <v-col cols="12" md="6">
          <v-card class="pa-5 h-100" variant="outlined" border="primary">
            <div class="d-flex align-center mb-2">
              <CiraLogo :size="32" :with-wordmark="false" />
              <span class="text-h6 ml-3">CiRA Quantum — Optimization</span>
              <v-spacer />
              <v-chip size="small" color="success">live</v-chip>
            </div>
            <div class="text-body-2 text-medium-emphasis">
              This app. Formulation, solving, benchmarking across 9 solver
              tiers — classical, quantum-inspired, quantum simulator, and
              real superconducting QPU.
            </div>
          </v-card>
        </v-col>
        <v-col cols="12" md="6">
          <v-card class="pa-5 h-100" variant="outlined" border="dashed">
            <div class="d-flex align-center mb-2">
              <v-icon icon="mdi-brain" size="32" color="grey" />
              <span class="text-h6 ml-3 text-medium-emphasis">CiRA Quantum — QML</span>
              <v-spacer />
              <v-chip size="small" variant="tonal">coming soon</v-chip>
            </div>
            <div class="text-body-2 text-medium-emphasis">
              Quantum Machine Learning experiments — variational classifiers,
              quantum kernels, hybrid encoders. Sister application under
              development.
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- Footer -->
    <v-divider class="mt-8" />
    <v-container class="py-6">
      <div class="d-flex align-center flex-wrap ga-3">
        <CiraLogo :size="24" :with-wordmark="false" />
        <div class="text-body-2 text-medium-emphasis flex-grow-1">
          CiRA Quantum — built at KMITL.  An academic platform; not a commercial service.
        </div>
        <v-btn
          variant="text"
          size="small"
          prepend-icon="mdi-github"
          href="https://github.com/siridech-bo/CiRA-Quantum"
          target="_blank"
        >
          GitHub
        </v-btn>
      </div>
    </v-container>
  </v-main>
</template>

<style scoped>
.hero {
  min-height: 60vh;
  display: flex;
  align-items: center;
}
.logo-large {
  display: flex;
  justify-content: center;
}
.solver-tag {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.78rem;
}
</style>
