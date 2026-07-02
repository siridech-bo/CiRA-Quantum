<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import CiraLogo from '@/components/CiraLogo.vue'
import AuthDialog from '@/components/AuthDialog.vue'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

// Auth dialog wiring. The dialog handles both login and signup via a
// mode prop; opening it from either app-bar button just sets the mode
// before flipping the model. The auth guard bounces protected routes
// back to ``/`` with a ``?auth=login`` query — we react to that on
// mount + on watch so a user who tried to visit /solve unauthenticated
// gets the login form immediately without a manual click.
const authDialogOpen = ref(false)
const authMode = ref<'login' | 'signup'>('login')

function openLogin() {
  authMode.value = 'login'
  authDialogOpen.value = true
}
function openSignup() {
  authMode.value = 'signup'
  authDialogOpen.value = true
}

function onAuthenticated() {
  // If the auth guard redirected here from a protected route, follow
  // through. Otherwise stay on the landing page — the user menu now
  // shows their identity and their next click decides where to go.
  const redirect = route.query.redirect as string | undefined
  if (redirect) {
    router.push(redirect)
    return
  }
  // Clear the ``auth=login`` query so a hard refresh doesn't re-open
  // the dialog.
  if (route.query.auth) {
    router.replace({ path: '/', query: {} })
  }
}

function maybeAutoOpenDialog() {
  if (!auth.user && route.query.auth === 'login') openLogin()
  if (!auth.user && route.query.auth === 'signup') openSignup()
}

onMounted(maybeAutoOpenDialog)
watch(() => route.query.auth, maybeAutoOpenDialog)

function openOptimization() {
  if (auth.user) {
    router.push('/solve')
  } else {
    // Not signed in — pop the login dialog inline instead of routing.
    router.replace({ path: '/', query: { auth: 'login', redirect: '/solve' } })
    openLogin()
  }
}

async function logout() {
  await auth.logout()
}

function openQml() {
  // QML gallery is public — no login required to browse.
  router.push('/qml')
}

function openQldpc() {
  // qLDPC code-family gallery is public — same logic as QML.
  router.push('/qldpc')
}

// Per-app tier showcases. Hardcoded so the landing page renders fast
// and without auth; updated when new tiers ship.
const optSolverTiers = [
  { name: 'Classical SOTA',      count: 3, solvers: ['exact_cqm', 'cpsat', 'highs'],                color: 'success' },
  { name: 'QUBO heuristic',      count: 2, solvers: ['gpu_sa', 'cpu_sa_neal'],                       color: 'primary' },
  { name: 'Quantum-inspired',    count: 2, solvers: ['parallel_tempering', 'simulated_bifurcation'], color: 'info' },
  { name: 'Quantum (simulator)', count: 1, solvers: ['qaoa_sim'],                                    color: 'warning' },
  { name: 'Quantum (real QPU)',  count: 2, solvers: ['qaoa_originqc', 'qaoa_ibmq'],                  color: 'error' },
]

const qmlModelTiers = [
  { name: 'Variational quantum',  count: 1, items: ['vqc (statevector)'],                                color: 'accent'  },
  { name: 'Classical baselines',  count: 4, items: ['logreg', 'svm_rbf', 'random_forest', 'mlp'],         color: 'primary' },
  { name: 'Datasets',             count: 6, items: ['moons', 'circles', 'iris', 'wine', 'mnist_0v1', 'breast_cancer'], color: 'info' },
  { name: 'Quantum (real QPU)',   count: 2, items: ['vqc_ibmq', 'vqc_originqc'],                          color: 'error'   },
]

// qLDPC Sprint 0 — four code families across two regimes. Sprint 3
// adds the threshold-benchmark tier; Sprint 4 adds the QPU-execution
// tier. "soon" rendering in the template keys off count === 0.
const qldpcCodeTiers = [
  { name: 'Topological codes',    count: 2, items: ['surface', 'toric'],                                color: 'info'    },
  { name: 'CSS (classical)',      count: 1, items: ['bicycle'],                                         color: 'success' },
  { name: 'CSS (product)',        count: 1, items: ['hypergraph_product'],                              color: 'accent'  },
  { name: 'Threshold benchmarks', count: 0, items: ['stim Monte Carlo (Sprint 3)'],                     color: 'warning' },
  { name: 'Hardware execution',   count: 0, items: ['qiskit-qec on IBMQ (Sprint 4)'],                   color: 'error'   },
]
</script>

<template>
  <v-app-bar color="transparent" flat>
    <v-container class="d-flex align-center pa-0">
      <CiraLogo :size="36" />
      <v-spacer />
      <v-menu>
        <template #activator="{ props: act }">
          <v-btn variant="text" v-bind="act">
            Optimization
            <v-icon icon="mdi-menu-down" end />
          </v-btn>
        </template>
        <v-list density="compact">
          <v-list-item @click="openOptimization">
            <v-list-item-title>Open app</v-list-item-title>
          </v-list-item>
          <v-list-item @click="router.push('/templates')">
            <v-list-item-title>Examples</v-list-item-title>
          </v-list-item>
          <v-list-item @click="router.push('/benchmarks')">
            <v-list-item-title>Benchmarks</v-list-item-title>
          </v-list-item>
          <v-list-item @click="router.push('/learn/quantum')">
            <v-list-item-title>Quantum 101</v-list-item-title>
          </v-list-item>
          <v-list-item @click="router.push('/learn/playground')">
            <v-list-item-title>Playground</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu>
      <v-menu>
        <template #activator="{ props: act }">
          <v-btn variant="text" v-bind="act">
            QML
            <v-icon icon="mdi-menu-down" end />
          </v-btn>
        </template>
        <v-list density="compact">
          <v-list-item @click="openQml">
            <v-list-item-title>Dataset gallery</v-list-item-title>
          </v-list-item>
          <v-list-item @click="router.push('/qml/learn')">
            <v-list-item-title>Primer</v-list-item-title>
          </v-list-item>
          <v-list-item @click="router.push('/qml/benchmarks')">
            <v-list-item-title>Benchmarks</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu>
      <v-menu>
        <template #activator="{ props: act }">
          <v-btn variant="text" v-bind="act">
            qLDPC
            <v-icon icon="mdi-menu-down" end />
          </v-btn>
        </template>
        <v-list density="compact">
          <v-list-item @click="openQldpc">
            <v-list-item-title>Code-family gallery</v-list-item-title>
          </v-list-item>
          <v-list-item @click="router.push('/qldpc/learn')">
            <v-list-item-title>Primer</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu>
      <!-- Auth-state-dependent controls. The dialog-based flow keeps
           the user on ``/`` throughout auth — no route churn, no lost
           scroll. Signed-in users get a compact identity chip plus a
           menu with Settings and Log out; the "Open Optimization"
           entry moves inside the menu so the app bar stays uncluttered. -->
      <template v-if="!auth.user">
        <v-btn
          variant="text"
          class="ml-2"
          @click="openLogin"
        >Log in</v-btn>
        <v-btn
          variant="outlined"
          class="ml-2"
          @click="openSignup"
        >Sign up</v-btn>
      </template>
      <template v-else>
        <v-btn
          variant="outlined"
          class="ml-2"
          @click="openOptimization"
        >Open Optimization</v-btn>
        <v-menu>
          <template #activator="{ props: act }">
            <v-btn
              variant="text"
              icon
              v-bind="act"
              class="ml-2"
              :aria-label="`Signed in as ${auth.user.display_name}`"
            >
              <v-avatar size="34" color="primary" variant="tonal">
                <span class="text-body-2 font-weight-bold">
                  {{ auth.user.display_name?.[0]?.toUpperCase() || 'U' }}
                </span>
              </v-avatar>
            </v-btn>
          </template>
          <v-list density="compact" min-width="220">
            <v-list-item>
              <v-list-item-title>
                {{ auth.user.display_name }}
                <v-chip
                  v-if="auth.user.role === 'admin'"
                  size="x-small"
                  color="accent"
                  class="ml-1"
                >admin</v-chip>
              </v-list-item-title>
              <v-list-item-subtitle>
                @{{ auth.user.username }}
              </v-list-item-subtitle>
            </v-list-item>
            <v-divider />
            <v-list-item
              prepend-icon="mdi-cog-outline"
              @click="router.push('/settings')"
            >
              <v-list-item-title>Settings</v-list-item-title>
            </v-list-item>
            <v-list-item
              v-if="auth.user.role === 'admin'"
              prepend-icon="mdi-shield-account"
              @click="router.push('/admin')"
            >
              <v-list-item-title>Admin</v-list-item-title>
            </v-list-item>
            <v-divider />
            <v-list-item
              prepend-icon="mdi-logout"
              @click="logout"
            >
              <v-list-item-title>Log out</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-menu>
      </template>
    </v-container>
  </v-app-bar>

  <!-- The single authentication surface on the platform. Mounted here
       so both direct-clicks (Log in / Sign up buttons above) and
       redirected clicks (?auth=login from the router guard) flow
       through one component. -->
  <AuthDialog
    v-model="authDialogOpen"
    v-model:mode="authMode"
    @authenticated="onAuthenticated"
  />

  <v-main>
    <!-- Hero — app-agnostic now. The two-app showcase below IS the
         primary action; the hero just frames what the platform is. -->
    <v-container class="hero pt-12 pb-8">
      <v-row align="center" justify="center">
        <v-col cols="12" md="9" lg="8" class="text-center">
          <div class="logo-large mb-6">
            <CiraLogo :size="90" :with-wordmark="true" />
          </div>
          <div class="text-h3 font-weight-bold mb-3">
            Quantum applications in one academic platform.
          </div>
          <div class="text-h6 text-medium-emphasis mb-2" style="max-width: 760px; margin: 0 auto">
            <strong>Optimization</strong> for solving combinatorial problems with
            quantum + classical solvers side-by-side.
            <strong>QML</strong> for training variational classifiers
            with classical baselines on every dataset.
            <strong>qLDPC</strong> for designing, routing, and benchmarking
            quantum error-correction codes end-to-end.
          </div>
          <div class="text-body-2 text-medium-emphasis mt-3">
            Real hardware, real baselines, reproducible by design.
          </div>
        </v-col>
      </v-row>
    </v-container>

    <!-- ============================================================
         The application showcase — the centerpiece of the page.
         Each card mirrors the others so they read as siblings, not as
         "main app + side projects."
         ============================================================ -->
    <v-container class="pt-4 pb-8" id="apps">
      <v-row>
        <!-- ====== Optimization ====== -->
        <v-col cols="12" md="4">
          <v-card
            class="pa-6 h-100 app-card"
            variant="outlined"
            border="primary"
          >
            <div class="d-flex align-center mb-3">
              <v-icon icon="mdi-rocket-launch" color="primary" size="36" />
              <div class="ml-3 flex-grow-1">
                <div class="text-overline text-medium-emphasis">application</div>
                <div class="text-h5">CiRA Quantum — Optimization</div>
              </div>
              <v-chip size="small" color="success" variant="flat">live</v-chip>
            </div>
            <div class="text-body-1 text-medium-emphasis mb-4">
              Paste a combinatorial problem in plain English. An LLM
              formulates it as a Constrained Quadratic Model; the platform
              runs every registered solver tier on it — exact, heuristic,
              quantum-inspired, simulator, real QPU — and shows you the
              results side-by-side.
            </div>

            <div class="text-overline text-medium-emphasis mb-1">what's inside</div>
            <v-list density="compact" class="pa-0 mb-3">
              <v-list-item class="px-0" prepend-icon="mdi-format-list-bulleted">
                <v-list-item-title class="text-body-2">
                  10 solvers across 5 tiers (classical → real QPU)
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-school">
                <v-list-item-title class="text-body-2">
                  Curated example library with documented optima
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-chart-box">
                <v-list-item-title class="text-body-2">
                  Public benchmark archive — every run cited by BibTeX
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-atom">
                <v-list-item-title class="text-body-2">
                  Real Origin Wukong + IBM Quantum submissions via BYOK
                </v-list-item-title>
              </v-list-item>
            </v-list>

            <div class="d-flex ga-2 flex-wrap">
              <v-btn
                color="primary"
                variant="flat"
                prepend-icon="mdi-rocket-launch"
                @click="openOptimization"
              >
                {{ auth.user ? 'Open' : 'Sign in to solve' }}
              </v-btn>
              <v-btn
                variant="outlined"
                prepend-icon="mdi-chart-bar"
                @click="router.push('/benchmarks')"
              >
                Benchmarks
              </v-btn>
              <v-btn
                variant="text"
                prepend-icon="mdi-school"
                @click="router.push('/templates')"
              >
                Examples
              </v-btn>
            </div>
          </v-card>
        </v-col>

        <!-- ====== QML ====== -->
        <v-col cols="12" md="4">
          <v-card
            class="pa-6 h-100 app-card"
            variant="outlined"
            border="accent"
          >
            <div class="d-flex align-center mb-3">
              <v-icon icon="mdi-brain" color="accent" size="36" />
              <div class="ml-3 flex-grow-1">
                <div class="text-overline text-medium-emphasis">application</div>
                <div class="text-h5">CiRA Quantum — QML</div>
              </div>
              <v-chip size="small" color="accent" variant="flat">live</v-chip>
            </div>
            <div class="text-body-1 text-medium-emphasis mb-4">
              Train a Variational Quantum Classifier on a curated dataset.
              The same train/test split is also handed to four classical
              baselines (LogReg, SVM-RBF, Random Forest, MLP), so every
              result is an honest head-to-head between quantum and classical.
            </div>

            <div class="text-overline text-medium-emphasis mb-1">what's inside</div>
            <v-list density="compact" class="pa-0 mb-3">
              <v-list-item class="px-0" prepend-icon="mdi-database">
                <v-list-item-title class="text-body-2">
                  6 datasets (Moons, Circles, Iris, Wine, MNIST 0v1, WDBC)
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-atom-variant">
                <v-list-item-title class="text-body-2">
                  PennyLane statevector simulator, exact gradients
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-vector-curve">
                <v-list-item-title class="text-body-2">
                  Live decision boundary + loss curve during training
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-school">
                <v-list-item-title class="text-body-2">
                  Interactive primer (Bloch sphere → measurement) — 5 min read
                </v-list-item-title>
              </v-list-item>
            </v-list>

            <div class="d-flex ga-2 flex-wrap">
              <v-btn
                color="accent"
                variant="flat"
                prepend-icon="mdi-brain"
                @click="openQml"
              >
                Open QML
              </v-btn>
              <v-btn
                variant="outlined"
                prepend-icon="mdi-school"
                @click="router.push('/qml/learn')"
              >
                Primer
              </v-btn>
              <v-btn
                variant="text"
                prepend-icon="mdi-database"
                @click="router.push('/qml/datasets/moons')"
              >
                Try Two Moons
              </v-btn>
            </div>
          </v-card>
        </v-col>

        <!-- ====== qLDPC ====== -->
        <v-col cols="12" md="4">
          <v-card
            class="pa-6 h-100 app-card"
            variant="outlined"
            border="info"
          >
            <div class="d-flex align-center mb-3">
              <v-icon icon="mdi-grid" color="info" size="36" />
              <div class="ml-3 flex-grow-1">
                <div class="text-overline text-medium-emphasis">application</div>
                <div class="text-h5">CiRA Quantum — qLDPC</div>
              </div>
              <v-chip size="small" color="info" variant="flat">new</v-chip>
            </div>
            <div class="text-body-1 text-medium-emphasis mb-4">
              Inspect, route, and execute quantum low-density parity-check
              codes end-to-end — from algebraic generation of the
              parity-check matrices, through Tanner-graph layout on a
              physical chip, to syndrome-extraction circuits running on
              real IBMQ hardware.
            </div>

            <div class="text-overline text-medium-emphasis mb-1">what's inside</div>
            <v-list density="compact" class="pa-0 mb-3">
              <v-list-item class="px-0" prepend-icon="mdi-grid">
                <v-list-item-title class="text-body-2">
                  4 code families (Bicycle, Surface, Hypergraph product, Toric)
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-graph-outline">
                <v-list-item-title class="text-body-2">
                  Tanner-graph layout + routing analysis (Sprint 2)
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-chart-bell-curve">
                <v-list-item-title class="text-body-2">
                  stim Monte Carlo threshold benchmarks (Sprint 3)
                </v-list-item-title>
              </v-list-item>
              <v-list-item class="px-0" prepend-icon="mdi-atom">
                <v-list-item-title class="text-body-2">
                  qiskit-qec syndrome circuits on IBMQ hardware (Sprint 4)
                </v-list-item-title>
              </v-list-item>
            </v-list>

            <div class="d-flex ga-2 flex-wrap">
              <v-btn
                color="info"
                variant="flat"
                prepend-icon="mdi-grid"
                @click="openQldpc"
              >
                Open qLDPC
              </v-btn>
              <v-btn
                variant="outlined"
                prepend-icon="mdi-school"
                @click="router.push('/qldpc/learn')"
              >
                Primer
              </v-btn>
              <v-btn
                variant="text"
                prepend-icon="mdi-grid"
                @click="router.push('/qldpc/codes/surface')"
              >
                See Surface code
              </v-btn>
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- ============================================================
         Per-app deep-dive #1 — Optimization solver tiers.
         ============================================================ -->
    <v-container class="pt-8 pb-8">
      <div class="d-flex align-center mb-2">
        <v-icon icon="mdi-rocket-launch" color="primary" class="mr-2" />
        <div class="text-h6">Optimization · five solver-tier categories</div>
      </div>
      <div class="text-body-2 text-medium-emphasis mb-4">
        Most platforms cover one or two. We cover all five — and the
        dashboard tells you honestly which tier wins on which problem class.
      </div>
      <v-row dense>
        <v-col
          v-for="t in optSolverTiers"
          :key="t.name"
          cols="12"
          sm="6"
          md="4"
          lg="auto"
          class="flex-grow-1"
        >
          <v-card class="pa-4 h-100" variant="tonal" :color="t.color">
            <div class="d-flex align-center mb-2">
              <span class="text-subtitle-2 font-weight-bold flex-grow-1">
                {{ t.name }}
              </span>
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

    <!-- ============================================================
         Per-app deep-dive #2 — QML model/dataset matrix.
         ============================================================ -->
    <v-container class="pt-8 pb-8">
      <div class="d-flex align-center mb-2">
        <v-icon icon="mdi-brain" color="accent" class="mr-2" />
        <div class="text-h6">QML · the comparison matrix</div>
      </div>
      <div class="text-body-2 text-medium-emphasis mb-4">
        One training run produces five trained models on the same split:
        the VQC under study plus four classical baselines. Real QPU model
        tiers arrive in QML-5 / 6.
      </div>
      <v-row dense>
        <v-col
          v-for="t in qmlModelTiers"
          :key="t.name"
          cols="12"
          sm="6"
          md="3"
          class="flex-grow-1"
        >
          <v-card class="pa-4 h-100" variant="tonal" :color="t.color">
            <div class="d-flex align-center mb-2">
              <span class="text-subtitle-2 font-weight-bold flex-grow-1">
                {{ t.name }}
              </span>
              <v-chip size="x-small" variant="flat">
                {{ t.count > 0 ? t.count : 'soon' }}
              </v-chip>
            </div>
            <div class="text-caption">
              <code v-for="(s, i) in t.items" :key="s" class="solver-tag">
                {{ s }}<span v-if="i < t.items.length - 1">, </span>
              </code>
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- ============================================================
         Per-app deep-dive #3 — qLDPC code-family categories.
         ============================================================ -->
    <v-container class="pt-8 pb-8">
      <div class="d-flex align-center mb-2">
        <v-icon icon="mdi-grid" color="info" class="mr-2" />
        <div class="text-h6">qLDPC · five module tiers</div>
      </div>
      <div class="text-body-2 text-medium-emphasis mb-4">
        Sprint 0 ships the first three tiers (the code-family
        gallery + canonical metadata). Sprint 3 lights up threshold
        benchmarks; Sprint 4 lights up real-QPU syndrome execution.
        "soon" tiers indicate what's already roadmapped.
      </div>
      <v-row dense>
        <v-col
          v-for="t in qldpcCodeTiers"
          :key="t.name"
          cols="12"
          sm="6"
          md="4"
          lg="auto"
          class="flex-grow-1"
        >
          <v-card class="pa-4 h-100" variant="tonal" :color="t.color">
            <div class="d-flex align-center mb-2">
              <span class="text-subtitle-2 font-weight-bold flex-grow-1">
                {{ t.name }}
              </span>
              <v-chip size="x-small" variant="flat">
                {{ t.count > 0 ? t.count : 'soon' }}
              </v-chip>
            </div>
            <div class="text-caption">
              <code v-for="(s, i) in t.items" :key="s" class="solver-tag">
                {{ s }}<span v-if="i < t.items.length - 1">, </span>
              </code>
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- ============================================================
         What all apps share (the platform layer).
         ============================================================ -->
    <v-container class="pt-8 pb-12">
      <div class="text-h6 text-center mb-2">What all apps share</div>
      <div class="text-body-2 text-medium-emphasis mb-6 text-center">
        One account, one BYOK keyring, one auth surface. Each app is
        a first-class citizen — but you don't sign up three times.
      </div>
      <v-row>
        <v-col cols="12" sm="6" md="3">
          <v-card class="pa-4 h-100 text-center" variant="tonal">
            <v-icon icon="mdi-account-circle" size="36" color="primary" class="mb-2" />
            <div class="text-subtitle-2 mb-1">One account</div>
            <div class="text-caption text-medium-emphasis">
              Same login for both apps; history stays separate per app.
            </div>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card class="pa-4 h-100 text-center" variant="tonal">
            <v-icon icon="mdi-key" size="36" color="accent" class="mb-2" />
            <div class="text-subtitle-2 mb-1">Shared BYOK</div>
            <div class="text-caption text-medium-emphasis">
              Your IBM Quantum / Origin keys work in both apps.
            </div>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card class="pa-4 h-100 text-center" variant="tonal">
            <v-icon icon="mdi-flask-outline" size="36" color="info" class="mb-2" />
            <div class="text-subtitle-2 mb-1">Reproducible</div>
            <div class="text-caption text-medium-emphasis">
              Every run pins seeds, library versions, and split indices.
            </div>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card class="pa-4 h-100 text-center" variant="tonal">
            <v-icon icon="mdi-school" size="36" color="success" class="mb-2" />
            <div class="text-subtitle-2 mb-1">Educational</div>
            <div class="text-caption text-medium-emphasis">
              Every gate, every parameter, every baseline visible.
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- Footer -->
    <v-divider />
    <v-container class="py-6">
      <div class="d-flex align-center flex-wrap ga-3">
        <CiraLogo :size="24" :with-wordmark="false" />
        <div class="text-body-2 text-medium-emphasis flex-grow-1">
          CiRA Quantum — built at KMITL. An academic platform; not a commercial service.
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
  min-height: 42vh;
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
.app-card {
  transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
}
.app-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
}
</style>
