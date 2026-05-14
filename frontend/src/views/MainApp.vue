<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useSolveStore } from '@/stores/solve'
import ProblemInput from '@/components/ProblemInput.vue'
import SolveStatus from '@/components/SolveStatus.vue'
import ResultDisplay from '@/components/ResultDisplay.vue'
import MultiSolverResultDisplay from '@/components/MultiSolverResultDisplay.vue'
import JobHistory from '@/components/JobHistory.vue'
import ApiKeyManager from '@/components/ApiKeyManager.vue'
import TemplateMatchBadge from '@/components/TemplateMatchBadge.vue'
import CiraLogo from '@/components/CiraLogo.vue'

const auth = useAuthStore()
const solve = useSolveStore()
const router = useRouter()

const tab = ref<'solve' | 'history' | 'keys'>('solve')

const hasResult = computed(
  () => solve.currentJob && solve.currentJob.status === 'complete',
)
const hasErrorResult = computed(
  () => solve.currentJob && solve.currentJob.status === 'error',
)

// Show the "try an example" banner when a brand-new user lands with no
// jobs of their own and nothing in flight. Hides itself the moment a
// solve starts or any history exists.
const isNewUser = computed(
  () => solve.history.length === 0 && !solve.currentJob,
)

async function logout() {
  solve.reset()
  await auth.logout()
  router.push('/')
}

onMounted(() => {
  // Cheap probe so the empty-state banner can disappear for returning
  // users without forcing them to click "History" first.
  void solve.loadHistory(1, 1)
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA Quantum app bar">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
    </div>
    <v-spacer />
    <v-btn
      variant="text"
      prepend-icon="mdi-school"
      class="mr-2"
      @click="router.push('/templates')"
    >
      Examples
    </v-btn>
    <v-btn
      variant="text"
      prepend-icon="mdi-chart-bar"
      class="mr-2"
      @click="router.push('/benchmarks')"
    >
      Benchmarks
    </v-btn>
    <span class="text-body-2 mr-4" v-if="auth.user">
      Signed in as <strong>{{ auth.user.display_name }}</strong>
      <v-chip
        v-if="auth.user.role === 'admin'"
        size="x-small"
        color="accent"
        class="ml-2"
      >
        admin
      </v-chip>
    </span>
    <v-btn variant="outlined" @click="logout">Log out</v-btn>
  </v-app-bar>

  <v-main>
    <v-container fluid>
      <v-alert
        v-if="isNewUser && tab === 'solve'"
        type="info"
        variant="tonal"
        prominent
        class="mb-4"
        icon="mdi-school"
      >
        <div class="d-flex align-center flex-wrap">
          <div class="flex-grow-1">
            <div class="text-subtitle-1 font-weight-medium">
              New here? Start with an example.
            </div>
            <div class="text-body-2">
              The Examples gallery has 10 curated optimization problems —
              knapsack, max-cut, TSP, JSS — each with a documented optimum
              you can verify the solver against.
            </div>
          </div>
          <v-btn
            color="primary"
            variant="flat"
            class="ml-3"
            prepend-icon="mdi-school"
            @click="router.push('/templates')"
          >
            Browse examples
          </v-btn>
        </div>
      </v-alert>

      <v-tabs v-model="tab" color="primary" align-tabs="start" class="mb-4">
        <v-tab value="solve">
          <v-icon icon="mdi-rocket-launch" start /> Solve
        </v-tab>
        <v-tab value="history">
          <v-icon icon="mdi-history" start /> History
        </v-tab>
        <v-tab value="keys">
          <v-icon icon="mdi-key" start /> API Keys
        </v-tab>
      </v-tabs>

      <v-window v-model="tab">
        <!-- Solve -->
        <v-window-item value="solve">
          <v-row>
            <v-col cols="12" md="6">
              <ProblemInput />
            </v-col>
            <v-col cols="12" md="6">
              <SolveStatus v-if="solve.currentJob && !hasResult" />
              <template v-else-if="hasResult || hasErrorResult">
                <TemplateMatchBadge
                  v-if="hasResult && solve.currentJob?.template_id"
                  :job="solve.currentJob!"
                  class="mb-3"
                />
                <MultiSolverResultDisplay
                  v-if="hasResult && solve.currentJob?.solver_results"
                  :job="solve.currentJob!"
                />
                <ResultDisplay :job="solve.currentJob!" />
              </template>
              <v-card v-else class="pa-5 h-100">
                <v-card-title class="text-h6 pa-0">Ready when you are</v-card-title>
                <v-card-subtitle class="pa-0 mt-1">
                  Submit a problem on the left. Live progress and the final
                  solution will appear here.
                </v-card-subtitle>
                <v-divider class="my-4" />
                <div class="text-body-2 text-medium-emphasis">
                  <p class="mb-2">
                    The platform runs a five-stage pipeline for every solve:
                    formulate → compile → validate → solve → interpret.
                    A small, single-constraint problem like the canonical
                    knapsack typically completes in 5–10 s end-to-end.
                  </p>
                  <p class="mb-2">
                    For best results on harder problems (JSS, large MIP),
                    pick Claude or OpenAI. The local LLM is best-effort —
                    see the platform README for the per-model tier table.
                  </p>
                </div>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>

        <!-- History -->
        <v-window-item value="history">
          <JobHistory />
        </v-window-item>

        <!-- Keys -->
        <v-window-item value="keys">
          <ApiKeyManager />
        </v-window-item>
      </v-window>
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
