<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useSolveStore } from '@/stores/solve'
import SolveStatus from '@/components/SolveStatus.vue'
import ResultDisplay from '@/components/ResultDisplay.vue'
import MultiSolverResultDisplay from '@/components/MultiSolverResultDisplay.vue'
import TemplateMatchBadge from '@/components/TemplateMatchBadge.vue'
import CiraLogo from '@/components/CiraLogo.vue'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const solve = useSolveStore()

const error = ref<string | null>(null)
const loading = ref(true)

async function load() {
  error.value = null
  loading.value = true
  try {
    const jobId = String(route.params.id || '')
    await Promise.all([solve.subscribeToJob(jobId), solve.loadSolvers()])
  } catch (e: any) {
    error.value =
      e?.response?.status === 404
        ? 'Job not found (or not yours).'
        : e?.response?.data?.error || e?.message || 'Failed to load job'
  } finally {
    loading.value = false
  }
}

async function logout() {
  solve.reset()
  await auth.logout()
  router.push('/')
}

onMounted(load)
onBeforeUnmount(() => {
  // Tear down the SSE so navigating away doesn't leak the EventSource.
  solve.closeStream()
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA Quantum app bar">
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/solve')" />
    <div
      class="d-flex align-center logo-link"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
    </div>
    <v-spacer />
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
    <v-container>
      <v-skeleton-loader v-if="loading" type="article" />
      <v-alert v-else-if="error" type="error" variant="tonal">
        {{ error }}
      </v-alert>
      <template v-else-if="solve.currentJob">
        <v-card class="pa-5 mb-4">
          <v-card-title class="pa-0 text-h6 mb-1">Problem statement</v-card-title>
          <v-card-subtitle class="pa-0 mb-3 text-medium-emphasis">
            Job <code>{{ solve.currentJob.id }}</code> · submitted
            {{ new Date(solve.currentJob.created_at).toLocaleString() }}
            via {{ solve.currentJob.provider }}
          </v-card-subtitle>
          <pre class="problem-statement">{{ solve.currentJob.problem_statement }}</pre>
        </v-card>

        <TemplateMatchBadge
          v-if="
            solve.currentJob.status === 'complete' &&
            solve.currentJob.template_id
          "
          :job="solve.currentJob"
          class="mb-3"
        />
        <SolveStatus v-if="!['complete', 'error'].includes(solve.currentJob.status)" />
        <template v-else>
          <MultiSolverResultDisplay
            v-if="solve.currentJob.status === 'complete' && solve.currentJob.solver_results"
            :job="solve.currentJob"
          />
          <ResultDisplay :job="solve.currentJob" />
        </template>
      </template>
    </v-container>
  </v-main>
</template>

<style scoped>
.problem-statement {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 0.75rem;
  font-family: inherit;
  white-space: pre-wrap;
  word-wrap: break-word;
}
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
