<script setup lang="ts">
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useBenchmarksStore } from '@/stores/benchmarks'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
const auth = useAuthStore()
const benchmarks = useBenchmarksStore()

onMounted(() => {
  void benchmarks.loadSuites()
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA Quantum Benchmarks app bar">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-body-2 text-medium-emphasis ml-3 d-none d-sm-inline">
        Benchmarks
      </span>
    </div>
    <v-spacer />
    <v-btn
      v-if="auth.user"
      variant="text"
      prepend-icon="mdi-rocket-launch"
      @click="router.push('/solve')"
    >
      Open app
    </v-btn>
    <v-btn
      v-else
      variant="outlined"
      @click="router.push('/login')"
    >
      Log in
    </v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-card class="pa-6 mb-4" variant="tonal">
        <div class="d-flex align-center flex-wrap ga-3">
          <div class="flex-grow-1">
            <div class="text-h5 mb-1">
              The honest scoreboard for quantum + classical optimization
            </div>
            <div class="text-body-2">
              Every benchmark run is reproducible, citable, and append-only.
              Solvers and instances drift over time — that's the signal this
              archive exists to preserve.
            </div>
          </div>
          <v-btn
            color="primary"
            variant="flat"
            prepend-icon="mdi-chart-box"
            @click="router.push('/benchmarks/findings')"
          >
            View Experiment #1 findings
          </v-btn>
        </div>
      </v-card>

      <v-skeleton-loader v-if="benchmarks.loading && !benchmarks.suites.length" type="card, card" />
      <v-alert v-else-if="benchmarks.error" type="error" variant="tonal" class="mb-3">
        {{ benchmarks.error }}
      </v-alert>

      <div class="text-h6 mb-2">Suites</div>
      <v-row>
        <v-col
          v-for="s in benchmarks.suites"
          :key="s.suite_id"
          cols="12"
          md="6"
          lg="4"
        >
          <v-card
            class="pa-4 h-100"
            hover
            :variant="s.num_records ? 'elevated' : 'outlined'"
            @click="router.push(`/benchmarks/suites/${s.suite_id}`)"
          >
            <div class="d-flex align-center mb-1">
              <v-icon icon="mdi-folder-multiple-outline" start />
              <span class="font-weight-medium">{{ s.suite_id }}</span>
              <v-spacer />
              <v-chip size="x-small" :color="s.num_records ? 'primary' : 'default'">
                {{ s.num_records }} records
              </v-chip>
            </div>
            <div class="text-caption text-medium-emphasis mb-2">
              {{
                s.solvers_seen.length
                  ? `${s.solvers_seen.length} solvers seen: ${s.solvers_seen.join(', ')}`
                  : 'No runs archived yet — use the Phase-2 CLI to populate.'
              }}
            </div>
          </v-card>
        </v-col>
      </v-row>
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
