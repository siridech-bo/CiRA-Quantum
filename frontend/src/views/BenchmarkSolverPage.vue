<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useBenchmarksStore } from '@/stores/benchmarks'
import CitationButton from '@/components/CitationButton.vue'
import ReproducibilityBadge from '@/components/ReproducibilityBadge.vue'
import CiraLogo from '@/components/CiraLogo.vue'

const route = useRoute()
const router = useRouter()
const benchmarks = useBenchmarksStore()

const solverName = computed(() => String(route.params.solverName || ''))

async function load() {
  if (!solverName.value) return
  await benchmarks.loadSolver(solverName.value)
}

onMounted(load)
watch(solverName, load)

const detail = computed(() => benchmarks.solverDetail[solverName.value] || null)

function totalRecords(): number {
  if (!detail.value) return 0
  return Object.values(detail.value.records_by_suite).reduce((acc, recs) => acc + recs.length, 0)
}

function fmtTime(ms: number | null | undefined): string {
  if (ms == null) return '—'
  if (ms < 10) return ms.toFixed(2) + ' ms'
  if (ms < 1000) return Math.round(ms) + ' ms'
  return (ms / 1000).toFixed(2) + ' s'
}
</script>

<template>
  <v-app-bar color="surface" flat aria-label="Benchmark solver app bar">
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/benchmarks')" />
    <div
      class="d-flex align-center logo-link"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="28" />
    </div>
    <v-app-bar-title class="text-medium-emphasis ml-2">
      Solver · <span class="font-weight-medium">{{ solverName }}</span>
    </v-app-bar-title>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-skeleton-loader v-if="benchmarks.loading && !detail" type="article" />
      <v-alert v-else-if="benchmarks.error" type="error" variant="tonal">
        {{ benchmarks.error }}
      </v-alert>

      <template v-else-if="detail">
        <v-card class="pa-5 mb-4">
          <div class="d-flex align-center mb-2">
            <v-icon icon="mdi-cog-outline" start />
            <span class="text-h6">{{ detail.solver.name }}</span>
            <v-chip v-if="detail.solver.version" size="small" class="ml-2">
              v{{ detail.solver.version }}
            </v-chip>
            <v-chip
              v-if="!detail.is_currently_registered"
              size="small"
              color="warning"
              variant="tonal"
              class="ml-2"
            >
              not currently registered
            </v-chip>
            <v-spacer />
            <span class="text-body-2 text-medium-emphasis">{{ totalRecords() }} records</span>
          </div>
          <div class="text-body-2 text-medium-emphasis">
            <span v-if="detail.solver.source">Source: <code>{{ detail.solver.source }}</code></span>
            <span v-if="detail.solver.hardware" class="ml-3">
              Hardware: <code>{{ detail.solver.hardware }}</code>
            </span>
          </div>
        </v-card>

        <div
          v-for="(records, suiteId) in detail.records_by_suite"
          :key="suiteId"
          class="mb-4"
        >
          <div class="text-subtitle-1 mb-1 d-flex align-center">
            <a
              class="suite-link mr-2"
              @click="router.push(`/benchmarks/suites/${suiteId}`)"
            >
              {{ suiteId }}
            </a>
            <span class="text-medium-emphasis text-body-2">
              {{ records.length }} run<span v-if="records.length !== 1">s</span>
            </span>
          </div>
          <v-table density="compact">
            <thead>
              <tr>
                <th class="text-left">Instance</th>
                <th class="text-left">Started</th>
                <th class="text-right">Best</th>
                <th class="text-right">Time</th>
                <th class="text-left">Repro</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in records" :key="r.record_id">
                <td>
                  <a class="suite-link" @click="router.push(`/benchmarks/instances/${r.instance_id}`)">
                    {{ r.instance_id.split('/').pop() }}
                  </a>
                </td>
                <td>
                  <span class="text-caption text-medium-emphasis">
                    {{ new Date(r.started_at).toLocaleString() }}
                  </span>
                </td>
                <td class="text-right">
                  <strong>{{ r.best_user_energy ?? '—' }}</strong>
                  <v-icon
                    v-if="r.converged_to_expected === true"
                    icon="mdi-check-circle"
                    color="success"
                    size="x-small"
                    class="ml-1"
                  />
                  <v-icon
                    v-else-if="r.converged_to_expected === false"
                    icon="mdi-alert-circle"
                    color="warning"
                    size="x-small"
                    class="ml-1"
                  />
                </td>
                <td class="text-right">{{ fmtTime(r.elapsed_ms) }}</td>
                <td>
                  <ReproducibilityBadge
                    :repro-hash="r.record_id.slice(-6)"
                    size="x-small"
                  />
                </td>
                <td>
                  <CitationButton :record-id="r.record_id" size="x-small" variant="text" />
                </td>
              </tr>
            </tbody>
          </v-table>
        </div>

        <v-card v-if="!Object.keys(detail.records_by_suite).length" class="pa-6 text-center" variant="tonal">
          <v-icon icon="mdi-magnify-close" size="large" class="mb-2" />
          <div>No archived runs for this solver yet.</div>
        </v-card>
      </template>
    </v-container>
  </v-main>
</template>

<style scoped>
.suite-link {
  cursor: pointer;
  border-bottom: 1px dotted currentColor;
}
.suite-link:hover {
  color: rgb(var(--v-theme-primary));
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
