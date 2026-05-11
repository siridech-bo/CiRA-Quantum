<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useBenchmarksStore } from '@/stores/benchmarks'
import CitationButton from '@/components/CitationButton.vue'

const route = useRoute()
const router = useRouter()
const benchmarks = useBenchmarksStore()

const instanceId = computed(() => String((route.params as any).pathMatch || ''))

async function load() {
  if (!instanceId.value) return
  await benchmarks.loadInstance(instanceId.value)
}

onMounted(load)
watch(instanceId, load)

const detail = computed(() => benchmarks.instanceDetail[instanceId.value] || null)

function fmtTime(ms: number | null | undefined): string {
  if (ms == null) return '—'
  if (ms < 10) return ms.toFixed(2) + ' ms'
  if (ms < 1000) return Math.round(ms) + ' ms'
  return (ms / 1000).toFixed(2) + ' s'
}
</script>

<template>
  <v-app-bar color="surface" flat>
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/benchmarks')" />
    <v-app-bar-title class="text-primary font-weight-bold">
      Instance · {{ instanceId.split('/').pop() }}
    </v-app-bar-title>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-skeleton-loader v-if="benchmarks.loading && !detail" type="table" />
      <v-alert v-else-if="benchmarks.error" type="error" variant="tonal">
        {{ benchmarks.error }}
      </v-alert>

      <template v-else-if="detail">
        <v-card class="pa-4 mb-4" variant="tonal">
          <div class="text-body-2 text-medium-emphasis mb-1">Instance ID</div>
          <code>{{ detail.instance_id }}</code>
        </v-card>

        <div class="text-h6 mb-2">Leaderboard</div>
        <v-card v-if="!detail.leaderboard.length" class="pa-6 text-center" variant="tonal">
          <v-icon icon="mdi-magnify-close" size="large" class="mb-2" />
          <div>No solver has run on this instance yet.</div>
        </v-card>
        <v-table v-else density="comfortable">
          <thead>
            <tr>
              <th class="text-left">#</th>
              <th class="text-left">Solver</th>
              <th class="text-right">Best energy</th>
              <th class="text-right">Gap to expected</th>
              <th class="text-right">Time</th>
              <th class="text-left">Hardware</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(r, i) in detail.leaderboard" :key="r.record_id">
              <td class="text-medium-emphasis">{{ i + 1 }}</td>
              <td>
                <a class="solver-link" @click="router.push(`/benchmarks/solvers/${r.solver_name}`)">
                  {{ r.solver_name }}
                </a>
                <span class="text-caption text-medium-emphasis ml-1">
                  v{{ r.solver_version }}
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
              <td class="text-right">
                <span v-if="r.gap_to_expected !== null">{{ r.gap_to_expected }}</span>
                <span v-else class="text-medium-emphasis">—</span>
              </td>
              <td class="text-right">{{ fmtTime(r.elapsed_ms) }}</td>
              <td>
                <span class="text-caption text-medium-emphasis">
                  {{ r.hardware_id?.split('|')[0] || 'unknown' }}
                </span>
              </td>
              <td>
                <CitationButton :record-id="r.record_id" size="x-small" variant="text" />
              </td>
            </tr>
          </tbody>
        </v-table>
      </template>
    </v-container>
  </v-main>
</template>

<style scoped>
.solver-link {
  cursor: pointer;
  border-bottom: 1px dotted currentColor;
}
.solver-link:hover {
  color: rgb(var(--v-theme-primary));
}
</style>
