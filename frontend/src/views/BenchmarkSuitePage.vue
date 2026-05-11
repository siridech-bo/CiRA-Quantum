<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useBenchmarksStore } from '@/stores/benchmarks'
import SolverComparisonTable from '@/components/SolverComparisonTable.vue'

const route = useRoute()
const router = useRouter()
const benchmarks = useBenchmarksStore()

// The suite_id can contain slashes (e.g. `knapsack/small`), so we
// reassemble it from the splat-style path param the router passes.
const suiteId = computed(() => String((route.params as any).pathMatch || ''))

async function load() {
  if (!suiteId.value) return
  await benchmarks.loadSuite(suiteId.value)
}

onMounted(load)
watch(suiteId, load)

const detail = computed(() => benchmarks.suiteDetail[suiteId.value] || null)

function openInstance(instanceId: string) {
  router.push(`/benchmarks/instances/${instanceId}`)
}

function openSolver(solverName: string) {
  router.push(`/benchmarks/solvers/${solverName}`)
}
</script>

<template>
  <v-app-bar color="surface" flat>
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/benchmarks')" />
    <v-app-bar-title class="text-primary font-weight-bold">
      Benchmarks · {{ suiteId }}
    </v-app-bar-title>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-skeleton-loader v-if="benchmarks.loading && !detail" type="table" />
      <v-alert v-else-if="benchmarks.error" type="error" variant="tonal">
        {{ benchmarks.error }}
      </v-alert>
      <SolverComparisonTable
        v-else-if="detail"
        :suite-id="detail.suite_id"
        :instances="detail.instances"
        :records="detail.records"
        @open-instance="openInstance"
        @open-solver="openSolver"
      />
    </v-container>
  </v-main>
</template>
