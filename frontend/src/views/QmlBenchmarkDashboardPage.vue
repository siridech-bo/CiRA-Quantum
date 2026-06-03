<script setup lang="ts">
/**
 * QmlBenchmarkDashboardPage — public scoreboard of archived QML runs.
 *
 * Mirrors the optimization-side benchmark dashboard but with QML's
 * specifics: leaderboard sorted by test accuracy on a given dataset,
 * filter by dataset / model, columns for hardware tier (was it a real
 * QPU run? what backends?).
 */
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
const api = axios.create({ withCredentials: true })

interface RecordSummary {
  record_id: string
  model: string
  dataset_id: string
  repro_hash: string
  started_at: string
  completed_at: string
  contributor_display_name: string
  hardware_id: string
  final_test_accuracy: number | null
  final_train_accuracy: number | null
  final_loss: number | null
  train_time_ms: number | null
  n_qubits: number | null
  n_baselines: number
  n_qpu_runs: number
  has_real_qpu_run: boolean
}

interface Facets {
  datasets: Record<string, number>
  models: Record<string, number>
}

const loading = ref(true)
const error = ref<string | null>(null)
const records = ref<RecordSummary[]>([])
const facets = ref<Facets>({ datasets: {}, models: {} })
const datasetFilter = ref<string | null>(null)
const modelFilter = ref<string | null>(null)

const datasetOptions = computed(() => {
  return Object.entries(facets.value.datasets)
    .map(([id, count]) => ({ title: `${id} (${count})`, value: id }))
})

const filteredSorted = computed(() => {
  // The backend already filters + sorts newest-first; here we re-sort
  // by test accuracy descending so the dashboard reads as a leaderboard.
  return [...records.value].sort((a, b) => {
    const aa = a.final_test_accuracy ?? -1
    const bb = b.final_test_accuracy ?? -1
    return bb - aa
  })
})

async function load() {
  loading.value = true
  error.value = null
  try {
    const params: Record<string, string> = {}
    if (datasetFilter.value) params.dataset_id = datasetFilter.value
    if (modelFilter.value) params.model = modelFilter.value
    const r = await api.get('/api/qml/benchmarks', { params })
    records.value = r.data.records
    facets.value = r.data.facets
  } catch (e: any) {
    error.value =
      e?.response?.data?.error || e?.message || 'Failed to load benchmarks'
  } finally {
    loading.value = false
  }
}

onMounted(load)

function fmtAcc(p: number | null) {
  if (p == null) return '—'
  return (p * 100).toFixed(1) + '%'
}
function fmtMs(ms: number | null) {
  if (ms == null) return '—'
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(1)} s`
}
function fmtDate(iso: string) {
  try { return new Date(iso).toLocaleString() } catch { return iso }
}
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QML benchmarks">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QML benchmarks</span>
    </div>
    <v-spacer />
    <v-btn variant="text" prepend-icon="mdi-school" @click="router.push('/qml/learn')">
      Primer
    </v-btn>
    <v-btn variant="text" @click="router.push('/qml')">Datasets</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <!-- Hero -->
      <div class="pt-6 pb-3">
        <div class="text-overline text-medium-emphasis">QML benchmark archive</div>
        <div class="text-h4 font-weight-bold mb-2">
          Public scoreboard of archived training runs
        </div>
        <div class="text-body-1 text-medium-emphasis" style="max-width: 760px">
          Every record is a snapshot of a completed VQC training run —
          including the four classical baselines on the same split and
          any real-QPU evaluations. Append-only and citable by BibTeX.
        </div>
      </div>

      <!-- Filters -->
      <v-row dense class="mb-3">
        <v-col cols="12" sm="6" md="4">
          <v-select
            v-model="datasetFilter"
            :items="datasetOptions"
            label="Filter by dataset"
            clearable
            density="comfortable"
            hide-details="auto"
            @update:model-value="load"
          />
        </v-col>
        <v-col cols="12" sm="6" md="4">
          <v-select
            v-model="modelFilter"
            :items="Object.entries(facets.models).map(([id, c]) => ({ title: `${id} (${c})`, value: id }))"
            label="Filter by model"
            clearable
            density="comfortable"
            hide-details="auto"
            @update:model-value="load"
          />
        </v-col>
        <v-col cols="12" md="4" class="d-flex align-center">
          <v-chip size="small" variant="tonal" prepend-icon="mdi-database-outline">
            {{ records.length }} records
          </v-chip>
          <v-spacer />
          <v-btn
            variant="text"
            prepend-icon="mdi-refresh"
            :loading="loading"
            @click="load"
          >Refresh</v-btn>
        </v-col>
      </v-row>

      <!-- States -->
      <v-progress-circular
        v-if="loading"
        indeterminate
        class="d-block mx-auto my-6"
      />
      <v-alert v-else-if="error" type="error" variant="tonal">{{ error }}</v-alert>
      <v-alert
        v-else-if="!records.length"
        type="info"
        variant="tonal"
        icon="mdi-database-outline"
      >
        The archive is empty.
        Admins can archive completed training runs via the
        <strong>Archive to benchmarks</strong> button on a job detail
        page. The first reference run is up to you.
      </v-alert>

      <!-- Leaderboard table -->
      <v-card v-else variant="outlined" class="pa-0">
        <v-table density="comfortable" hover>
          <thead>
            <tr>
              <th class="text-left">#</th>
              <th class="text-left">Dataset</th>
              <th class="text-left">Model</th>
              <th class="text-right">Test acc</th>
              <th class="text-right">Train acc</th>
              <th class="text-right">Qubits</th>
              <th class="text-right">Wall</th>
              <th class="text-left">Contributor</th>
              <th class="text-left">QPU</th>
              <th class="text-left">When</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(r, i) in filteredSorted"
              :key="r.record_id"
              class="row-link"
              @click="router.push(`/qml/benchmarks/${r.record_id}`)"
            >
              <td class="text-medium-emphasis">{{ i + 1 }}</td>
              <td><code>{{ r.dataset_id }}</code></td>
              <td>
                <v-chip
                  size="x-small"
                  :color="r.model === 'vqc' ? 'accent' : 'primary'"
                  variant="flat"
                >{{ r.model }}</v-chip>
              </td>
              <td class="text-right">
                <strong>{{ fmtAcc(r.final_test_accuracy) }}</strong>
              </td>
              <td class="text-right text-medium-emphasis">
                {{ fmtAcc(r.final_train_accuracy) }}
              </td>
              <td class="text-right">{{ r.n_qubits ?? '—' }}</td>
              <td class="text-right text-medium-emphasis">
                {{ fmtMs(r.train_time_ms) }}
              </td>
              <td>{{ r.contributor_display_name }}</td>
              <td>
                <v-chip
                  v-if="r.has_real_qpu_run"
                  size="x-small"
                  color="error"
                  variant="flat"
                  prepend-icon="mdi-atom"
                >real QPU</v-chip>
                <v-chip
                  v-else-if="r.n_qpu_runs > 0"
                  size="x-small"
                  color="warning"
                  variant="tonal"
                >cloud sim</v-chip>
                <span v-else class="text-medium-emphasis text-caption">—</span>
              </td>
              <td class="text-caption text-medium-emphasis">
                {{ fmtDate(r.completed_at) }}
              </td>
            </tr>
          </tbody>
        </v-table>
      </v-card>

      <div class="text-caption text-medium-emphasis mt-3">
        Ordered by test accuracy descending. Click a row for the full
        record (training history, baselines, QPU runs, BibTeX cite).
      </div>
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
.row-link {
  cursor: pointer;
}
.row-link:hover td {
  background: rgba(168, 85, 247, 0.06);
}
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
</style>
