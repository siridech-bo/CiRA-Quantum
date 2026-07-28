<script setup lang="ts">
/**
 * QrcDashboardPage — §0.D dashboard: list of runs with live status, the
 * "New run" control, and per-run Stop buttons. Mirrors QmlLandingPage's
 * layout (app bar, hero, gated action buttons) with JobHistory's
 * v-table convention for the run list.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useQrcStore, type QrcRunStatus, type QrcRunSummary } from '@/stores/qrc'
import CiraLogo from '@/components/CiraLogo.vue'
import QrcNewRunDialog from '@/components/QrcNewRunDialog.vue'

const router = useRouter()
const auth = useAuthStore()
const qrc = useQrcStore()

const loading = ref(true)
const fatal = ref<string | null>(null)
const newRunDialog = ref(false)
const stoppingId = ref<string | null>(null)
const stopError = ref<string | null>(null)

const STATUS_COLORS: Record<QrcRunStatus, string> = {
  running: 'primary',
  done: 'success',
  error: 'error',
  stopped: 'warning',
}

function isActive(s: QrcRunStatus): boolean {
  return s === 'running'
}

function fmtEta(s: number | null): string {
  if (s == null) return '—'
  if (s <= 0) return 'done'
  const m = Math.floor(s / 60)
  const sec = Math.round(s % 60)
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`
}

function fmtDate(s: string): string {
  return new Date(s).toLocaleString()
}

async function refresh() {
  try {
    await qrc.loadRuns()
    fatal.value = null
  } catch (e: any) {
    fatal.value = e?.response?.data?.error || e?.message || 'Failed to load QRC runs'
  }
}

async function stopRun(run: QrcRunSummary) {
  stoppingId.value = run.id
  stopError.value = null
  try {
    await qrc.stopRun(run.id)
  } catch (e: any) {
    stopError.value = e?.message || 'Failed to stop run'
  } finally {
    stoppingId.value = null
  }
}

function onCreated(id: string) {
  router.push(`/qrc/runs/${id}`)
}

const sortedRuns = computed(() =>
  [...qrc.runs].sort(
    (a, b) => new Date(b.created_utc).getTime() - new Date(a.created_utc).getTime(),
  ),
)

onMounted(async () => {
  loading.value = true
  await refresh()
  loading.value = false
  qrc.startRunsPolling(5000)
})

onBeforeUnmount(() => {
  qrc.stopRunsPolling()
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QRC app bar">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QRC</span>
    </div>
    <v-spacer />
    <v-btn variant="text" @click="router.push('/')">Home</v-btn>
    <v-btn v-if="!auth.user" variant="outlined" class="ml-2" @click="router.push('/login')">
      Log in
    </v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <!-- Hero -->
      <div class="pt-8 pb-4">
        <div class="text-overline text-medium-emphasis">
          Quantum Reservoir Computing — remote control plane
        </div>
        <div class="text-h4 font-weight-bold mb-2">QRC runs</div>
        <div class="text-body-1 text-medium-emphasis" style="max-width: 720px">
          Launch, monitor, and stop GPU reservoir runs on the dev box from
          the browser. Each run's FID trace, spectrum, live progress, and
          NARMA/weather results are on its detail page.
        </div>
      </div>

      <v-alert
        v-if="!auth.user"
        type="info"
        variant="tonal"
        density="compact"
        class="mb-4"
        icon="mdi-lock-outline"
      >
        Log in to launch or stop runs. The run list and every detail page
        below are viewable without an account.
      </v-alert>

      <v-alert
        v-if="qrc.usingMockRuns"
        type="warning"
        variant="tonal"
        density="compact"
        class="mb-4"
        icon="mdi-cloud-off-outline"
      >
        The QRC backend isn't reachable right now — showing demo data so
        the UI can still be exercised. Live runs will appear here once
        <code>/api/qrc</code> answers.
      </v-alert>

      <v-alert v-if="stopError" type="error" variant="tonal" density="compact" class="mb-4">
        {{ stopError }}
      </v-alert>

      <div class="d-flex align-center mb-2">
        <div class="text-h6 flex-grow-1">Runs</div>
        <v-btn
          size="small"
          variant="tonal"
          prepend-icon="mdi-refresh"
          class="mr-2"
          @click="refresh"
        >Refresh</v-btn>
        <v-btn
          v-if="auth.user"
          color="primary"
          variant="flat"
          prepend-icon="mdi-play-circle"
          :disabled="qrc.anyRunActive"
          @click="newRunDialog = true"
        >New run</v-btn>
      </div>
      <div v-if="auth.user && qrc.anyRunActive" class="text-caption text-medium-emphasis mb-2">
        A run is already active — only one heavy job may run at a time.
      </div>

      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />
      <v-alert v-else-if="fatal" type="error" variant="tonal">{{ fatal }}</v-alert>

      <v-card v-else class="pa-2">
        <v-table density="comfortable">
          <thead>
            <tr>
              <th>ID</th>
              <th>Task</th>
              <th>Status</th>
              <th>Progress</th>
              <th>ETA</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-if="!sortedRuns.length">
              <td colspan="7" class="text-center text-medium-emphasis py-6">
                No QRC runs yet.
                <span v-if="auth.user">Launch one with "New run" above.</span>
              </td>
            </tr>
            <tr
              v-for="run in sortedRuns"
              :key="run.id"
              class="run-row"
              @click="router.push(`/qrc/runs/${run.id}`)"
            >
              <td><code class="text-caption">{{ run.id.slice(0, 12) }}</code></td>
              <td>{{ run.task }}</td>
              <td>
                <v-chip size="x-small" :color="STATUS_COLORS[run.status]" variant="flat">
                  {{ run.status }}
                </v-chip>
              </td>
              <td style="min-width: 140px">
                <div class="d-flex align-center ga-2">
                  <v-progress-linear
                    :model-value="run.progress_pct"
                    height="6"
                    rounded
                    :color="STATUS_COLORS[run.status]"
                    style="max-width: 100px"
                  />
                  <span class="text-caption">{{ run.progress_pct.toFixed(0) }}%</span>
                </div>
              </td>
              <td>{{ fmtEta(run.eta_s) }}</td>
              <td class="text-caption">{{ fmtDate(run.created_utc) }}</td>
              <td>
                <v-btn
                  v-if="auth.user && isActive(run.status)"
                  size="x-small"
                  variant="tonal"
                  color="error"
                  prepend-icon="mdi-stop"
                  :loading="stoppingId === run.id"
                  @click.stop="stopRun(run)"
                >Stop</v-btn>
              </td>
            </tr>
          </tbody>
        </v-table>
      </v-card>
    </v-container>

    <QrcNewRunDialog v-model="newRunDialog" @created="onCreated" />
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
.run-row {
  cursor: pointer;
}
.run-row:hover {
  background: rgba(255, 255, 255, 0.03);
}
</style>
