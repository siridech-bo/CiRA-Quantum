<script setup lang="ts">
/**
 * QrcRunDetailPage — §0.D core view: FID time-domain + spectrum plots
 * with a step slider, a live progress trace (poll every ~3s while
 * running), and the results panel. Mirrors QmlJobDetailPage's layout
 * (app bar, header chip, cards) adapted to QRC's polling model instead
 * of QML's SSE stream — §3 only specifies plain GET polling endpoints
 * for QRC, no stream.
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useQrcStore, type QrcRunStatus } from '@/stores/qrc'
import CiraLogo from '@/components/CiraLogo.vue'
import QrcFidTimeChart from '@/components/QrcFidTimeChart.vue'
import QrcSpectrumChart from '@/components/QrcSpectrumChart.vue'
import QrcResultsPanel from '@/components/QrcResultsPanel.vue'
import QrcFeatureLabPanel from '@/components/QrcFeatureLabPanel.vue'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const qrc = useQrcStore()

const runId = computed(() => route.params.id as string)

const loading = ref(true)
const fatal = ref<string | null>(null)
const step = ref(0)
const fidLoading = ref(false)
const fidError = ref<string | null>(null)
const selectedTrace = ref<string>('') // which encoding's saved waveform (sweep runs)
const stopping = ref(false)
const stopError = ref<string | null>(null)
const eventLogEl = ref<HTMLDivElement | null>(null)

const STATUS_COLORS: Record<QrcRunStatus, string> = {
  running: 'primary',
  done: 'success',
  error: 'error',
  stopped: 'warning',
}

const runSummary = computed(() => qrc.runs.find((r) => r.id === runId.value) || null)

const status = computed<QrcRunStatus | null>(
  () => qrc.currentProgress?.status ?? runSummary.value?.status ?? null,
)

const isActive = computed(() => status.value === 'running')

// Feature-Lab (embedding + Exp 1.1/1.2/1.3 charts) applies to trace-backed
// runs: a phase1 sweep (charts + embedding) or a trace-gen run (embedding).
const showFeatureLab = computed(() => {
  const t = runSummary.value?.task
  return t === 'phase1' || t === 'trace-gen'
})

const progressPct = computed(() => {
  const p = qrc.currentProgress
  if (!p || !p.total) return runSummary.value?.progress_pct ?? 0
  return Math.min(100, (p.step / p.total) * 100)
})

function fmtEta(s: number | null | undefined): string {
  if (s == null) return '—'
  if (s <= 0) return 'done'
  const m = Math.floor(s / 60)
  const sec = Math.round(s % 60)
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`
}

let debounceTimer: ReturnType<typeof setTimeout> | null = null

async function fetchFid(k: number) {
  fidLoading.value = true
  fidError.value = null
  try {
    await qrc.loadFid(runId.value, k, selectedTrace.value || undefined)
    // Default the picker to whatever trace the backend served.
    if (!selectedTrace.value && qrc.currentFid?.trace_name) {
      const t = qrc.currentFid.available_traces?.find((x) => x.name === qrc.currentFid?.trace_name)
      if (t) selectedTrace.value = t.name
    }
  } catch (e: any) {
    fidError.value = e?.response?.data?.error || e?.message || 'Failed to load FID'
  } finally {
    fidLoading.value = false
  }
}

function onTraceChange(name: string) {
  selectedTrace.value = name
  followLive.value = false // viewing a specific saved encoding, not the live stream
  fetchFid(step.value)
}

function onStepInput(v: number | number[]) {
  const k = Array.isArray(v) ? v[0] : v
  step.value = k
  followLive.value = false // user is scrubbing manually — stop auto-following
  if (debounceTimer) clearTimeout(debounceTimer)
  debounceTimer = setTimeout(() => fetchFid(k), 150)
}

async function loadProgressOnce() {
  try {
    await qrc.loadProgress(runId.value)
  } catch (e: any) {
    fatal.value = e?.response?.data?.error || e?.message || 'Failed to load run progress'
  }
}

// While the run is active, follow the live FID: re-fetch the latest computed
// step every few seconds so the FID + spectrum stream on screen (no refresh).
let liveFidTimer: ReturnType<typeof setInterval> | null = null
const followLive = ref(true)

async function pollLiveFid() {
  if (!followLive.value) return
  // Request "latest": the backend clamps an over-range step to the newest
  // computed one, so this follows the current encoding regardless of transitions.
  await fetchFid(1_000_000).catch(() => {}) // 404 while nothing computed yet is fine
  const k = qrc.currentFid?.step
  if (typeof k === 'number') step.value = k
}

function startLiveFid() {
  stopLiveFid()
  liveFidTimer = setInterval(() => {
    pollLiveFid()
    if (status.value && status.value !== 'running') stopLiveFid()
  }, 3000)
}
function stopLiveFid() {
  if (liveFidTimer) {
    clearInterval(liveFidTimer)
    liveFidTimer = null
  }
}

async function loadResultsIfDone() {
  if (status.value !== 'done') return
  try {
    await qrc.loadResults(runId.value)
  } catch {
    // Non-fatal — the results panel just shows its own empty state.
  }
}

watch(
  () => qrc.currentProgress?.events.length,
  async () => {
    await nextTick()
    if (eventLogEl.value) {
      eventLogEl.value.scrollTop = eventLogEl.value.scrollHeight
    }
  },
)

watch(status, (s, prev) => {
  if (s === 'done' && prev !== 'done') {
    loadResultsIfDone()
  }
  if (s === 'running') {
    startLiveFid()
  }
  if (s && s !== 'running') {
    qrc.stopProgressPolling()
    stopLiveFid()
    fetchFid(step.value).catch(() => {}) // final: pick up the saved .npz if any
  }
})

async function stopRun() {
  stopping.value = true
  stopError.value = null
  try {
    await qrc.stopRun(runId.value)
    await loadProgressOnce()
  } catch (e: any) {
    stopError.value = e?.message || 'Failed to stop run'
  } finally {
    stopping.value = false
  }
}

onMounted(async () => {
  loading.value = true
  qrc.reset()
  followLive.value = true // page load = follow the live FID again
  try {
    await Promise.all([
      qrc.loadRuns().catch(() => {}),
      loadProgressOnce(),
      fetchFid(0),
    ])
    await loadResultsIfDone()
    if (isActive.value) {
      qrc.startProgressPolling(runId.value, 3000)
      startLiveFid()
    }
  } finally {
    loading.value = false
  }
})

onBeforeUnmount(() => {
  qrc.stopProgressPolling()
  stopLiveFid()
  if (debounceTimer) clearTimeout(debounceTimer)
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QRC run detail">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QRC run</span>
    </div>
    <v-spacer />
    <v-btn variant="text" @click="router.push('/qrc')">All runs</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />
      <v-alert v-else-if="fatal" type="error" variant="tonal">{{ fatal }}</v-alert>

      <template v-else>
        <!-- Header -->
        <div class="d-flex align-center flex-wrap ga-3 mb-2">
          <v-btn
            icon="mdi-arrow-left"
            variant="text"
            @click="router.push('/qrc')"
            aria-label="Back to QRC runs"
          />
          <div class="flex-grow-1">
            <div class="text-overline text-medium-emphasis">
              {{ runSummary?.task || 'QRC run' }}
            </div>
            <div class="text-h5">Run {{ runId.slice(0, 12) }}</div>
          </div>
          <v-chip v-if="status" :color="STATUS_COLORS[status]" variant="flat">
            {{ status }}
          </v-chip>
          <v-btn
            v-if="auth.user && isActive"
            size="small"
            variant="tonal"
            color="error"
            prepend-icon="mdi-stop"
            :loading="stopping"
            @click="stopRun"
          >Stop</v-btn>
        </div>

        <v-alert
          v-if="qrc.usingMockProgress || qrc.usingMockFid || qrc.usingMockResults"
          type="warning"
          variant="tonal"
          density="compact"
          class="mb-4"
          icon="mdi-cloud-off-outline"
        >
          The QRC backend isn't reachable right now — this page is
          showing demo data so the layout can still be exercised.
        </v-alert>

        <v-alert v-if="stopError" type="error" variant="tonal" density="compact" class="mb-4">
          {{ stopError }}
        </v-alert>

        <!-- Live progress trace -->
        <v-card v-if="qrc.currentProgress" class="pa-4 mb-4">
          <div class="d-flex align-center mb-2 flex-wrap ga-2">
            <v-progress-circular
              v-if="isActive"
              indeterminate
              size="20"
              width="2"
              class="mr-1"
            />
            <span class="text-subtitle-1">
              Phase: {{ qrc.currentProgress.phase || '—' }}
            </span>
            <v-spacer />
            <span class="text-caption text-medium-emphasis">
              step {{ qrc.currentProgress.step }} / {{ qrc.currentProgress.total || '?' }}
            </span>
            <v-chip size="x-small" variant="tonal">
              ETA {{ fmtEta(qrc.currentProgress.eta_s) }}
            </v-chip>
          </div>
          <v-progress-linear
            :model-value="progressPct"
            :indeterminate="isActive && !qrc.currentProgress.total"
            :color="status ? STATUS_COLORS[status] : 'primary'"
            height="8"
            rounded
            class="mb-3"
          />
          <div class="text-caption text-medium-emphasis mb-1">Event log</div>
          <div ref="eventLogEl" class="event-log">
            <div v-if="!qrc.currentProgress.events.length" class="text-body-2 text-medium-emphasis">
              No events yet.
            </div>
            <div
              v-for="(ev, i) in qrc.currentProgress.events"
              :key="i"
              class="event-row"
            >
              <span class="text-caption text-medium-emphasis">
                +{{ ev.elapsed_s.toFixed(0) }}s
              </span>
              <v-chip size="x-small" variant="tonal" class="mx-2">{{ ev.kind }}</v-chip>
              <span class="text-body-2">{{ ev.message }}</span>
            </div>
          </div>
        </v-card>

        <!-- Trace picker: sweep runs (phase2/memcap) save one waveform per encoding -->
        <v-card
          v-if="(qrc.currentFid?.available_traces?.length || 0) > 1"
          class="pa-3 mb-3 d-flex align-center flex-wrap ga-3"
        >
          <v-icon icon="mdi-database-search-outline" />
          <div class="text-body-2 text-medium-emphasis">Saved waveform — pick encoding:</div>
          <v-select
            :model-value="selectedTrace"
            :items="qrc.currentFid?.available_traces || []"
            item-title="fn"
            item-value="name"
            density="compact"
            hide-details
            variant="outlined"
            style="max-width: 280px"
            @update:model-value="onTraceChange"
          />
        </v-card>

        <!-- FID + spectrum -->
        <v-row>
          <v-col cols="12" md="6">
            <v-card class="pa-4 h-100">
              <div class="d-flex align-center mb-2">
                <div class="text-subtitle-1 flex-grow-1">FID — time domain</div>
                <v-progress-circular v-if="fidLoading" indeterminate size="16" width="2" />
              </div>
              <QrcFidTimeChart :fid="qrc.currentFid" />
            </v-card>
          </v-col>
          <v-col cols="12" md="6">
            <v-card class="pa-4 h-100">
              <div class="d-flex align-center mb-2">
                <div class="text-subtitle-1 flex-grow-1">Spectrum — frequency domain</div>
                <v-progress-circular v-if="fidLoading" indeterminate size="16" width="2" />
              </div>
              <QrcSpectrumChart :fid="qrc.currentFid" />
            </v-card>
          </v-col>
        </v-row>

        <v-alert v-if="fidError" type="error" variant="tonal" density="compact" class="my-3">
          {{ fidError }}
        </v-alert>

        <!-- Step slider -->
        <v-card class="pa-4 my-4">
          <div class="d-flex align-center mb-1">
            <div class="text-subtitle-2 flex-grow-1">Input step</div>
            <v-chip
              v-if="isActive"
              size="x-small"
              variant="flat"
              :color="followLive ? 'success' : 'warning'"
              class="mr-2"
              :prepend-icon="followLive ? 'mdi-record' : 'mdi-pause'"
              style="cursor: pointer"
              @click="followLive = true"
            >{{ followLive ? 'LIVE — following' : 'paused · resume' }}</v-chip>
            <span class="text-caption text-medium-emphasis">
              step {{ step }} / {{ (qrc.currentFid?.n_steps || 1) - 1 }}
            </span>
          </div>
          <v-slider
            :model-value="step"
            :min="0"
            :max="Math.max(0, (qrc.currentFid?.n_steps || 1) - 1)"
            :step="1"
            hide-details
            thumb-label
            @update:model-value="onStepInput"
          />
        </v-card>

        <!-- Feature Lab (embedding + Exp 1.1/1.2/1.3) -->
        <QrcFeatureLabPanel
          v-if="showFeatureLab"
          :run-id="runId"
          :results="qrc.currentResults"
          :task="runSummary?.task"
        />

        <!-- Results -->
        <QrcResultsPanel :results="qrc.currentResults" />
      </template>
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
.event-log {
  max-height: 180px;
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 6px;
  padding: 8px 10px;
}
.event-row {
  padding: 2px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.event-row:last-child {
  border-bottom: none;
}
</style>
