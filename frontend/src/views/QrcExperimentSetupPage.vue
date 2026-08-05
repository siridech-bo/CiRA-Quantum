<script setup lang="ts">
/**
 * QrcExperimentSetupPage — /qrc/new. The full experiment-setup form: every
 * parameter visible (user-set + program/physics defaults), a dataset dropdown,
 * a readout panel that surfaces the physics-informed D (SOP §0/§3.2), a
 * compare-against selector (benchmarks/tiers), and Run/Stop. Launches through
 * the launcher API so the run is visible/stoppable in the dashboard; GPU tasks
 * require an explicit confirm first (CLAUDE.md GPU hard rule).
 *
 * The schema is served by GET /api/qrc/schema, so the form can never submit a
 * config the launcher would reject — the two cannot drift.
 */
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useQrcStore, type QrcField, type QrcTaskSchema } from '@/stores/qrc'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
const auth = useAuthStore()
const qrc = useQrcStore()

const loading = ref(true)
const fatal = ref<string | null>(null)
const launching = ref(false)
const launchError = ref<string | null>(null)
const gpuConfirm = ref(false)

const taskKey = ref<string>('memory')
const values = reactive<Record<string, any>>({})
const selectedRefs = ref<string[]>([])

const tasks = computed<QrcTaskSchema[]>(() => qrc.schema?.tasks ?? [])
const task = computed<QrcTaskSchema | undefined>(() => tasks.value.find((t) => t.key === taskKey.value))
const groups = computed(() => qrc.schema?.groups ?? [])

/** Fields of the current task grouped by their group key (only non-empty groups). */
const fieldsByGroup = computed<{ key: string; label: string; fields: QrcField[] }[]>(() => {
  const t = task.value
  if (!t) return []
  return groups.value
    .map((g) => ({ key: g.key, label: g.label, fields: t.fields.filter((f) => f.group === g.key) }))
    .filter((g) => g.fields.length > 0)
})

const datasetField = computed<QrcField | undefined>(() => {
  const t = task.value
  if (!t?.dataset_field) return undefined
  return t.fields.find((f) => f.name === t.dataset_field)
})

/** Readout / feature-count (D) panel — reacts to readout + system/n_spins. */
const readoutInfo = computed(() => {
  const cat = qrc.schema?.readouts
  if (!cat) return null
  const ro = values['readout']
  const sysKey = String(values['system_learn'] ?? values['n_spins'] ?? '')
  if (ro === 'fid_reduced' || taskKey.value === 'learnable') {
    const per = cat.fid_reduced?.per_system?.[sysKey]
    if (per) {
      return {
        label: 'Physics-informed reduced FID (D_eff)',
        D: per.D_features,
        detail: `D_eff=${per.D_eff} lines (from ${per.n_raw} raw transitions, merged at `
          + `${per.linewidth_hz} Hz) -> ${per.D_features} features. Suggest M~${per.suggest_M} FID `
          + `samples, dwell~${per.suggest_dwell_ms} ms. SOP 3.2.`,
      }
    }
    return { label: 'Physics-informed reduced FID', D: null,
             detail: 'D_eff computed at runtime for this system (precomputed for 3/6/crotonic9).' }
  }
  if (ro === 'observable') {
    return { label: 'Observable <sigma>xV', D: null,
             detail: 'D = 3 * n_spins * V (varies across the V sweep). Label D on results. SOP 0.' }
  }
  return { label: cat.fid653?.label, D: cat.fid653?.D, detail: cat.fid653?.note }
})

/** Deep-copy a schema default, unwrapping Vue reactive proxies (the schema is
 *  stored in a reactive ref, so nested arrays are Proxies — structuredClone
 *  can't clone those). All defaults are JSON-safe (primitives / number arrays). */
function cloneDefault(v: any): any {
  if (Array.isArray(v)) return v.map((x) => x)
  if (v && typeof v === 'object') return JSON.parse(JSON.stringify(v))
  return v
}

function initValues() {
  const t = task.value
  Object.keys(values).forEach((k) => delete values[k])
  if (!t) return
  for (const f of t.fields) values[f.name] = cloneDefault(f.default)
}

function onTaskChange() {
  launchError.value = null
  initValues()
}

/** list <-> "1, 2, 5" text bridge */
function listText(name: string): string {
  const v = values[name]
  return Array.isArray(v) ? v.join(', ') : ''
}
function setListText(name: string, text: string) {
  values[name] = text.split(/[\s,]+/).filter(Boolean).map((x) => Number(x)).filter((x) => !Number.isNaN(x))
}

/** Build the launch config from the current field values (all fields; the
 *  launcher ignores false flags and validates ranges). */
function buildConfig(): Record<string, any> {
  const t = task.value!
  const cfg: Record<string, any> = {}
  for (const f of t.fields) {
    const v = values[f.name]
    if (f.kind === 'flag') { if (v) cfg[f.name] = true }
    else if (f.kind === 'list') { if (Array.isArray(v) && v.length) cfg[f.name] = v }
    else if (v !== null && v !== undefined && v !== '') cfg[f.name] = v
  }
  return cfg
}

async function doLaunch() {
  launching.value = true
  launchError.value = null
  try {
    const { id } = await qrc.createRun(taskKey.value, buildConfig())
    // stash the compare-against selection for the run detail / compare page
    if (selectedRefs.value.length) {
      sessionStorage.setItem(`qrc-compare-${id}`, JSON.stringify(selectedRefs.value))
    }
    router.push(`/qrc/runs/${id}`)
  } catch (e: any) {
    launchError.value = e?.message || 'Failed to launch run'
  } finally {
    launching.value = false
    gpuConfirm.value = false
  }
}

function onRun() {
  launchError.value = null
  if (task.value?.gpu) { gpuConfirm.value = true; return }
  doLaunch()
}

async function onStopActive() {
  const active = qrc.runs.find((r) => r.status === 'running')
  if (active) {
    try { await qrc.stopRun(active.id) } catch { /* surfaced via qrc.error */ }
  }
}

onMounted(async () => {
  loading.value = true
  try {
    await qrc.loadSchema()
    await qrc.loadReferences().catch(() => {})
    await qrc.loadRuns().catch(() => {})
    if (!tasks.value.find((t) => t.key === taskKey.value) && tasks.value.length) {
      taskKey.value = tasks.value[0].key
    }
    initValues()
    fatal.value = null
  } catch (e: any) {
    fatal.value = e?.response?.data?.error || e?.message || 'Failed to load the experiment schema'
  } finally {
    loading.value = false
  }
})

const refItems = computed(() =>
  qrc.references.map((r) => ({
    title: `${r.label}  ·  ${r.kind}${r.D ? `  (D=${r.D})` : ''}`,
    value: r.id,
  })),
)
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QRC setup app bar">
    <div class="d-flex align-center logo-link ml-3" role="button" tabindex="0"
         @click="router.push('/qrc')" @keydown.enter="router.push('/qrc')">
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QRC / New experiment</span>
    </div>
    <v-spacer />
    <v-btn variant="text" prepend-icon="mdi-format-list-bulleted" @click="router.push('/qrc')">Runs</v-btn>
    <v-btn variant="text" prepend-icon="mdi-compare" @click="router.push('/qrc/compare')">Compare</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <div class="pt-8 pb-4">
        <div class="text-overline text-medium-emphasis">Experiment setup</div>
        <div class="text-h4 font-weight-bold mb-2">New QRC experiment</div>
        <div class="text-body-1 text-medium-emphasis" style="max-width: 760px">
          Every parameter is shown — user-settable inputs and program/physics defaults —
          so nothing is missed. Pick a dataset, set the knobs, choose what to compare
          against, then Run. Launches on the dev box and appears in the run list.
        </div>
      </div>

      <v-alert v-if="!auth.user" type="info" variant="tonal" density="compact" class="mb-4"
               icon="mdi-lock-outline">
        Log in to launch a run. You can still browse every parameter here.
      </v-alert>
      <v-alert v-if="fatal" type="error" variant="tonal" class="mb-4">{{ fatal }}</v-alert>
      <v-progress-circular v-if="loading" indeterminate class="d-block mx-auto my-8" />

      <v-row v-else-if="task">
        <!-- Left: task + parameter panel -->
        <v-col cols="12" md="8">
          <v-card variant="outlined" class="pa-4 mb-4">
            <div class="d-flex align-center flex-wrap ga-3">
              <v-select
                v-model="taskKey" :items="tasks" item-title="label" item-value="key"
                label="Experiment" variant="outlined" density="comfortable" hide-details
                style="min-width: 320px" @update:model-value="onTaskChange" />
              <v-chip v-if="task.gpu" size="small" color="warning" variant="tonal" prepend-icon="mdi-expansion-card">
                GPU
              </v-chip>
              <v-chip v-else size="small" color="success" variant="tonal" prepend-icon="mdi-cpu-64-bit">CPU</v-chip>
            </div>
            <div class="text-body-2 text-medium-emphasis mt-3">{{ task.desc }}</div>
          </v-card>

          <v-card v-for="g in fieldsByGroup" :key="g.key" variant="outlined" class="pa-4 mb-4">
            <div class="text-overline text-medium-emphasis mb-2">{{ g.label }}</div>
            <v-row dense>
              <v-col v-for="f in g.fields" :key="f.name" cols="12" sm="6">
                <!-- choice -->
                <v-select
                  v-if="f.kind === 'choice'"
                  v-model="values[f.name]" :items="f.options" :label="f.label"
                  variant="outlined" density="comfortable"
                  :hint="f.help || undefined" persistent-hint />
                <!-- flag -->
                <v-switch
                  v-else-if="f.kind === 'flag'"
                  v-model="values[f.name]" :label="f.label" color="primary"
                  density="comfortable" hide-details inset />
                <!-- list -->
                <v-text-field
                  v-else-if="f.kind === 'list'"
                  :model-value="listText(f.name)" :label="f.label + ' (comma-separated)'"
                  variant="outlined" density="comfortable"
                  :hint="f.help || undefined" persistent-hint
                  @update:model-value="(v: string) => setListText(f.name, v)" />
                <!-- number -->
                <v-text-field
                  v-else
                  v-model.number="values[f.name]" type="number" :label="f.label"
                  :min="f.min" :max="f.max"
                  :step="f.ntype === 'float' ? 'any' : 1"
                  variant="outlined" density="comfortable"
                  :hint="f.help || undefined" persistent-hint />
              </v-col>
            </v-row>
          </v-card>
        </v-col>

        <!-- Right: dataset, readout/D, compare-against, run/stop -->
        <v-col cols="12" md="4">
          <div class="sticky-side">
            <v-card v-if="datasetField" variant="outlined" class="pa-4 mb-4">
              <div class="text-overline text-medium-emphasis mb-2">Dataset</div>
              <v-select
                v-model="values[datasetField.name]" :items="datasetField.options"
                :label="datasetField.label" variant="outlined" density="comfortable"
                :hint="datasetField.help || undefined" persistent-hint />
            </v-card>

            <v-card v-if="readoutInfo" variant="outlined" class="pa-4 mb-4">
              <div class="text-overline text-medium-emphasis mb-1">Readout &amp; D</div>
              <div class="text-body-2 font-weight-medium">{{ readoutInfo.label }}</div>
              <div v-if="readoutInfo.D" class="text-h5 my-1">D = {{ readoutInfo.D }}</div>
              <div class="text-caption text-medium-emphasis">{{ readoutInfo.detail }}</div>
            </v-card>

            <v-card variant="outlined" class="pa-4 mb-4">
              <div class="text-overline text-medium-emphasis mb-2">Compare against</div>
              <v-select
                v-model="selectedRefs" :items="refItems" multiple chips closable-chips
                label="Benchmarks / tiers" variant="outlined" density="comfortable"
                hint="Scored against these on the compare page." persistent-hint />
            </v-card>

            <v-card variant="outlined" class="pa-4">
              <v-alert v-if="task.gpu" type="warning" variant="tonal" density="compact" class="mb-3"
                       icon="mdi-alert">
                GPU run on the shared dev box — you'll confirm the cost before it launches.
              </v-alert>
              <v-alert v-if="qrc.anyRunActive" type="info" variant="tonal" density="compact" class="mb-3">
                A run is already active — only one heavy job at a time.
              </v-alert>
              <v-alert v-if="launchError" type="error" variant="tonal" density="compact" class="mb-3">
                {{ launchError }}
              </v-alert>
              <div class="d-flex ga-2">
                <v-btn color="primary" variant="flat" prepend-icon="mdi-play"
                       :loading="launching" :disabled="!auth.user || qrc.anyRunActive"
                       @click="onRun">Run</v-btn>
                <v-btn color="error" variant="tonal" prepend-icon="mdi-stop"
                       :disabled="!auth.user || !qrc.anyRunActive"
                       @click="onStopActive">Stop active</v-btn>
              </div>
            </v-card>
          </div>
        </v-col>
      </v-row>
    </v-container>

    <!-- GPU confirm (hard rule: surface cost + explicit go) -->
    <v-dialog v-model="gpuConfirm" max-width="520">
      <v-card class="pa-2">
        <v-card-title class="text-h6">Confirm GPU run</v-card-title>
        <v-card-text>
          <p class="mb-2">
            <b>{{ task?.label }}</b> runs on the GPU of the shared dev box and consumes
            real, billable GPU time.
          </p>
          <p class="text-body-2 text-medium-emphasis mb-2">
            Wall-clock scales with <code>T</code> × <code>steps</code> × the number of
            conditions. A short validation run first is cheaper than a full sweep.
            Only one heavy job may run at a time.
          </p>
          <p class="text-body-2">Launch it now?</p>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="gpuConfirm = false">Cancel</v-btn>
          <v-btn color="warning" variant="flat" :loading="launching" @click="doLaunch">
            Launch GPU run
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-main>
</template>

<style scoped>
.logo-link { cursor: pointer; transition: opacity 0.15s ease-in-out; }
.logo-link:hover { opacity: 0.8; }
.sticky-side { position: sticky; top: 88px; }
</style>
