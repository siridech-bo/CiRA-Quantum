<script setup lang="ts">
/**
 * SolverPicker — tier-grouped checklist of registered solvers.
 *
 * Loads /api/solvers on mount; renders solvers grouped by ``tier`` with
 * the tier label as a section header and a colored chip per row. The
 * "recommended_default" hint from the backend drives the initial
 * selection — usually that means all locally-runnable tiers ON and the
 * real-QPU tier OFF (it's slow and costs cloud quota).
 *
 * v-model: ``string[]`` of selected solver names.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useSolveStore, type SolverInfo } from '@/stores/solve'

const solve = useSolveStore()

const model = defineModel<string[]>({ required: true })

const loading = ref(true)
const loadError = ref<string | null>(null)

const storedKeyProviders = computed(
  () => new Set(solve.keys.map((k) => k.provider)),
)

function isBlocked(row: SolverInfo): boolean {
  return !!row.requires_key && !storedKeyProviders.value.has(row.requires_key)
}

function blockedReason(row: SolverInfo): string {
  return `No "${row.requires_key}" key on file. Add one in the API Keys tab to enable this solver.`
}

const groups = computed(() => {
  const by: Record<string, { label: string; color: string; rows: SolverInfo[] }> = {}
  const tierOrder = [
    'classical_exact',
    'qubo_heuristic',
    'quantum_inspired',
    'quantum_simulator',
    'quantum_qpu',
    'other',
  ]
  for (const s of solve.solvers) {
    const key = s.tier || 'other'
    if (!by[key]) {
      by[key] = { label: s.tier_label, color: s.tier_color, rows: [] }
    }
    by[key].rows.push(s)
  }
  return tierOrder
    .filter((t) => by[t])
    .map((t) => ({ tier: t, ...by[t] }))
})

const totalCount = computed(() => solve.solvers.length)
const selectedCount = computed(() => model.value.length)

function toggle(name: string) {
  const row = solve.solvers.find((s) => s.name === name)
  if (row && isBlocked(row) && !model.value.includes(name)) {
    return  // BYOK gate — silent no-op until the user stores the required key
  }
  const i = model.value.indexOf(name)
  if (i >= 0) {
    model.value = model.value.filter((s) => s !== name)
  } else {
    model.value = [...model.value, name]
  }
}

function selectAll() {
  // BYOK-blocked solvers are excluded even from "All" — selecting them
  // would just queue a guaranteed error row at solve time.
  model.value = solve.solvers.filter((s) => !isBlocked(s)).map((s) => s.name)
}

function selectRecommended() {
  model.value = solve.solvers
    .filter((s) => s.recommended_default && !isBlocked(s))
    .map((s) => s.name)
}

function selectNone() {
  model.value = []
}

onMounted(async () => {
  try {
    const [list] = await Promise.all([solve.loadSolvers(), solve.loadKeys()])
    // Initial selection = all recommended_default solvers minus any
    // BYOK-blocked rows. Matches the user-requested "all available
    // solvers (overwhelming but thorough)" preference while keeping
    // the real-QPU tier opt-in until a key is on file.
    if (model.value.length === 0) {
      model.value = list
        .filter((s) => s.recommended_default && !isBlocked(s))
        .map((s) => s.name)
    }
  } catch (e: any) {
    loadError.value =
      e?.response?.data?.error || e?.message || 'Failed to load solver registry'
  } finally {
    loading.value = false
  }
})

// Keep the selection coherent if the registry changes shape (rare —
// mostly relevant during dev when an optional dep is hot-installed).
watch(
  () => solve.solvers,
  (list) => {
    const registered = new Set(list.map((s) => s.name))
    const kept = model.value.filter((s) => registered.has(s))
    if (kept.length !== model.value.length) {
      model.value = kept
    }
  },
)
</script>

<template>
  <div class="solver-picker">
    <div class="d-flex align-center mb-2">
      <div class="text-subtitle-2 flex-grow-1">
        <v-icon icon="mdi-set-merge" start size="small" />
        Solvers
        <span class="text-caption text-medium-emphasis ml-1">
          ({{ selectedCount }} / {{ totalCount }})
        </span>
      </div>
      <v-btn-toggle density="compact" variant="text" divided>
        <v-btn size="x-small" @click="selectRecommended" title="Local-only tiers — fast, no cloud quota">
          Recommended
        </v-btn>
        <v-btn size="x-small" @click="selectAll" title="Every registered solver including the real QPU">
          All
        </v-btn>
        <v-btn size="x-small" @click="selectNone" title="Clear selection">
          None
        </v-btn>
      </v-btn-toggle>
    </div>

    <v-skeleton-loader v-if="loading" type="list-item, list-item, list-item" />
    <v-alert v-else-if="loadError" type="error" variant="tonal" density="compact">
      {{ loadError }}
    </v-alert>

    <template v-else>
      <div v-for="g in groups" :key="g.tier" class="tier-block mb-2">
        <div class="d-flex align-center mb-1">
          <v-chip size="x-small" :color="g.color" variant="flat" class="tier-label">
            {{ g.label }}
          </v-chip>
        </div>
        <div
          v-for="row in g.rows"
          :key="row.name"
          class="solver-row d-flex align-center"
          :class="{
            'solver-row--active': model.includes(row.name),
            'solver-row--blocked': isBlocked(row),
          }"
          @click="toggle(row.name)"
        >
          <v-checkbox
            :model-value="model.includes(row.name)"
            :disabled="isBlocked(row)"
            density="compact"
            hide-details
            :color="g.color"
            class="solver-check"
            @click.stop
            @update:model-value="toggle(row.name)"
          />
          <div class="flex-grow-1">
            <div class="d-flex align-center">
              <code class="solver-name">{{ row.name }}</code>
              <span class="text-caption text-medium-emphasis ml-2">
                {{ row.source }} · v{{ row.version }}
              </span>
              <v-tooltip v-if="isBlocked(row)" location="top">
                <template #activator="{ props }">
                  <v-icon
                    v-bind="props"
                    icon="mdi-key-off-outline"
                    size="x-small"
                    color="error"
                    class="ml-1"
                  />
                </template>
                <span>{{ blockedReason(row) }}</span>
              </v-tooltip>
              <v-tooltip v-else-if="row.warning" location="top">
                <template #activator="{ props }">
                  <v-icon
                    v-bind="props"
                    icon="mdi-alert-circle-outline"
                    size="x-small"
                    color="warning"
                    class="ml-1"
                  />
                </template>
                <span>{{ row.warning }}</span>
              </v-tooltip>
            </div>
            <div v-if="isBlocked(row)" class="text-caption text-error">
              Needs an <code>{{ row.requires_key }}</code> key on file
            </div>
            <div v-else-if="row.hardware" class="text-caption text-medium-emphasis">
              {{ row.hardware }}
            </div>
          </div>
        </div>
      </div>
    </template>

    <v-alert
      v-if="!loading && !loadError && selectedCount === 0"
      type="warning"
      variant="tonal"
      density="compact"
      class="mt-2"
    >
      Pick at least one solver to compare.
    </v-alert>
  </div>
</template>

<style scoped>
.solver-picker {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  padding: 0.75rem;
}
.tier-label {
  font-weight: 600;
  letter-spacing: 0.02em;
}
.solver-row {
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 4px;
  transition: background 0.1s ease-in-out;
}
.solver-row:hover {
  background: rgba(255, 255, 255, 0.04);
}
.solver-row--active {
  background: rgba(124, 58, 237, 0.06);
}
.solver-row--blocked {
  opacity: 0.65;
  cursor: not-allowed;
}
.solver-row--blocked:hover {
  background: rgba(255, 255, 255, 0.02);
}
.solver-check {
  flex: 0 0 auto;
  margin-right: 0.25rem;
}
.solver-name {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
}
</style>
