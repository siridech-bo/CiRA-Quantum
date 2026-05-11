<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useSolveStore } from '@/stores/solve'

const solve = useSolveStore()

const MAX_LEN = 8000
const PROVIDERS = [
  { value: 'claude', title: 'Claude (Sonnet 4.6)' },
  { value: 'openai', title: 'OpenAI (GPT-5-mini)' },
  { value: 'local',  title: 'Local LLM (Ollama)' },
] as const
type Provider = (typeof PROVIDERS)[number]['value']

const statement = ref('')
const provider = ref<Provider>('claude')
const useStoredKey = ref(true)
const apiKey = ref('')
const submitError = ref<string | null>(null)
const submitting = ref(false)

const charCount = computed(() => statement.value.length)
const overLength = computed(() => charCount.value > MAX_LEN)

const hasStoredKey = computed(() => {
  if (provider.value === 'local') return true   // Ollama needs no key.
  return solve.keys.some((k) => k.provider === provider.value)
})

// When the provider changes, default `useStoredKey` to whether one exists
// for that provider. The user can still flip the checkbox manually.
watch(provider, () => {
  useStoredKey.value = hasStoredKey.value
})

// Cost preview — back-of-envelope, conservative on purpose. Matches the
// estimate_cost heuristic in app/formulation/claude.py + openai.py.
const estimatedCost = computed(() => {
  if (provider.value === 'local') return 0.0
  const promptChars = statement.value.length
  const promptTokens = 5500 + Math.max(1, promptChars / 4)
  const completionTokens = Math.min(8192, Math.max(800, promptTokens * 0.25))
  const rates = {
    claude: { input: 3.0 / 1_000_000, output: 15.0 / 1_000_000 },
    openai: { input: 0.25 / 1_000_000, output: 2.0 / 1_000_000 },
  } as const
  const r = rates[provider.value]
  return promptTokens * r.input + completionTokens * r.output
})

const canSubmit = computed(() => {
  if (solve.isRunning) return false
  if (!statement.value.trim()) return false
  if (overLength.value) return false
  if (provider.value !== 'local' && useStoredKey.value && !hasStoredKey.value) return false
  if (provider.value !== 'local' && !useStoredKey.value && !apiKey.value) return false
  return !submitting.value
})

async function submit() {
  submitError.value = null
  submitting.value = true
  try {
    await solve.submitProblem({
      problem_statement: statement.value,
      provider: provider.value,
      use_stored_key: useStoredKey.value,
      ...(useStoredKey.value ? {} : { api_key: apiKey.value }),
    })
  } catch (e: any) {
    submitError.value = e?.response?.data?.error || e?.message || 'Submit failed'
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  // Best-effort: surface stored keys so the "use stored" checkbox starts
  // in the right position.
  try {
    await solve.loadKeys()
    useStoredKey.value = hasStoredKey.value
  } catch {
    // unauthenticated or transient — the store will retry on next mount
  }
})
</script>

<template>
  <v-card class="pa-5">
    <v-card-title class="text-h6">Describe your problem</v-card-title>
    <v-card-subtitle>
      In plain English. The platform formulates, validates, solves, and explains.
    </v-card-subtitle>

    <v-textarea
      v-model="statement"
      label="Problem statement"
      rows="6"
      auto-grow
      :counter="MAX_LEN"
      :error="overLength"
      :error-messages="overLength ? `Exceeded ${MAX_LEN} characters` : ''"
      class="mt-3"
    />

    <v-row class="mt-1">
      <v-col cols="12" sm="6">
        <v-select
          v-model="provider"
          :items="PROVIDERS"
          item-title="title"
          item-value="value"
          label="Formulation provider"
          density="comfortable"
        />
      </v-col>
      <v-col cols="12" sm="6" class="d-flex align-center">
        <v-chip
          v-if="provider !== 'local'"
          variant="tonal"
          :color="estimatedCost < 0.1 ? 'success' : 'warning'"
          prepend-icon="mdi-cash"
        >
          ~${{ estimatedCost.toFixed(4) }} per solve
        </v-chip>
        <v-chip v-else variant="tonal" color="info" prepend-icon="mdi-laptop">
          Free (runs on your GPU)
        </v-chip>
      </v-col>
    </v-row>

    <template v-if="provider !== 'local'">
      <v-checkbox
        v-model="useStoredKey"
        :disabled="!hasStoredKey"
        density="comfortable"
        hide-details
        :label="
          hasStoredKey
            ? `Use stored ${provider} key`
            : `No stored ${provider} key (add one in API Keys tab, or enter inline below)`
        "
      />
      <v-text-field
        v-if="!useStoredKey"
        v-model="apiKey"
        :label="`${provider} API key`"
        type="password"
        autocomplete="off"
        prepend-inner-icon="mdi-key"
        density="comfortable"
      />
    </template>

    <v-alert v-if="submitError" type="error" variant="tonal" class="mt-3">
      {{ submitError }}
    </v-alert>

    <v-card-actions class="mt-2">
      <v-spacer />
      <v-btn
        color="primary"
        :disabled="!canSubmit"
        :loading="submitting"
        prepend-icon="mdi-rocket-launch"
        @click="submit"
      >
        {{ solve.isRunning ? 'A solve is already running…' : 'Solve' }}
      </v-btn>
    </v-card-actions>
  </v-card>
</template>
