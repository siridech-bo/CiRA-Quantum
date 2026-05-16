<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useSolveStore } from '@/stores/solve'

const solve = useSolveStore()

const PROVIDERS = [
  { value: 'claude', title: 'Claude (Anthropic)' },
  { value: 'openai', title: 'OpenAI' },
  { value: 'originqc', title: 'Origin Quantum (Wukong cloud)' },
  { value: 'ibm_quantum', title: 'IBM Quantum (Open Plan)' },
] as const

interface ProviderHelp {
  /** Marketing-style one-line summary. */
  blurb: string
  /** URL to open in a new tab for the user to sign up / retrieve their key. */
  consoleUrl: string
  /** Display label for the open-in-new-tab button (typically the URL's host). */
  consoleHost: string
  /** Step-by-step instructions. Each item is one short line. */
  steps: string[]
  /** Whether the provider has a usable free tier. Used to pick the chip color. */
  freeTier: boolean
  /** Free-tier note (when applicable) or pricing note. Optional. */
  pricingNote?: string
  /** Optional format hint to show next to the API key input. */
  formatHint?: string
}

const PROVIDER_HELP: Record<string, ProviderHelp> = {
  claude: {
    blurb: 'Anthropic Claude — used by the LLM formulator to turn plain-English problems into CQM JSON.',
    consoleUrl: 'https://console.anthropic.com/settings/keys',
    consoleHost: 'console.anthropic.com',
    steps: [
      'Sign in (or sign up) at console.anthropic.com.',
      'Open Settings → API Keys.',
      'Click "Create Key" and copy the resulting value.',
    ],
    freeTier: false,
    pricingNote: 'Pay-as-you-go (Claude Sonnet 4.6: $3 / 1M input · $15 / 1M output tokens). A typical solve costs about $0.01–$0.05.',
    formatHint: 'Starts with sk-ant-...',
  },
  openai: {
    blurb: 'OpenAI — alternative LLM formulator (GPT-5-mini by default).',
    consoleUrl: 'https://platform.openai.com/api-keys',
    consoleHost: 'platform.openai.com',
    steps: [
      'Sign in (or sign up) at platform.openai.com.',
      'Open API Keys.',
      'Click "Create new secret key" and copy the value immediately — it is shown only once.',
    ],
    freeTier: false,
    pricingNote: 'Pay-as-you-go. GPT-5-mini is cheaper than Claude per token; a typical solve costs about $0.001–$0.01. New accounts sometimes get $5 in trial credit.',
    formatHint: 'Starts with sk-...',
  },
  originqc: {
    blurb: 'Origin Quantum — submits the trained QAOA circuit to real superconducting hardware (Wukong) or their cloud simulator.',
    consoleUrl: 'https://console.originqc.com.cn',
    consoleHost: 'console.originqc.com.cn',
    steps: [
      'Sign in (or sign up) at console.originqc.com.cn — academic email is fine.',
      'Open your account / Personal Center → API Key (or "My API key").',
      'Generate or copy the existing API key. It is a long hex-encoded string.',
    ],
    freeTier: true,
    pricingNote: 'Free quota for academic users. Submissions queue on the Origin scheduler — sometimes seconds, sometimes minutes. We auto-retry transient pilot-task errors.',
    formatHint: 'Long hex-encoded string. Recent keys start ~890… (older PKCS#8-style 30 2e 02 01… keys may no longer authenticate).',
  },
  ibm_quantum: {
    blurb: 'IBM Quantum — Free Open Plan with access to real 127+ qubit superconducting QPUs (Eagle, Heron r2).',
    consoleUrl: 'https://quantum.ibm.com',
    consoleHost: 'quantum.ibm.com',
    steps: [
      'Sign in (or sign up free) at quantum.ibm.com — academic email is welcome.',
      'On the Dashboard your "API token" is shown at the top-right with a Copy button.',
      'Paste the copied value below and save.',
    ],
    freeTier: true,
    pricingNote: '10 min/month free on real QPUs + 180 min bonus for 12 months once you log 20 min of total usage. No credit card. Jobs queue 30 s to several hours depending on chip load.',
    formatHint: 'Long alphanumeric string from the IBM Quantum dashboard.',
  },
}

// Providers that the backend has a liveness test for. When we add
// /api/keys/<provider>/test for other providers, append here.
const TESTABLE_PROVIDERS = new Set(['originqc', 'ibm_quantum'])

const addProvider = ref<'claude' | 'openai' | 'originqc' | 'ibm_quantum'>('claude')

const currentHelp = computed<ProviderHelp | undefined>(
  () => PROVIDER_HELP[addProvider.value],
)
const currentProviderTitle = computed(
  () => PROVIDERS.find((p) => p.value === addProvider.value)?.title || addProvider.value,
)
const addKey = ref('')
const saving = ref(false)
const error = ref<string | null>(null)
const confirmDeleteProvider = ref<string | null>(null)

// Per-provider test state. Keyed by provider name so we can show
// rolling feedback if the user clicks Test on several keys.
const testing = ref<Set<string>>(new Set())
const testResult = ref<Record<string, {
  ok: boolean
  message?: string
  error?: string
  elapsed_ms?: number
  at: number
}>>({})

async function runTest(provider: string) {
  testing.value.add(provider)
  try {
    const r = await solve.testKey(provider)
    testResult.value = {
      ...testResult.value,
      [provider]: { ...r, at: Date.now() },
    }
  } finally {
    testing.value.delete(provider)
  }
}

async function save() {
  error.value = null
  saving.value = true
  try {
    await solve.putKey(addProvider.value, addKey.value)
    addKey.value = ''
  } catch (e: any) {
    error.value = e?.response?.data?.error || e?.message || 'Save failed'
  } finally {
    saving.value = false
  }
}

async function remove(provider: string) {
  await solve.deleteKey(provider)
  confirmDeleteProvider.value = null
}

function fmtDate(s: string): string {
  return new Date(s).toLocaleString()
}

onMounted(() => {
  void solve.loadKeys()
})
</script>

<template>
  <v-row>
    <v-col cols="12" md="6">
      <v-card class="pa-5 h-100">
        <v-card-title class="text-h6 pa-0 mb-2">Stored providers</v-card-title>
        <v-card-subtitle class="pa-0 mb-4">
          Keys are encrypted at rest. The platform never echoes them back.
        </v-card-subtitle>

        <div v-if="!solve.keys.length" class="text-medium-emphasis text-center py-6">
          No keys stored yet — add one on the right.
        </div>
        <v-list v-else density="comfortable" class="bg-transparent">
          <v-list-item
            v-for="k in solve.keys"
            :key="k.provider"
            class="px-0"
          >
            <template #prepend>
              <v-icon icon="mdi-key" />
            </template>
            <v-list-item-title class="font-weight-medium">
              {{ k.provider }}
            </v-list-item-title>
            <v-list-item-subtitle>
              Added {{ fmtDate(k.created_at) }}
              <span v-if="testResult[k.provider]" class="ml-2">
                <v-chip
                  size="x-small"
                  :color="testResult[k.provider].ok ? 'success' : 'error'"
                  variant="tonal"
                >
                  <v-icon
                    :icon="testResult[k.provider].ok ? 'mdi-check' : 'mdi-alert-circle'"
                    size="x-small"
                    start
                  />
                  {{
                    testResult[k.provider].ok
                      ? `Auth OK (${testResult[k.provider].elapsed_ms ?? '?'} ms)`
                      : 'Auth failed'
                  }}
                </v-chip>
                <span
                  v-if="!testResult[k.provider].ok"
                  class="text-caption text-error ml-2"
                >
                  {{ testResult[k.provider].error }}
                </span>
              </span>
            </v-list-item-subtitle>
            <template #append>
              <v-btn
                v-if="TESTABLE_PROVIDERS.has(k.provider)"
                size="small"
                variant="text"
                prepend-icon="mdi-cloud-check-outline"
                :loading="testing.has(k.provider)"
                :aria-label="`Test ${k.provider} key`"
                class="mr-1"
                @click="runTest(k.provider)"
              >
                Test
              </v-btn>
              <v-btn
                icon="mdi-delete-outline"
                size="small"
                variant="text"
                color="error"
                @click="confirmDeleteProvider = k.provider"
                :aria-label="`Delete ${k.provider} key`"
              />
            </template>
          </v-list-item>
        </v-list>
      </v-card>
    </v-col>

    <v-col cols="12" md="6">
      <v-card class="pa-5 h-100">
        <v-card-title class="text-h6 pa-0 mb-2">Add or replace a key</v-card-title>
        <v-card-subtitle class="pa-0 mb-4">
          Pasting a new value for a provider you've already stored will
          overwrite the previous one.
        </v-card-subtitle>

        <v-select
          v-model="addProvider"
          :items="PROVIDERS"
          item-title="title"
          item-value="value"
          label="Provider"
          density="comfortable"
        />

        <!-- Per-provider help card. Drives off the selected provider so
             switching the dropdown updates the instructions live. -->
        <div v-if="currentHelp" class="provider-help mb-3 pa-3">
          <div class="d-flex align-center mb-2">
            <v-icon icon="mdi-information-outline" size="small" class="mr-2" color="info" />
            <span class="text-subtitle-2 flex-grow-1">
              How to get your {{ currentProviderTitle }} key
            </span>
            <v-chip
              size="x-small"
              :color="currentHelp.freeTier ? 'success' : 'warning'"
              variant="tonal"
            >
              {{ currentHelp.freeTier ? 'Free tier' : 'Paid' }}
            </v-chip>
          </div>
          <div class="text-body-2 text-medium-emphasis mb-2">
            {{ currentHelp.blurb }}
          </div>
          <v-btn
            :href="currentHelp.consoleUrl"
            target="_blank"
            rel="noopener noreferrer"
            variant="tonal"
            color="primary"
            size="small"
            prepend-icon="mdi-open-in-new"
            class="mb-2"
          >
            Open {{ currentHelp.consoleHost }}
          </v-btn>
          <ol class="help-steps text-body-2">
            <li v-for="(step, i) in currentHelp.steps" :key="i">{{ step }}</li>
          </ol>
          <div v-if="currentHelp.pricingNote" class="text-caption text-medium-emphasis mt-2">
            <v-icon icon="mdi-cash-multiple" size="x-small" /> {{ currentHelp.pricingNote }}
          </div>
        </div>

        <v-text-field
          v-model="addKey"
          label="API key"
          type="password"
          autocomplete="off"
          prepend-inner-icon="mdi-key"
          density="comfortable"
          :hint="currentHelp?.formatHint"
          persistent-hint
        />
        <v-alert v-if="error" type="error" variant="tonal" class="mb-2 mt-2">
          {{ error }}
        </v-alert>
        <div class="d-flex align-center mt-3">
          <v-btn
            color="primary"
            :loading="saving"
            :disabled="!addKey"
            @click="save"
            prepend-icon="mdi-content-save"
          >
            Save key
          </v-btn>
          <span
            v-if="TESTABLE_PROVIDERS.has(addProvider)"
            class="text-caption text-medium-emphasis ml-3"
          >
            After saving, click <strong>TEST</strong> on the stored row to verify the key works.
          </span>
        </div>
      </v-card>
    </v-col>

    <v-dialog
      :model-value="!!confirmDeleteProvider"
      max-width="400"
      @update:model-value="(v: boolean) => { if (!v) confirmDeleteProvider = null }"
    >
      <v-card>
        <v-card-title>Delete this stored key?</v-card-title>
        <v-card-text>
          You'll need to re-enter the key (inline or by storing it again)
          before your next {{ confirmDeleteProvider }} solve.
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn @click="confirmDeleteProvider = null">Cancel</v-btn>
          <v-btn
            color="error"
            variant="flat"
            @click="confirmDeleteProvider && remove(confirmDeleteProvider)"
          >
            Delete
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-row>
</template>

<style scoped>
.provider-help {
  background: rgba(33, 150, 243, 0.04);
  border-left: 3px solid rgb(var(--v-theme-info));
  border-radius: 4px;
}
.help-steps {
  margin: 0;
  padding-left: 1.2rem;
}
.help-steps li {
  margin-bottom: 0.25rem;
  line-height: 1.4;
}
</style>
