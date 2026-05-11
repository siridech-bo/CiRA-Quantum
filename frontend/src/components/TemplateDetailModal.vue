<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useTemplatesStore, type TemplateDetail } from '@/stores/templates'
import { useSolveStore } from '@/stores/solve'

const props = defineProps<{
  modelValue: boolean
  template: TemplateDetail | null
}>()
const emit = defineEmits<{ (e: 'update:modelValue', v: boolean): void }>()

const templates = useTemplatesStore()
const solve = useSolveStore()
const router = useRouter()

const provider = ref<'claude' | 'openai' | 'local'>('claude')
const useStoredKey = ref(true)
const inlineKey = ref('')
const submitting = ref(false)
const error = ref<string | null>(null)
const justCopied = ref(false)

const hasStoredKey = computed(() => {
  if (provider.value === 'local') return true
  return solve.keys.some((k) => k.provider === provider.value)
})

function close() {
  emit('update:modelValue', false)
}

function copyProblem() {
  if (!props.template || !navigator.clipboard) return
  void navigator.clipboard
    .writeText(props.template.problem_statement)
    .then(() => {
      justCopied.value = true
      setTimeout(() => (justCopied.value = false), 1400)
    })
}

async function tryThis() {
  if (!props.template) return
  error.value = null
  submitting.value = true
  try {
    const job = await templates.solveFromTemplate(props.template.id, {
      provider: provider.value,
      use_stored_key: useStoredKey.value,
      ...(useStoredKey.value ? {} : { api_key: inlineKey.value }),
    })
    // Subscribe the solve store to live status, and navigate to the
    // job-detail page so the user sees the timeline.
    await solve.subscribeToJob(job.id)
    close()
    router.push(`/jobs/${job.id}`)
  } catch (e: any) {
    error.value = e?.response?.data?.error || e?.message || 'submit failed'
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <v-dialog
    :model-value="modelValue"
    @update:model-value="(v: boolean) => emit('update:modelValue', v)"
    max-width="820"
    scrollable
  >
    <v-card v-if="template">
      <v-card-title class="d-flex align-center pa-5 pb-2">
        <span class="text-h6 flex-grow-1">{{ template.title }}</span>
        <v-chip size="small" color="primary" variant="tonal" class="mr-2">
          {{ template.difficulty }}
        </v-chip>
        <v-chip size="small" variant="tonal">{{ template.category }}</v-chip>
        <v-btn
          icon="mdi-close"
          variant="text"
          size="small"
          class="ml-2"
          @click="close"
          aria-label="Close"
        />
      </v-card-title>

      <v-card-text class="pa-5 pt-0">
        <v-card variant="tonal" class="pa-3 mb-3">
          <div class="text-subtitle-2 mb-1">Summary</div>
          <div class="text-body-2">{{ template.summary }}</div>
        </v-card>

        <v-card variant="tonal" class="pa-3 mb-3">
          <div class="d-flex align-center mb-1">
            <span class="text-subtitle-2 flex-grow-1">Problem statement</span>
            <v-btn
              size="x-small"
              variant="text"
              :prepend-icon="justCopied ? 'mdi-check' : 'mdi-content-copy'"
              @click="copyProblem"
            >
              {{ justCopied ? 'Copied' : 'Copy' }}
            </v-btn>
          </div>
          <pre class="problem-statement">{{ template.problem_statement }}</pre>
        </v-card>

        <v-card variant="tonal" class="pa-3 mb-3" v-if="template.real_world_example">
          <div class="text-subtitle-2 mb-1">Real-world example</div>
          <div class="text-body-2">{{ template.real_world_example }}</div>
        </v-card>

        <v-card variant="tonal" class="pa-3 mb-3" v-if="template.expected_pattern">
          <div class="text-subtitle-2 mb-2">Expected pattern</div>
          <div class="pattern-row" v-if="template.expected_pattern.variables">
            <span class="pattern-label">Variables:</span>
            <code>{{ template.expected_pattern.variables }}</code>
          </div>
          <div class="pattern-row" v-if="template.expected_pattern.objective">
            <span class="pattern-label">Objective:</span>
            <code>{{ template.expected_pattern.objective }}</code>
          </div>
          <div class="pattern-row" v-if="template.expected_pattern.constraints">
            <span class="pattern-label">Constraints:</span>
            <code>{{ template.expected_pattern.constraints }}</code>
          </div>
        </v-card>

        <v-card variant="tonal" class="pa-3 mb-3" v-if="template.learning_notes">
          <div class="text-subtitle-2 mb-1">Learning notes</div>
          <div class="text-body-2">{{ template.learning_notes }}</div>
        </v-card>

        <v-card v-if="template.module" variant="tonal" class="pa-3 mb-3" color="accent">
          <div class="text-subtitle-2 mb-1">
            <v-icon icon="mdi-school" start size="small" />
            Part of Module: {{ template.module.module_id }}
            (lesson #{{ template.module.order }})
          </div>
          <ul class="text-body-2 mt-1 pl-4">
            <li v-for="obj in template.module.learning_objectives" :key="obj">
              {{ obj }}
            </li>
          </ul>
          <div
            v-if="(template.module.prerequisites || []).length"
            class="text-caption mt-2 text-medium-emphasis"
          >
            Prerequisites:
            <v-chip
              v-for="p in template.module.prerequisites"
              :key="p"
              size="x-small"
              variant="outlined"
              class="ml-1"
            >
              {{ p }}
            </v-chip>
          </div>
        </v-card>

        <v-divider class="my-3" />

        <div class="text-subtitle-2 mb-2">Run this example</div>
        <v-row no-gutters class="ga-3">
          <v-col cols="12" sm="6">
            <v-select
              v-model="provider"
              :items="[
                { value: 'claude', title: 'Claude (Sonnet 4.6)' },
                { value: 'openai', title: 'OpenAI (GPT-5-mini)' },
                { value: 'local',  title: 'Local LLM (Ollama)' },
              ]"
              item-title="title"
              item-value="value"
              label="Provider"
              density="comfortable"
            />
          </v-col>
          <v-col cols="12" sm="6" class="d-flex align-center px-sm-3">
            <template v-if="provider !== 'local'">
              <v-checkbox
                v-model="useStoredKey"
                :disabled="!hasStoredKey"
                hide-details
                density="comfortable"
                :label="
                  hasStoredKey
                    ? `Use stored ${provider} key`
                    : `No stored ${provider} key`
                "
              />
            </template>
            <v-chip v-else color="info" variant="tonal">Free (local)</v-chip>
          </v-col>
        </v-row>
        <v-text-field
          v-if="provider !== 'local' && !useStoredKey"
          v-model="inlineKey"
          :label="`${provider} API key (one-shot)`"
          type="password"
          autocomplete="off"
          density="comfortable"
          class="mt-2"
        />
        <v-alert v-if="error" type="error" variant="tonal" class="mt-3">
          {{ error }}
        </v-alert>
      </v-card-text>

      <v-card-actions class="pa-5 pt-0">
        <v-btn variant="outlined" @click="copyProblem">
          {{ justCopied ? 'Copied' : 'Copy problem statement' }}
        </v-btn>
        <v-spacer />
        <v-btn @click="close">Close</v-btn>
        <v-btn
          color="primary"
          :loading="submitting"
          :disabled="
            (provider !== 'local' && useStoredKey && !hasStoredKey) ||
            (provider !== 'local' && !useStoredKey && !inlineKey)
          "
          @click="tryThis"
          prepend-icon="mdi-rocket-launch"
        >
          Try this example
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.problem-statement {
  font-family: 'JetBrains Mono', Consolas, ui-monospace, monospace;
  font-size: 0.85rem;
  white-space: pre-wrap;
  margin: 0;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  padding: 0.5rem 0.75rem;
}
.pattern-row {
  margin: 0.25rem 0;
}
.pattern-label {
  font-weight: 600;
  margin-right: 0.5rem;
}
code {
  font-family: 'JetBrains Mono', Consolas, ui-monospace, monospace;
  font-size: 0.85rem;
}
</style>
