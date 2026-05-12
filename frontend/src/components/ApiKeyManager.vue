<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useSolveStore } from '@/stores/solve'

const solve = useSolveStore()

const PROVIDERS = [
  { value: 'claude', title: 'Claude (Anthropic)' },
  { value: 'openai', title: 'OpenAI' },
  { value: 'originqc', title: 'Origin Quantum (Wukong cloud)' },
] as const

const addProvider = ref<'claude' | 'openai' | 'originqc'>('claude')
const addKey = ref('')
const saving = ref(false)
const error = ref<string | null>(null)
const confirmDeleteProvider = ref<string | null>(null)

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
            </v-list-item-subtitle>
            <template #append>
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
        <v-text-field
          v-model="addKey"
          label="API key"
          type="password"
          autocomplete="off"
          prepend-inner-icon="mdi-key"
          density="comfortable"
        />
        <v-alert v-if="error" type="error" variant="tonal" class="mb-2">
          {{ error }}
        </v-alert>
        <v-btn
          color="primary"
          :loading="saving"
          :disabled="!addKey"
          @click="save"
          prepend-icon="mdi-content-save"
        >
          Save key
        </v-btn>
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
