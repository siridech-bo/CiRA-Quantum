<script setup lang="ts">
import { ref } from 'vue'
import { useBenchmarksStore } from '@/stores/benchmarks'

/**
 * Phase 5C — "Cite this record" button.
 *
 * Loads the BibTeX entry on demand and exposes a copy-to-clipboard.
 * A short inline citation is also offered for users who don't want
 * the full BibTeX block. The reproducibility hash is part of the
 * entry's `note` field so a reader following the citation can pin
 * the exact preserved record.
 */
const props = defineProps<{
  recordId: string
  variant?: 'flat' | 'outlined' | 'text'
  size?: 'x-small' | 'small' | 'default' | 'large'
}>()

const benchmarks = useBenchmarksStore()

const open = ref(false)
const citation = ref<string>('')
const kind = ref<'bibtex' | 'string'>('bibtex')
const loading = ref(false)
const copied = ref(false)

async function openDialog() {
  open.value = true
  await load()
}

async function load() {
  loading.value = true
  copied.value = false
  const text = await benchmarks.fetchCitation(props.recordId, kind.value)
  citation.value = text || '(failed to load citation)'
  loading.value = false
}

async function copy() {
  if (!citation.value) return
  await navigator.clipboard.writeText(citation.value)
  copied.value = true
  setTimeout(() => (copied.value = false), 1500)
}

async function switchKind(newKind: 'bibtex' | 'string') {
  kind.value = newKind
  await load()
}
</script>

<template>
  <v-btn
    :variant="variant || 'outlined'"
    :size="size || 'small'"
    prepend-icon="mdi-format-quote-close"
    @click="openDialog"
  >
    Cite
  </v-btn>

  <v-dialog v-model="open" max-width="720">
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon icon="mdi-format-quote-close" start />
        Citation for record
        <v-spacer />
        <v-btn icon="mdi-close" variant="text" size="small" @click="open = false" />
      </v-card-title>

      <v-card-subtitle class="pa-4 pt-0 text-medium-emphasis">
        Every CiRA Quantum Benchmark record is uniquely citable. The
        entry's <code>note</code> field pins the exact reproducibility
        hash and hardware ID.
      </v-card-subtitle>

      <v-card-text>
        <v-btn-toggle
          v-model="kind"
          mandatory
          density="compact"
          variant="outlined"
          divided
          class="mb-3"
        >
          <v-btn value="bibtex" @click="switchKind('bibtex')">BibTeX</v-btn>
          <v-btn value="string" @click="switchKind('string')">Inline</v-btn>
        </v-btn-toggle>

        <v-skeleton-loader v-if="loading" type="paragraph, paragraph" />
        <pre v-else class="citation-block">{{ citation }}</pre>
      </v-card-text>

      <v-card-actions>
        <v-spacer />
        <v-btn @click="open = false">Close</v-btn>
        <v-btn
          color="primary"
          variant="flat"
          :prepend-icon="copied ? 'mdi-check' : 'mdi-content-copy'"
          :disabled="loading || !citation"
          @click="copy"
        >
          {{ copied ? 'Copied!' : 'Copy' }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.citation-block {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 300px;
  overflow-y: auto;
}
</style>
