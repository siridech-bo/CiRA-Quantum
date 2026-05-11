<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Job } from '@/stores/solve'

const props = defineProps<{ job: Job }>()

const tab = ref<'solution' | 'cqm' | 'validation' | 'raw'>('solution')
const justCopied = ref<string | null>(null)

const validationPassed = computed(
  () => props.job.validation_report?.passed === true,
)
const oracleSkipped = computed(
  () => props.job.validation_report?.oracle_skipped === true,
)

function fmt(obj: any): string {
  if (obj === null || obj === undefined) return ''
  if (typeof obj === 'string') return obj
  return JSON.stringify(obj, null, 2)
}

function copy(label: string, text: string) {
  if (!navigator.clipboard) return
  void navigator.clipboard.writeText(text).then(() => {
    justCopied.value = label
    setTimeout(() => {
      if (justCopied.value === label) justCopied.value = null
    }, 1400)
  })
}
</script>

<template>
  <v-card class="pa-5">
    <div class="d-flex align-center mb-2">
      <v-card-title class="pa-0 text-h6 flex-grow-1">Solution</v-card-title>
      <v-chip
        v-if="job.status === 'complete'"
        size="small"
        color="success"
        variant="tonal"
        prepend-icon="mdi-check-circle"
      >
        Complete
      </v-chip>
      <v-chip
        v-if="job.solve_time_ms !== undefined && job.solve_time_ms !== null"
        size="small"
        variant="tonal"
        class="ml-2"
        prepend-icon="mdi-timer"
      >
        {{ (job.solve_time_ms / 1000).toFixed(2) }}s
      </v-chip>
    </div>

    <v-tabs v-model="tab" color="primary" density="compact" class="mb-2">
      <v-tab value="solution">Solution</v-tab>
      <v-tab value="cqm">CQM</v-tab>
      <v-tab value="validation">Validation</v-tab>
      <v-tab value="raw">Raw LLM</v-tab>
    </v-tabs>

    <v-window v-model="tab">
      <!-- Solution -->
      <v-window-item value="solution">
        <div class="d-flex align-center mb-2">
          <span class="text-caption text-medium-emphasis flex-grow-1">
            Interpreted from the solver's first feasible sample
          </span>
          <v-btn
            v-if="job.interpreted_solution"
            size="small"
            variant="tonal"
            :prepend-icon="justCopied === 'solution' ? 'mdi-check' : 'mdi-content-copy'"
            @click="copy('solution', job.interpreted_solution || '')"
          >
            {{ justCopied === 'solution' ? 'Copied' : 'Copy' }}
          </v-btn>
        </div>
        <pre class="result-text">{{ job.interpreted_solution || '<no interpreted solution>' }}</pre>
      </v-window-item>

      <!-- CQM -->
      <v-window-item value="cqm">
        <div class="d-flex align-center mb-2">
          <span class="text-caption text-medium-emphasis flex-grow-1">
            cqm_v1 JSON the formulation provider emitted ({{ job.num_variables }} vars,
            {{ job.num_constraints }} constraints)
          </span>
          <v-btn
            size="small"
            variant="tonal"
            :prepend-icon="justCopied === 'cqm' ? 'mdi-check' : 'mdi-content-copy'"
            @click="copy('cqm', fmt(job.cqm_json))"
          >
            {{ justCopied === 'cqm' ? 'Copied' : 'Copy' }}
          </v-btn>
        </div>
        <pre class="result-text">{{ fmt(job.cqm_json) || '<no CQM>' }}</pre>
      </v-window-item>

      <!-- Validation -->
      <v-window-item value="validation">
        <div v-if="!job.validation_report" class="text-medium-emphasis">
          No validation report on file.
        </div>
        <template v-else>
          <v-alert
            :type="validationPassed ? 'success' : 'warning'"
            variant="tonal"
            class="mb-3"
            :title="validationPassed ? 'Validation passed' : 'Validation failed'"
          >
            <span v-if="validationPassed">
              All three layers green.
              <span v-if="oracleSkipped">
                Layer A (oracle) skipped because the CQM exceeds the brute-force variable ceiling.
              </span>
            </span>
            <span v-else>
              One or more layers reported a problem. See per-layer detail below.
            </span>
          </v-alert>

          <v-list density="compact" class="bg-transparent">
            <v-list-item>
              <template #prepend>
                <v-icon
                  :color="oracleSkipped ? 'grey' : job.validation_report.oracle_agreement ? 'success' : 'error'"
                  :icon="oracleSkipped ? 'mdi-minus-circle' :
                         job.validation_report.oracle_agreement ? 'mdi-check-circle' : 'mdi-close-circle'"
                />
              </template>
              <v-list-item-title>
                Layer A — Oracle agreement
                <span v-if="oracleSkipped" class="text-caption text-medium-emphasis ml-2">
                  (skipped: too large to brute-force)
                </span>
              </v-list-item-title>
              <v-list-item-subtitle v-if="job.validation_report.energy_oracle !== null">
                Oracle energy = {{ job.validation_report.energy_oracle }}
              </v-list-item-subtitle>
            </v-list-item>

            <v-list-item>
              <template #prepend>
                <v-icon
                  :color="job.validation_report.solver_agreement ? 'success' : 'error'"
                  :icon="job.validation_report.solver_agreement ? 'mdi-check-circle' : 'mdi-close-circle'"
                />
              </template>
              <v-list-item-title>
                Layer B — Multi-solver agreement
              </v-list-item-title>
              <v-list-item-subtitle class="text-caption">
                Skipped during live solves; recovered in Phase 5C dashboard
              </v-list-item-subtitle>
            </v-list-item>

            <v-list-item>
              <template #prepend>
                <v-icon
                  :color="(Object.values(job.validation_report.constraints_active || {}).every(Boolean))
                    ? 'success' : 'warning'"
                  :icon="(Object.values(job.validation_report.constraints_active || {}).every(Boolean))
                    ? 'mdi-check-circle' : 'mdi-alert-circle'"
                />
              </template>
              <v-list-item-title>Layer C — Constraint coverage</v-list-item-title>
              <v-list-item-subtitle>
                <span v-for="(active, label) in job.validation_report.constraints_active" :key="label">
                  <v-chip
                    :color="active ? 'success' : 'warning'"
                    size="x-small"
                    variant="tonal"
                    class="mr-1 mb-1"
                  >
                    {{ label }}
                  </v-chip>
                </span>
              </v-list-item-subtitle>
            </v-list-item>
          </v-list>

          <div v-if="(job.validation_report.warnings || []).length" class="mt-3">
            <div class="text-subtitle-2 mb-1">Warnings</div>
            <v-alert
              v-for="(w, i) in job.validation_report.warnings"
              :key="i"
              type="warning"
              variant="tonal"
              density="compact"
              class="mb-1"
            >
              {{ w }}
            </v-alert>
          </div>
        </template>
      </v-window-item>

      <!-- Raw LLM (not stored in v1 schema; show what is available) -->
      <v-window-item value="raw">
        <div class="text-caption text-medium-emphasis mb-2">
          The exact JSON the formulation provider emitted, before compilation.
          (The platform doesn't archive provider tokens or the surrounding LLM
          chatter — only the parsed CQM JSON above.)
        </div>
        <pre class="result-text">{{ fmt(job.cqm_json) }}</pre>
      </v-window-item>
    </v-window>
  </v-card>
</template>

<style scoped>
.result-text {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 0.75rem;
  font-family: 'JetBrains Mono', Consolas, ui-monospace, monospace;
  font-size: 0.85rem;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 480px;
  overflow: auto;
}
</style>
