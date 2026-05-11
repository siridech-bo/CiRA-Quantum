<script setup lang="ts">
import { computed } from 'vue'
import type { Job } from '@/stores/solve'

/**
 * Phase 5B match badge — renders on jobs that came from a template.
 *
 * Three states:
 *   • match     — solver's value within 1e-3 of expected_optimum
 *   • mismatch  — solver returned a different value
 *   • unknown   — no expected_optimum on file (badge hidden)
 *
 * The matching rule mirrors the backend's RunRecord
 * `converged_to_expected` logic (DECISIONS.md, Phase 2 v2 addendum).
 */
const props = defineProps<{ job: Job }>()

const expected = computed(() =>
  props.job.expected_optimum ?? null,
)

const actual = computed<number | null>(() => {
  // Pull the user-facing objective from the validation report when
  // present, falling back to interpretation-time data if not.
  const report = props.job.validation_report
  if (!report) return null
  // For minimize problems, `energy_oracle` IS user-facing.
  // For maximize problems, the oracle's stored value is already
  // sign-flipped by the orchestrator (see app/optimization/validation.py).
  if (typeof report.energy_oracle === 'number') return report.energy_oracle
  return null
})

const state = computed<'match' | 'mismatch' | 'unknown'>(() => {
  if (expected.value === null || expected.value === undefined) return 'unknown'
  if (actual.value === null) return 'unknown'
  const denom = Math.max(Math.abs(expected.value), 1.0)
  return Math.abs(actual.value - expected.value) / denom <= 1e-3
    ? 'match'
    : 'mismatch'
})

const tooltipText = computed(() => {
  if (state.value === 'match') return 'Solver matches the template\'s documented optimum.'
  if (state.value === 'mismatch')
    return (
      'Solver value differs from the template\'s documented optimum. ' +
      'Possible causes: LLM picked a different (still-valid) formulation, ' +
      'the solver didn\'t reach the optimum, or the template\'s expected value is wrong.'
    )
  return ''
})
</script>

<template>
  <v-tooltip
    v-if="state !== 'unknown'"
    :text="tooltipText"
    location="bottom"
    open-on-hover
  >
    <template #activator="{ props: tProps }">
      <v-chip
        v-bind="tProps"
        :color="state === 'match' ? 'success' : 'warning'"
        variant="tonal"
        :prepend-icon="state === 'match' ? 'mdi-check-circle' : 'mdi-alert-circle'"
      >
        <span v-if="state === 'match'">
          ✓ Matches expected optimum: {{ expected }}
        </span>
        <span v-else>
          ✗ Got {{ actual }} (expected {{ expected }})
        </span>
      </v-chip>
    </template>
  </v-tooltip>
</template>
