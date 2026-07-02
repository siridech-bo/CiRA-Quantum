<script setup lang="ts">
/**
 * ApprovalPanel — surfaces the LLM's formulation + preflight estimate
 * so the user can sanity-check the encoding before any solver runs.
 *
 * Shown only when ``job.status === 'awaiting_approval'``. The
 * orchestrator pauses at this state after stage 3 (validation) and
 * writes the preflight summary (lowered qubit count + per-QAOA-tier
 * verdict) into the ``preflight`` field. Clicking Approve calls
 * ``POST /api/jobs/<id>/approve`` which resumes the pipeline at
 * stage 4; Cancel deletes the job.
 *
 * The point of this surface is to catch LLM-encoding regressions
 * before solvers burn time — e.g. the Max-Cut formulation that shipped
 * with 13 variables / 28 constraints and lowered to 69 qubits, which
 * skipped every QAOA tier.
 */
import { computed, ref } from 'vue'
import type { Job } from '@/stores/solve'
import { useSolveStore } from '@/stores/solve'

const props = defineProps<{ job: Job }>()
const solve = useSolveStore()

const approving = ref(false)
const cancelling = ref(false)
const error = ref<string | null>(null)
const showCqm = ref(false)

const preflight = computed(() => props.job.preflight)
const qpuFootprint = computed(() => preflight.value?.qpu_footprint ?? null)

// Formulation-routing audit trail (2026-07-02). When the classifier
// confidently matched a hardcoded family, ``route === 'hardcoded'`` and
// ``parameters`` holds the structured translation extracted from the
// user's prose. Rendered as a "problem statement (translated)" text
// box so the user can eyeball what the LLM saw before approving.
const route = computed(() => props.job.formulation_route ?? null)
const routeIsHardcoded = computed(() => route.value?.route === 'hardcoded')
const familyLabel = computed(() => {
  const f = route.value?.family
  if (!f) return ''
  return f.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
})
const translatedStatement = computed(() => {
  const r = route.value
  if (!r || !r.family) return ''
  const params = r.parameters ?? {}
  const paramLines = Object.entries(params).map(
    ([k, v]) => `  ${k} = ${JSON.stringify(v)}`,
  )
  return [
    `family: ${r.family}`,
    ...paramLines,
  ].join('\n')
})

// Validation report from stage 3. When validation found a problem
// (oracle disagreement, infeasible CQM, missing constraints) it goes
// here — the gate surfaces it as advisory so the user can decide
// whether to proceed on a flagged CQM. Typical failure on Max-Cut
// is the LLM emitting the right structure but mis-counting edge
// degrees so the brute-force oracle's optimum doesn't match the
// template's expected value (2026-07-01 demo).
const validation = computed(() => props.job.validation_report ?? null)
const validationPassed = computed(
  () => validation.value?.passed === true,
)
const validationWarnings = computed<string[]>(
  () => Array.isArray(validation.value?.warnings)
    ? validation.value!.warnings
    : [],
)

// Sort tier verdicts so "fits" rows render first (positive feedback at
// the top), and put "won't fit" in the same color the comparison table
// uses for skipped (gray) rather than red — these aren't errors.
const tierRows = computed(() => {
  const verdicts = preflight.value?.tier_verdicts ?? {}
  return Object.entries(verdicts)
    .map(([name, v]) => ({ name, cap: v.cap, fits: v.fits }))
    .sort((a, b) => Number(b.fits) - Number(a.fits))
})

const anyTierMisses = computed(
  () => tierRows.value.some((r) => !r.fits),
)
const allTiersMiss = computed(
  () => tierRows.value.length > 0 && tierRows.value.every((r) => !r.fits),
)

async function approve() {
  error.value = null
  approving.value = true
  try {
    await solve.approveJob(props.job.id)
  } catch (e: any) {
    error.value = e?.response?.data?.error || e?.message || 'approval failed'
  } finally {
    approving.value = false
  }
}

async function cancel() {
  error.value = null
  cancelling.value = true
  try {
    await solve.deleteJob(props.job.id)
  } catch (e: any) {
    error.value = e?.response?.data?.error || e?.message || 'cancel failed'
  } finally {
    cancelling.value = false
  }
}

function fmt(obj: any): string {
  if (obj === null || obj === undefined) return ''
  if (typeof obj === 'string') return obj
  return JSON.stringify(obj, null, 2)
}
</script>

<template>
  <v-card class="pa-5">
    <div class="d-flex align-center mb-2">
      <v-icon icon="mdi-clipboard-check-outline" class="mr-2" color="primary" />
      <v-card-title class="pa-0 text-h6 flex-grow-1">
        Review formulation before solving
      </v-card-title>
      <v-chip size="small" color="primary" variant="tonal" prepend-icon="mdi-pause-circle">
        awaiting approval
      </v-chip>
    </div>
    <v-card-subtitle class="pa-0 mb-3 text-body-2">
      The LLM has formulated your problem. Review the size + per-tier
      verdict below; click Approve to continue to the solvers, or
      Cancel to discard.
    </v-card-subtitle>

    <!-- Structured translation — visible representation of what the
         classifier extracted from the user's prose. Only shown when the
         hardcoded route fired (or, for the LLM-fallback route, when the
         classifier at least identified a family). Rendered as a
         monospace textbox so the user can compare their input to what
         the pipeline actually acted on. -->
    <v-card
      v-if="route && (routeIsHardcoded || route.family)"
      variant="tonal"
      class="pa-3 mb-3"
      :color="routeIsHardcoded ? 'success' : 'info'"
    >
      <div class="d-flex align-center mb-2">
        <v-icon
          :icon="routeIsHardcoded ? 'mdi-check-decagram' : 'mdi-translate'"
          size="small"
          class="mr-2"
        />
        <div class="text-subtitle-2 flex-grow-1">
          Problem statement (translated) — {{ familyLabel }}
        </div>
        <v-chip
          size="x-small"
          :color="routeIsHardcoded ? 'success' : 'grey'"
          variant="flat"
        >
          {{ routeIsHardcoded ? 'hardcoded route' : 'LLM CQM (fallback)' }}
        </v-chip>
      </div>
      <div class="text-caption text-medium-emphasis mb-2">
        <span v-if="routeIsHardcoded">
          The classifier recognized this as
          <strong>{{ familyLabel }}</strong>
          (confidence {{ (route.confidence * 100).toFixed(0) }}%) and the
          deterministic formulator emitted the CQM below — coefficients
          are exact by construction.
        </span>
        <span v-else>
          The classifier saw hints of {{ familyLabel }} but at
          {{ (route.confidence * 100).toFixed(0) }}% confidence, below
          the routing threshold. Falling back to the LLM.
        </span>
      </div>
      <pre class="translation-text">{{ translatedStatement }}</pre>
      <div
        v-if="route.reasoning"
        class="text-caption text-medium-emphasis mt-2"
      >
        <v-icon icon="mdi-comment-quote-outline" size="x-small" class="mr-1" />
        <em>{{ route.reasoning }}</em>
      </div>
    </v-card>

    <!-- Preflight summary card -->
    <v-card variant="tonal" class="pa-3 mb-3">
      <div class="d-flex flex-wrap ga-3 align-center">
        <div>
          <div class="text-caption text-medium-emphasis">CQM variables</div>
          <div class="text-h6">{{ preflight?.cqm_variables ?? '?' }}</div>
        </div>
        <v-divider vertical />
        <div>
          <div class="text-caption text-medium-emphasis">CQM constraints</div>
          <div class="text-h6">{{ preflight?.cqm_constraints ?? '?' }}</div>
        </div>
        <v-divider vertical />
        <div>
          <div class="text-caption text-medium-emphasis">Lowered qubits</div>
          <div class="text-h6">
            <span v-if="preflight?.lowered_qubits != null">
              {{ preflight.lowered_qubits }}
            </span>
            <span v-else class="text-medium-emphasis">—</span>
          </div>
        </div>
      </div>
      <div v-if="preflight?.lowering_error" class="text-caption text-warning mt-2">
        Could not estimate lowered qubits: {{ preflight.lowering_error }}
      </div>
    </v-card>

    <!-- Validation outcome — advisory, doesn't block approval -->
    <v-alert
      v-if="validation && !validationPassed"
      type="warning"
      variant="tonal"
      density="compact"
      class="mb-3"
      icon="mdi-alert-octagon-outline"
    >
      <div class="text-subtitle-2 mb-1">Validation flagged a problem</div>
      <ul class="pl-4 text-body-2">
        <li v-for="(w, i) in validationWarnings" :key="i">{{ w }}</li>
      </ul>
      <div class="text-caption mt-2">
        This usually means the LLM's CQM doesn't match the textbook
        optimum for the problem you described — typically the
        encoding pattern is right but a coefficient is off. You can
        still approve and run solvers (their answer may be close
        even on a slightly-wrong CQM), or cancel and re-submit to
        let the LLM try again.
      </div>
    </v-alert>
    <v-alert
      v-else-if="validation && validationPassed"
      type="success"
      variant="tonal"
      density="compact"
      class="mb-3"
      icon="mdi-check-circle-outline"
    >
      Validation passed — the brute-force oracle agrees with the
      expected optimum (or the CQM is too large to brute-force, in
      which case Layer A was skipped).
    </v-alert>

    <!-- QPU footprint — only rendered when a real Origin backend is
         selected. Simulator submissions don't decrement quota so
         showing an estimate there would be noise. -->
    <v-card
      v-if="qpuFootprint"
      variant="tonal"
      class="pa-3 mb-3"
      color="warning"
    >
      <div class="d-flex align-center mb-2">
        <v-icon icon="mdi-timer-sand" size="small" class="mr-2" />
        <div class="text-subtitle-2 flex-grow-1">
          QPU footprint — <code>{{ qpuFootprint.backend }}</code>
        </div>
        <v-btn
          size="x-small"
          variant="text"
          href="https://qcloud.originqc.com.cn/en/computerServies"
          target="_blank"
          rel="noopener"
          append-icon="mdi-open-in-new"
        >
          View balance
        </v-btn>
      </div>
      <div class="d-flex flex-wrap ga-4">
        <div>
          <div class="text-caption text-medium-emphasis">
            Estimated compute time
          </div>
          <div class="footprint-value">
            {{ qpuFootprint.compute_seconds_low }}–{{ qpuFootprint.compute_seconds_high }} s
          </div>
        </div>
        <v-divider vertical />
        <div>
          <div class="text-caption text-medium-emphasis">Shots</div>
          <div class="footprint-value">
            {{ qpuFootprint.assumptions.shots }}
          </div>
        </div>
        <v-divider vertical />
        <div>
          <div class="text-caption text-medium-emphasis">QAOA layers</div>
          <div class="footprint-value">
            {{ qpuFootprint.assumptions.qaoa_layers }}
          </div>
        </div>
        <v-divider vertical />
        <div>
          <div class="text-caption text-medium-emphasis">
            Circuit depth (approx)
          </div>
          <div class="footprint-value">
            ~{{ qpuFootprint.assumptions.estimated_gates }} gates
          </div>
        </div>
      </div>
      <div class="text-caption mt-3 footprint-note">
        <v-icon icon="mdi-information-outline" size="x-small" class="mr-1" />
        <strong>Only the compute-time seconds are billed.</strong>
        {{ qpuFootprint.note }} Queue wait affects when you see
        results, not how much they cost.
      </div>
    </v-card>

    <!-- Per-tier verdict -->
    <div v-if="tierRows.length" class="mb-3">
      <div class="text-subtitle-2 mb-1">QAOA tier compatibility</div>
      <v-list density="compact" class="bg-transparent pa-0">
        <v-list-item
          v-for="row in tierRows"
          :key="row.name"
          class="px-2 py-0"
        >
          <template #prepend>
            <v-icon
              :icon="row.fits ? 'mdi-check-circle' : 'mdi-debug-step-over'"
              :color="row.fits ? 'success' : 'grey'"
              size="small"
            />
          </template>
          <v-list-item-title>
            <code class="solver-name">{{ row.name }}</code>
            <span class="text-caption text-medium-emphasis ml-2">
              cap {{ row.cap }} qubits
            </span>
          </v-list-item-title>
          <v-list-item-subtitle>
            <span v-if="row.fits" class="text-success">
              fits — will run
            </span>
            <span v-else class="text-medium-emphasis">
              would skip ({{ preflight?.lowered_qubits ?? '?' }} > {{ row.cap }})
            </span>
          </v-list-item-subtitle>
        </v-list-item>
      </v-list>
    </div>

    <!-- Warning when LLM encoding wastes qubits -->
    <v-alert
      v-if="allTiersMiss"
      type="warning"
      variant="tonal"
      density="compact"
      class="mb-3"
      icon="mdi-alert-circle-outline"
    >
      <strong>Every QAOA tier will be skipped.</strong>
      The LLM's encoding produced more qubits than any QAOA backend
      supports. This usually means the formulation uses indicator
      variables or aux constraints that aren't strictly needed.
      The classical / quantum-inspired tiers will still run if you
      approve.
    </v-alert>
    <v-alert
      v-else-if="anyTierMisses"
      type="info"
      variant="tonal"
      density="compact"
      class="mb-3"
      icon="mdi-information-outline"
    >
      Some QAOA tiers won't fit this instance and will be skipped.
      The remaining tiers (including classical / quantum-inspired)
      will still run.
    </v-alert>

    <!-- Expandable CQM JSON for the curious -->
    <v-btn
      variant="text"
      size="small"
      :prepend-icon="showCqm ? 'mdi-chevron-down' : 'mdi-chevron-right'"
      class="mb-2"
      @click="showCqm = !showCqm"
    >
      {{ showCqm ? 'Hide' : 'Show' }} CQM JSON
    </v-btn>
    <pre v-if="showCqm" class="cqm-text">{{ fmt(job.cqm_json) || '<no CQM>' }}</pre>

    <v-alert v-if="error" type="error" variant="tonal" density="compact" class="mt-2">
      {{ error }}
    </v-alert>

    <v-card-actions class="px-0 pt-3">
      <v-btn
        variant="outlined"
        color="error"
        :loading="cancelling"
        :disabled="approving"
        prepend-icon="mdi-close"
        @click="cancel"
      >
        Cancel
      </v-btn>
      <v-spacer />
      <v-btn
        color="primary"
        :loading="approving"
        :disabled="cancelling"
        prepend-icon="mdi-rocket-launch"
        @click="approve"
      >
        Approve & solve
      </v-btn>
    </v-card-actions>
  </v-card>
</template>

<style scoped>
.cqm-text {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 0.75rem;
  font-family: 'JetBrains Mono', Consolas, ui-monospace, monospace;
  font-size: 0.78rem;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 360px;
  overflow: auto;
}
.solver-name {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85rem;
}
.footprint-value {
  font-size: 1.05rem;
  font-weight: 600;
}
.footprint-note {
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  padding-top: 0.6rem;
  line-height: 1.4;
}
.translation-text {
  background: rgba(0, 0, 0, 0.28);
  border-radius: 6px;
  padding: 0.65rem 0.8rem;
  font-family: 'JetBrains Mono', Consolas, ui-monospace, monospace;
  font-size: 0.82rem;
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.4;
}
</style>
