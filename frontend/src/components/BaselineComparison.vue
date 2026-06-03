<script setup lang="ts">
/**
 * BaselineComparison — side-by-side scoreboard of the VQC vs four
 * classical baselines on the same train/test split.
 *
 * The pedagogical point of this widget: a student should walk away
 * from each training run with an honest answer to "did the quantum
 * model actually help here, or did a 5-line sklearn baseline match it?"
 *
 * Layout: one row per model (VQC pinned at the top), columns for
 * test accuracy, train accuracy, wall time, family, with a horizontal
 * bar showing test accuracy so visual comparison is instant. The
 * winner gets a small trophy chip.
 */
import { computed } from 'vue'
import type { QmlBaseline, QmlMetrics } from '@/stores/qml'

const props = defineProps<{
  vqc: QmlMetrics
  baselines: QmlBaseline[]
}>()

interface Row {
  key: string
  title: string
  family: string
  family_color: string
  family_icon: string
  test_accuracy: number
  train_accuracy: number
  train_time_ms: number
  is_quantum: boolean
  notes: string
}

const FAMILY_STYLE: Record<string, { color: string; icon: string }> = {
  quantum:  { color: 'accent',   icon: 'mdi-atom-variant' },
  linear:   { color: 'grey',     icon: 'mdi-vector-line' },
  kernel:   { color: 'info',     icon: 'mdi-blur-radial' },
  ensemble: { color: 'success',  icon: 'mdi-forest' },
  neural:   { color: 'primary',  icon: 'mdi-graph-outline' },
}

const rows = computed<Row[]>(() => {
  const m = props.vqc
  const out: Row[] = []
  // VQC row pinned at the top.
  out.push({
    key: 'vqc',
    title: `Variational Quantum Classifier (${m.n_qubits} qubits)`,
    family: 'quantum',
    family_color: FAMILY_STYLE.quantum.color,
    family_icon: FAMILY_STYLE.quantum.icon,
    test_accuracy: m.final_test_accuracy,
    train_accuracy: m.final_train_accuracy,
    train_time_ms: m.train_time_ms,
    is_quantum: true,
    notes: 'PennyLane statevector simulator. Trainable params: '
      + `${m.circuit_info?.n_trainable_params ?? '?'}.`,
  })
  for (const b of props.baselines) {
    const style = FAMILY_STYLE[b.family] || FAMILY_STYLE.linear
    out.push({
      key: b.name,
      title: b.title,
      family: b.family,
      family_color: style.color,
      family_icon: style.icon,
      test_accuracy: b.test_accuracy,
      train_accuracy: b.train_accuracy,
      train_time_ms: b.train_time_ms,
      is_quantum: false,
      notes: b.notes,
    })
  }
  return out
})

const winnerKey = computed(() => {
  // Highest test accuracy wins. Ties broken by lower train time.
  let best: Row | null = null
  for (const r of rows.value) {
    if (best === null
        || r.test_accuracy > best.test_accuracy
        || (r.test_accuracy === best.test_accuracy
            && r.train_time_ms < best.train_time_ms)) {
      best = r
    }
  }
  return best?.key
})

function fmtPct(p: number): string {
  return (p * 100).toFixed(1) + '%'
}
function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(1)} s`
}
function barWidth(acc: number): string {
  // 50% accuracy = no signal on a balanced binary task → render at 0.
  // The bar starts at 50% and fills the remaining 50% up to 100%.
  const clipped = Math.max(0.5, Math.min(1, acc))
  return `${((clipped - 0.5) / 0.5) * 100}%`
}

// Educational verdict line — what the comparison says in one sentence.
const verdict = computed(() => {
  const winner = rows.value.find((r) => r.key === winnerKey.value)
  const vqcRow = rows.value.find((r) => r.is_quantum)
  if (!winner || !vqcRow) return null
  const margin = winner.test_accuracy - vqcRow.test_accuracy
  if (winner.is_quantum) {
    return {
      tone: 'success',
      icon: 'mdi-medal',
      title: 'The VQC matched or beat every classical baseline here.',
      detail: `Test accuracy ${fmtPct(vqcRow.test_accuracy)}. The classical `
        + `baselines that came closest were trained on the exact same `
        + `${vqcRow.train_accuracy.toFixed(0)}-point split — so this is `
        + `a like-for-like comparison.`,
    }
  }
  if (margin < 0.03) {
    return {
      tone: 'info',
      icon: 'mdi-scale-balance',
      title: `${winner.title} edged out the VQC by ${fmtPct(margin)}.`,
      detail: 'On these toy datasets a quantum model rarely beats a tuned '
        + 'classical baseline outright. The point isn\'t to win — it\'s to '
        + 'see how the two approaches sit alongside each other on the same data.',
    }
  }
  return {
    tone: 'warning',
    icon: 'mdi-information-outline',
    title: `${winner.title} beat the VQC by ${fmtPct(margin)} on this run.`,
    detail: 'This is the honest classical baseline. Try more layers / more '
      + 'epochs / a different encoding — or accept that this dataset doesn\'t '
      + 'have the kind of structure where a small VQC helps.',
  }
})
</script>

<template>
  <v-card variant="outlined" class="pa-4">
    <div class="d-flex align-center mb-3">
      <v-icon icon="mdi-podium" class="mr-2" />
      <div class="text-subtitle-1 flex-grow-1">
        Quantum vs classical — same train/test split
      </div>
      <v-chip size="small" variant="tonal">
        {{ rows.length }} models
      </v-chip>
    </div>

    <v-alert
      v-if="verdict"
      :type="verdict.tone as any"
      variant="tonal"
      density="compact"
      :icon="verdict.icon"
      class="mb-3"
    >
      <div class="text-subtitle-2 font-weight-medium">{{ verdict.title }}</div>
      <div class="text-body-2">{{ verdict.detail }}</div>
    </v-alert>

    <div class="comparison-table">
      <div class="ct-row ct-head">
        <div class="ct-model">Model</div>
        <div class="ct-fam">Family</div>
        <div class="ct-acc">Test accuracy</div>
        <div class="ct-train">Train acc</div>
        <div class="ct-time">Wall time</div>
      </div>
      <div
        v-for="r in rows"
        :key="r.key"
        class="ct-row"
        :class="{
          'ct-winner': r.key === winnerKey,
          'ct-quantum': r.is_quantum,
        }"
      >
        <div class="ct-model">
          <v-icon
            v-if="r.key === winnerKey"
            icon="mdi-trophy"
            color="warning"
            size="small"
            class="mr-1"
          />
          <span class="font-weight-medium">{{ r.title }}</span>
          <div class="text-caption text-medium-emphasis mt-1">
            {{ r.notes }}
          </div>
        </div>
        <div class="ct-fam">
          <v-chip
            size="x-small"
            :color="r.family_color"
            :prepend-icon="r.family_icon"
            variant="flat"
          >
            {{ r.family }}
          </v-chip>
        </div>
        <div class="ct-acc">
          <div class="acc-bar">
            <div
              class="acc-fill"
              :class="{ quantum: r.is_quantum, winner: r.key === winnerKey }"
              :style="{ width: barWidth(r.test_accuracy) }"
            />
            <div class="acc-label">{{ fmtPct(r.test_accuracy) }}</div>
          </div>
        </div>
        <div class="ct-train text-body-2 text-medium-emphasis">
          {{ fmtPct(r.train_accuracy) }}
        </div>
        <div class="ct-time text-body-2 text-medium-emphasis">
          {{ fmtMs(r.train_time_ms) }}
        </div>
      </div>
    </div>

    <div class="text-caption text-medium-emphasis mt-3">
      All five models trained on the exact same standard-scaled split
      <span v-if="vqc.pca_applied">(PCA pre-projected for the VQC)</span>.
      Bars start at <code>50%</code> (chance for a balanced binary task)
      and fill to <code>100%</code>. The trophy marks the highest test
      accuracy, ties broken by faster wall time.
    </div>
  </v-card>
</template>

<style scoped>
.comparison-table {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ct-row {
  display: grid;
  grid-template-columns: 2.4fr 0.9fr 2fr 0.7fr 0.7fr;
  gap: 12px;
  align-items: center;
  padding: 10px 12px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.ct-head {
  background: transparent;
  border: none;
  padding-bottom: 0;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: rgba(255, 255, 255, 0.5);
}
.ct-quantum {
  background: rgba(168, 85, 247, 0.06);
  border-color: rgba(168, 85, 247, 0.25);
}
.ct-winner {
  background: rgba(245, 158, 11, 0.08);
  border-color: rgba(245, 158, 11, 0.4);
}
.ct-quantum.ct-winner {
  background: linear-gradient(
    90deg,
    rgba(168, 85, 247, 0.10) 0%,
    rgba(245, 158, 11, 0.10) 100%
  );
}
.acc-bar {
  position: relative;
  height: 22px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  overflow: hidden;
}
.acc-fill {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: #3b82f6;
  transition: width 0.3s ease-out;
}
.acc-fill.quantum {
  background: #a855f7;
}
.acc-fill.winner {
  background: linear-gradient(90deg, #3b82f6 0%, #f59e0b 100%);
}
.acc-fill.quantum.winner {
  background: linear-gradient(90deg, #a855f7 0%, #f59e0b 100%);
}
.acc-label {
  position: absolute;
  top: 0;
  right: 8px;
  height: 100%;
  display: flex;
  align-items: center;
  font-size: 0.8rem;
  font-weight: 600;
  color: white;
  text-shadow: 0 0 4px rgba(0, 0, 0, 0.6);
}
@media (max-width: 700px) {
  .ct-row {
    grid-template-columns: 1.6fr 1fr 1.4fr;
  }
  .ct-train, .ct-time, .ct-head .ct-train, .ct-head .ct-time {
    display: none;
  }
}
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
</style>
