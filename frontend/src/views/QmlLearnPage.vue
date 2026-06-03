<script setup lang="ts">
/**
 * QmlLearnPage — primer for the QML pipeline.
 *
 * Structure follows the actual VQC flow a student will see on the
 * training pages, so the concepts here map 1:1 to the gates +
 * measurements they'll encounter:
 *
 *   1. What's a qubit? + Bloch sphere
 *   2. How does AngleEmbedding encode data?  ← interactive demo
 *   3. What do entangler layers buy us?
 *   4. What does measurement actually produce?
 *   5. How does the model learn (parameter shift)?
 *
 * Public route — no auth required. The goal is to lower the barrier
 * before a student spends simulator time.
 */
import { useRouter } from 'vue-router'
import CiraLogo from '@/components/CiraLogo.vue'
import BlochSphereDemo from '@/components/BlochSphereDemo.vue'

const router = useRouter()
</script>

<template>
  <v-app-bar color="surface" flat aria-label="CiRA QML learn">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
      <span class="text-subtitle-1 ml-3 text-medium-emphasis">— QML primer</span>
    </div>
    <v-spacer />
    <v-btn variant="text" @click="router.push('/qml')">Datasets</v-btn>
  </v-app-bar>

  <v-main>
    <v-container style="max-width: 1100px">
      <!-- Hero -->
      <div class="pt-8 pb-2">
        <div class="text-overline text-medium-emphasis">QML primer</div>
        <div class="text-h4 font-weight-bold mb-3">
          The five things a VQC is doing
        </div>
        <div class="text-body-1 text-medium-emphasis" style="max-width: 760px">
          A Variational Quantum Classifier is a small parametric circuit
          that turns a data point into a probability. This page walks
          through the five steps, in order, with the same gates and
          measurements you'll see on the training pages.
        </div>
        <div class="d-flex ga-2 mt-4 flex-wrap">
          <v-btn
            color="primary"
            variant="flat"
            prepend-icon="mdi-school"
            @click="router.push('/qml')"
          >
            Browse the dataset gallery
          </v-btn>
          <v-btn
            variant="outlined"
            prepend-icon="mdi-flask-outline"
            @click="router.push('/qml/datasets/moons')"
          >
            Study the Moons dataset
          </v-btn>
        </div>
      </div>

      <v-divider class="my-6" />

      <!-- Step 1: qubit -->
      <section class="mb-8">
        <div class="text-overline text-primary">step 1</div>
        <div class="text-h5 mb-2">
          A qubit is a unit vector with two complex components
        </div>
        <div class="text-body-1 text-medium-emphasis mb-3">
          Where a classical bit is a single number in <code>{0, 1}</code>,
          a qubit is a state <code>|ψ⟩ = α|0⟩ + β|1⟩</code> with
          <code>|α|² + |β|² = 1</code>. Two pieces of complex information,
          one constraint, so two real degrees of freedom — a point on
          the surface of a unit sphere called the <strong>Bloch sphere</strong>.
        </div>
        <div class="text-body-2 text-medium-emphasis">
          <strong>Why it matters:</strong> the input feature you hand to the
          VQC sets a rotation angle. The qubit's state moves on the sphere,
          and the geometry of that movement is what makes a quantum kernel
          different from a classical one.
        </div>
      </section>

      <!-- Step 2: AngleEmbedding — interactive -->
      <section class="mb-8">
        <div class="text-overline text-primary">step 2</div>
        <div class="text-h5 mb-2">
          Encoding: one feature → one rotation
        </div>
        <div class="text-body-1 text-medium-emphasis mb-4">
          The first column of the VQC circuit is
          <code>AngleEmbedding</code>: for each input feature
          <code>x<sub>i</sub></code> apply an <code>RX(x<sub>i</sub>)</code>
          rotation to qubit <code>i</code>. That's it — there's no
          "uploading the data," no quantum memory. Each data point
          becomes a configuration of qubit rotations.
        </div>
        <BlochSphereDemo class="mb-3" />
      </section>

      <!-- Step 3: entangler -->
      <section class="mb-8">
        <div class="text-overline text-primary">step 3</div>
        <div class="text-h5 mb-2">
          Entangler layers: trainable rotations + CNOTs
        </div>
        <div class="text-body-1 text-medium-emphasis mb-3">
          After encoding, each layer applies <code>RY(θ<sub>l,q</sub>)</code>
          on every qubit followed by a ring of CNOTs. The
          <code>θ</code>'s are the trainable parameters — they're the
          only thing Adam adjusts.
        </div>
        <v-card variant="tonal" color="info" class="pa-3 mb-3">
          <div class="text-subtitle-2 mb-1">
            <v-icon icon="mdi-lightbulb-on-outline" size="small" class="mr-1" />
            Why CNOTs and not just rotations?
          </div>
          <div class="text-body-2">
            A circuit of single-qubit rotations alone can't express
            correlations between features — every qubit evolves
            independently. CNOTs link them: after a CNOT, the joint
            state is entangled, and gradients of the loss with respect
            to one qubit's <code>θ</code> depend on every other qubit's
            state. That's the only place where "quantum-ness" enters
            the model.
          </div>
        </v-card>
        <div class="text-body-2 text-medium-emphasis">
          Stacking <code>L</code> layers gives the model
          <code>L × n_qubits</code> trainable rotation parameters.
          For a 2-qubit, 2-layer VQC: 4 trainable angles + 1 bias = 5
          scalars total. That's a deliberately tiny model.
        </div>
      </section>

      <!-- Step 4: measurement -->
      <section class="mb-8">
        <div class="text-overline text-primary">step 4</div>
        <div class="text-h5 mb-2">
          Measurement: collapsing a quantum state into a real number
        </div>
        <div class="text-body-1 text-medium-emphasis mb-3">
          The last column measures the <code>PauliZ</code> expectation on
          qubit 0. Concretely, <code>⟨Z⟩ = P(measure 0) − P(measure 1)</code>,
          a number in <code>[−1, 1]</code>. On a real QPU this comes from
          averaging shot outcomes; on our statevector simulator we
          compute it exactly.
        </div>
        <div class="text-body-2 text-medium-emphasis mb-2">
          We then add a trainable bias and squash through a sigmoid:
        </div>
        <pre class="formula">P(class = 1)  =  σ(⟨Z⟩ + b)</pre>
        <div class="text-body-2 text-medium-emphasis">
          That's the entire classifier — a 5-parameter model that lives
          inside one qubit's expectation value.
        </div>
      </section>

      <!-- Step 5: learning -->
      <section class="mb-8">
        <div class="text-overline text-primary">step 5</div>
        <div class="text-h5 mb-2">
          Learning: gradient descent on a quantum loss
        </div>
        <div class="text-body-1 text-medium-emphasis mb-3">
          The loss is plain binary cross-entropy on
          <code>P(class = 1)</code>. The gradient with respect to each
          <code>θ</code> is computed by the <strong>parameter-shift
          rule</strong>: evaluate the circuit at
          <code>θ + π/2</code> and <code>θ − π/2</code>, take half the
          difference. Two extra circuit runs per parameter per step.
        </div>
        <v-card variant="tonal" color="warning" class="pa-3 mb-3">
          <div class="text-subtitle-2 mb-1">
            <v-icon icon="mdi-alert-circle-outline" size="small" class="mr-1" />
            Why this matters for the cost of training
          </div>
          <div class="text-body-2">
            On the simulator the gradients are exact and cheap. On a
            <strong>real QPU</strong>, each parameter-shift evaluation
            is a cloud submission with shot noise — a 5-parameter
            circuit at 1000 shots costs 10000 shots per training step.
            That's why we train locally first and only run the trained
            parameters on a QPU for the final evaluation.
          </div>
        </v-card>
      </section>

      <v-divider class="my-6" />

      <!-- Closing -->
      <section class="pb-12">
        <div class="text-h6 mb-2">Ready to try one?</div>
        <div class="text-body-2 text-medium-emphasis mb-4">
          Start with <strong>Two Moons</strong> — the canonical
          non-linear toy. Two qubits, two trainable layers, 30 epochs;
          you'll see the loss fall and the decision boundary curve in
          under a minute on the statevector simulator.
        </div>
        <v-btn
          color="primary"
          variant="flat"
          size="large"
          prepend-icon="mdi-rocket-launch"
          @click="router.push('/qml/datasets/moons')"
        >
          Study Two Moons → Train
        </v-btn>
      </section>
    </v-container>
  </v-main>
</template>

<style scoped>
.logo-link {
  cursor: pointer;
  transition: opacity 0.15s ease-in-out;
}
.logo-link:hover {
  opacity: 0.8;
}
code {
  background: rgba(255, 255, 255, 0.08);
  padding: 0 4px;
  border-radius: 3px;
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.85em;
}
.formula {
  font-family: 'Cascadia Code', 'Consolas', monospace;
  font-size: 1.05rem;
  background: rgba(255, 255, 255, 0.05);
  padding: 12px 16px;
  border-radius: 4px;
  border-left: 3px solid #22c55e;
  margin: 8px 0 12px 0;
}
</style>
