<script setup lang="ts">
/**
 * AuthDialog — single component that handles both login and signup.
 *
 * Lives on the main landing page (``/``) as the only authentication
 * surface on the platform. Other module landing pages (``/qml``,
 * ``/qldpc``) intentionally have no login controls — users always sign
 * in from ``/`` and then navigate to whatever module they want. This
 * matches the platform's single-user architecture where one CiRA
 * Quantum account has access to every module through the same session.
 *
 * ``mode`` prop switches the form between login and signup. Both flows
 * write to the same Pinia auth store; on success we emit
 * ``authenticated`` so the parent can close the dialog and refresh
 * anything that depends on the auth state.
 */
import { computed, ref, watch } from 'vue'
import { useAuthStore } from '@/stores/auth'

const props = defineProps<{
  modelValue: boolean
  mode: 'login' | 'signup'
}>()
const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'update:mode', v: 'login' | 'signup'): void
  (e: 'authenticated'): void
}>()

const auth = useAuthStore()

const username = ref('')
const password = ref('')
const displayName = ref('')
const email = ref('')
const error = ref<string | null>(null)
const loading = ref(false)

// Reset the form whenever the dialog opens/closes so a previous
// failure message isn't sitting there next time.
watch(() => props.modelValue, (open) => {
  if (!open) {
    error.value = null
    loading.value = false
    username.value = ''
    password.value = ''
    displayName.value = ''
    email.value = ''
  }
})

const title = computed(
  () => (props.mode === 'login' ? 'Sign in' : 'Create an account'),
)
const submitLabel = computed(
  () => (props.mode === 'login' ? 'Sign in' : 'Create account'),
)
const canSubmit = computed(() => {
  if (loading.value) return false
  if (!username.value.trim() || !password.value) return false
  return true
})

function close() {
  emit('update:modelValue', false)
}

function switchTo(mode: 'login' | 'signup') {
  error.value = null
  emit('update:mode', mode)
}

async function submit() {
  error.value = null
  loading.value = true
  try {
    if (props.mode === 'login') {
      await auth.login(username.value.trim(), password.value)
    } else {
      await auth.signup(
        username.value.trim(),
        password.value,
        displayName.value.trim() || undefined,
        email.value.trim() || undefined,
      )
    }
    emit('authenticated')
    close()
  } catch (e: any) {
    error.value = e?.response?.data?.error
      || (props.mode === 'login' ? 'Login failed' : 'Signup failed')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <v-dialog
    :model-value="modelValue"
    @update:model-value="(v: boolean) => emit('update:modelValue', v)"
    max-width="440"
  >
    <v-card class="pa-2">
      <v-card-title class="d-flex align-center pa-4 pb-2">
        <v-icon
          :icon="mode === 'login' ? 'mdi-login' : 'mdi-account-plus'"
          class="mr-2"
        />
        <span class="text-h6 flex-grow-1">{{ title }}</span>
        <v-btn
          icon="mdi-close"
          variant="text"
          size="small"
          @click="close"
          aria-label="Close"
        />
      </v-card-title>

      <v-card-text class="pa-4 pt-2">
        <v-form @submit.prevent="submit">
          <v-text-field
            v-model="username"
            label="Username"
            autocomplete="username"
            autofocus
            density="comfortable"
            required
          />
          <v-text-field
            v-model="password"
            label="Password"
            type="password"
            :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
            density="comfortable"
            required
          />
          <template v-if="mode === 'signup'">
            <v-text-field
              v-model="displayName"
              label="Display name (optional)"
              density="comfortable"
            />
            <v-text-field
              v-model="email"
              label="Email (optional)"
              type="email"
              density="comfortable"
            />
          </template>

          <v-alert
            v-if="error"
            type="error"
            variant="tonal"
            density="compact"
            class="mt-1 mb-3"
          >
            {{ error }}
          </v-alert>

          <v-btn
            type="submit"
            color="primary"
            :loading="loading"
            :disabled="!canSubmit"
            block
            class="mt-2"
          >
            {{ submitLabel }}
          </v-btn>

          <div class="mt-4 text-center text-body-2">
            <template v-if="mode === 'login'">
              New here?
              <a href="#" @click.prevent="switchTo('signup')">Create an account</a>
            </template>
            <template v-else>
              Already have an account?
              <a href="#" @click.prevent="switchTo('login')">Sign in</a>
            </template>
          </div>
        </v-form>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>
