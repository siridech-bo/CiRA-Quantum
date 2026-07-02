<script setup lang="ts">
/**
 * ChangePasswordForm — rotates the signed-in user's password via
 * ``POST /api/auth/change-password``. Backend enforces the minimum
 * length; we surface whatever error the backend returns verbatim.
 *
 * Lives inside SettingsPage as one of its tabs. Not exposed as a
 * standalone route (nothing to bookmark; users navigate to /settings).
 */
import { computed, ref } from 'vue'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()

const current = ref('')
const next = ref('')
const confirm = ref('')
const submitting = ref(false)
const error = ref<string | null>(null)
const justSucceeded = ref(false)

const mismatch = computed(
  () => confirm.value.length > 0 && next.value !== confirm.value,
)
const canSubmit = computed(() => {
  if (submitting.value) return false
  if (!current.value || !next.value || !confirm.value) return false
  if (mismatch.value) return false
  if (next.value === current.value) return false
  return true
})

async function submit() {
  error.value = null
  justSucceeded.value = false
  submitting.value = true
  try {
    await auth.changePassword(current.value, next.value)
    justSucceeded.value = true
    current.value = ''
    next.value = ''
    confirm.value = ''
  } catch (e: any) {
    error.value = e?.response?.data?.error
      || e?.message
      || 'Failed to change password'
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <v-card class="pa-5">
    <v-card-title class="text-h6 pa-0 mb-1">
      <v-icon icon="mdi-lock-reset" start />
      Change password
    </v-card-title>
    <v-card-subtitle class="pa-0 mb-4">
      Rotate the password on this account. Signs you out of no other
      devices — session cookies remain valid until you log out
      explicitly.
    </v-card-subtitle>

    <v-form @submit.prevent="submit" style="max-width: 480px">
      <v-text-field
        v-model="current"
        label="Current password"
        type="password"
        autocomplete="current-password"
        density="comfortable"
        required
      />
      <v-text-field
        v-model="next"
        label="New password"
        type="password"
        autocomplete="new-password"
        density="comfortable"
        required
      />
      <v-text-field
        v-model="confirm"
        label="Confirm new password"
        type="password"
        autocomplete="new-password"
        density="comfortable"
        :error="mismatch"
        :error-messages="mismatch ? 'Does not match the new password' : ''"
        required
      />

      <v-alert
        v-if="error"
        type="error"
        variant="tonal"
        density="compact"
        class="mt-2 mb-3"
      >
        {{ error }}
      </v-alert>
      <v-alert
        v-if="justSucceeded"
        type="success"
        variant="tonal"
        density="compact"
        class="mt-2 mb-3"
      >
        Password updated.
      </v-alert>

      <v-btn
        type="submit"
        color="primary"
        :loading="submitting"
        :disabled="!canSubmit"
      >
        Update password
      </v-btn>
    </v-form>
  </v-card>
</template>
