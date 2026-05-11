<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()

const username = ref('')
const password = ref('')
const display_name = ref('')
const email = ref('')
const error = ref<string | null>(null)
const loading = ref(false)

async function submit() {
  error.value = null
  loading.value = true
  try {
    await auth.signup(
      username.value.trim(),
      password.value,
      display_name.value.trim() || undefined,
      email.value.trim() || undefined,
    )
    router.push('/')
  } catch (e: any) {
    error.value = e?.response?.data?.error || 'Signup failed'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <v-container fluid class="fill-height">
    <v-row align="center" justify="center">
      <v-col cols="12" sm="8" md="5" lg="4">
        <v-card class="pa-6">
          <v-card-title class="text-h5">Create an account</v-card-title>
          <v-card-subtitle>3-32 chars username; password 8+ chars</v-card-subtitle>
          <v-form @submit.prevent="submit">
            <v-text-field v-model="username" label="Username" autocomplete="username" required />
            <v-text-field
              v-model="display_name"
              label="Display name (optional)"
              autocomplete="name"
            />
            <v-text-field
              v-model="email"
              label="Email (optional)"
              type="email"
              autocomplete="email"
            />
            <v-text-field
              v-model="password"
              label="Password"
              type="password"
              autocomplete="new-password"
              required
            />
            <v-alert v-if="error" type="error" variant="tonal" class="mb-3">
              {{ error }}
            </v-alert>
            <v-btn
              type="submit"
              color="primary"
              :loading="loading"
              :disabled="!username || !password"
              block
            >
              Create account
            </v-btn>
            <div class="mt-4 text-center text-body-2">
              Already have an account?
              <router-link to="/login">Sign in</router-link>
            </div>
          </v-form>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>
