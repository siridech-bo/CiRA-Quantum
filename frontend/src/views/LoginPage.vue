<script setup lang="ts">
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

const username = ref('')
const password = ref('')
const error = ref<string | null>(null)
const loading = ref(false)

async function submit() {
  error.value = null
  loading.value = true
  try {
    await auth.login(username.value.trim(), password.value)
    const next = (route.query.redirect as string | undefined) || '/'
    router.push(next)
  } catch (e: any) {
    error.value = e?.response?.data?.error || 'Login failed'
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
          <v-card-title class="text-h5">Sign in to CiRA Quantum</v-card-title>
          <v-card-subtitle>Academic quantum-optimization platform</v-card-subtitle>
          <v-form @submit.prevent="submit">
            <v-text-field
              v-model="username"
              label="Username"
              autocomplete="username"
              autofocus
              required
            />
            <v-text-field
              v-model="password"
              label="Password"
              type="password"
              autocomplete="current-password"
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
              Sign in
            </v-btn>
            <div class="mt-4 text-center text-body-2">
              New here?
              <router-link to="/signup">Create an account</router-link>
            </div>
          </v-form>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>
