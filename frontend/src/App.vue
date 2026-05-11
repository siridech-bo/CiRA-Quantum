<script setup lang="ts">
import { onBeforeMount } from 'vue'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()

// Re-hydrate the session on app boot. The Flask cookie is HttpOnly + bound
// to /api so JS can't read it directly; instead we hit /api/auth/me which
// returns 401 cleanly when no session is present.
onBeforeMount(async () => {
  await auth.checkAuth()
})
</script>

<template>
  <v-app>
    <router-view />
  </v-app>
</template>
