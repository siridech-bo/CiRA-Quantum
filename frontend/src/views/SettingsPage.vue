<script setup lang="ts">
/**
 * SettingsPage — the platform's central account-settings surface.
 *
 * Auth requires-auth: guard bounces unauthenticated visitors back to
 * ``/``. Two tabs today: API Keys (BYOK provider credentials) and
 * Change Password. Extending later with Profile / Preferences / etc.
 * is a matter of adding another v-tab and its content view.
 *
 * Reached from the user menu on any authenticated app bar; nothing
 * links here anonymously.
 */
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import ApiKeyManager from '@/components/ApiKeyManager.vue'
import ChangePasswordForm from '@/components/ChangePasswordForm.vue'
import CiraLogo from '@/components/CiraLogo.vue'

const router = useRouter()
const auth = useAuthStore()

const tab = ref<'keys' | 'password'>('keys')

async function logout() {
  await auth.logout()
  router.push('/')
}
</script>

<template>
  <v-app-bar color="surface" flat aria-label="Settings app bar">
    <div
      class="d-flex align-center logo-link ml-3"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="32" />
    </div>
    <v-spacer />
    <span class="text-body-2 mr-4" v-if="auth.user">
      Signed in as <strong>{{ auth.user.display_name }}</strong>
      <v-chip
        v-if="auth.user.role === 'admin'"
        size="x-small"
        color="accent"
        class="ml-2"
      >
        admin
      </v-chip>
    </span>
    <v-btn variant="outlined" @click="logout">Log out</v-btn>
  </v-app-bar>

  <v-main>
    <v-container>
      <div class="d-flex align-center mb-3">
        <v-icon icon="mdi-cog-outline" size="large" class="mr-2" />
        <div>
          <div class="text-h5">Settings</div>
          <div class="text-body-2 text-medium-emphasis">
            Provider credentials and account controls that apply across
            every CiRA Quantum module.
          </div>
        </div>
      </div>

      <v-tabs v-model="tab" color="primary" density="compact" class="mb-4">
        <v-tab value="keys">
          <v-icon icon="mdi-key" start /> API Keys
        </v-tab>
        <v-tab value="password">
          <v-icon icon="mdi-lock-reset" start /> Change Password
        </v-tab>
      </v-tabs>

      <v-window v-model="tab">
        <v-window-item value="keys">
          <ApiKeyManager />
        </v-window-item>
        <v-window-item value="password">
          <ChangePasswordForm />
        </v-window-item>
      </v-window>
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
.logo-link:focus-visible {
  outline: 2px solid currentColor;
  outline-offset: 4px;
  border-radius: 4px;
}
</style>
