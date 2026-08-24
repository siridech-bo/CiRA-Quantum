import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@mdi/font/css/materialdesignicons.css'

// Dark theme palette from the v2 spec — "Quantum purple" primary,
// "Annealing cyan" accent. Keep these stable; Phase 5+ pages assume them.
// A matching light theme powers the app-wide day/night toggle, aligned with
// the embedded QuantumSim visualizer. The toggle lives on the landing page
// and the simulator wrapper (see useThemeToggle), applies globally, and
// persists in localStorage ('cira-theme'); we read that here so the first
// paint already matches the saved preference. Default: dark.
const initialTheme = (() => {
  try {
    const saved = localStorage.getItem('cira-theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* localStorage may be unavailable (private mode) — fall through */
  }
  return 'dark'
})()

const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: initialTheme,
    themes: {
      dark: {
        dark: true,
        colors: {
          primary: '#7C4DFF',
          secondary: '#424242',
          accent: '#00E5FF',
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FB8C00',
          background: '#121212',
          surface: '#1E1E1E',
        },
      },
      light: {
        dark: false,
        colors: {
          primary: '#7C4DFF',
          secondary: '#5F6368',
          accent: '#00B8D4',
          error: '#D32F2F',
          info: '#1976D2',
          success: '#2E7D32',
          warning: '#EF6C00',
          background: '#F5F5F7',
          surface: '#FFFFFF',
        },
      },
    },
  },
})

const pinia = createPinia()
const app = createApp(App)

app.use(pinia)
app.use(router)
app.use(vuetify)
app.mount('#app')
