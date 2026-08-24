import { computed } from 'vue'
import { useTheme } from 'vuetify'

/**
 * Shared day/night toggle. Vuetify's theme is global, so setting it here
 * themes the whole app; the choice persists in localStorage ('cira-theme')
 * and is re-applied at boot in main.ts. Used by the landing page and the
 * simulator wrapper so the platform and the embedded QuantumSim feel
 * consistent as you move between them.
 */
export function useThemeToggle() {
  const theme = useTheme()
  const isDark = computed(() => theme.global.current.value.dark)

  function setTheme(name: 'light' | 'dark') {
    theme.global.name.value = name
    try {
      localStorage.setItem('cira-theme', name)
    } catch {
      /* localStorage may be unavailable (private mode) — in-memory still holds */
    }
  }
  function toggleTheme() {
    setTheme(isDark.value ? 'light' : 'dark')
  }

  return { isDark, toggleTheme, setTheme }
}
