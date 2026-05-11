import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

// All Flask routes are cookie-session-authed; the frontend ships every
// request with credentials so the session cookie comes along.
const api = axios.create({ withCredentials: true })

export interface User {
  id: number
  username: string
  display_name: string
  role: 'user' | 'admin'
}

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const bootstrapped = ref(false)

  async function checkAuth(): Promise<void> {
    try {
      const r = await api.get<{ user: User }>('/api/auth/me')
      user.value = r.data.user
    } catch {
      user.value = null
    } finally {
      bootstrapped.value = true
    }
  }

  async function login(username: string, password: string): Promise<void> {
    const r = await api.post<{ user: User }>('/api/auth/login', {
      username,
      password,
    })
    user.value = r.data.user
  }

  async function signup(
    username: string,
    password: string,
    display_name?: string,
    email?: string,
  ): Promise<void> {
    const r = await api.post<{ user: User }>('/api/auth/signup', {
      username,
      password,
      display_name,
      email,
    })
    user.value = r.data.user
  }

  async function logout(): Promise<void> {
    try {
      await api.post('/api/auth/logout')
    } finally {
      user.value = null
    }
  }

  return { user, bootstrapped, checkAuth, login, signup, logout }
})
