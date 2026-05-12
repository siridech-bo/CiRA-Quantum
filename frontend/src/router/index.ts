import {
  createRouter,
  createWebHistory,
  type NavigationGuardWithThis,
  type RouteRecordRaw,
} from 'vue-router'
import LoginPage from '@/views/LoginPage.vue'
import SignupPage from '@/views/SignupPage.vue'
import MainApp from '@/views/MainApp.vue'
import JobDetailPage from '@/views/JobDetailPage.vue'
import TemplateGalleryPage from '@/views/TemplateGalleryPage.vue'
import BenchmarkDashboardPage from '@/views/BenchmarkDashboardPage.vue'
import BenchmarkSuitePage from '@/views/BenchmarkSuitePage.vue'
import BenchmarkSolverPage from '@/views/BenchmarkSolverPage.vue'
import BenchmarkInstancePage from '@/views/BenchmarkInstancePage.vue'
import BenchmarkFindingsPage from '@/views/BenchmarkFindingsPage.vue'
import { useAuthStore } from '@/stores/auth'

declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    requiresGuest?: boolean
    requiresAdmin?: boolean
  }
}

const routes: RouteRecordRaw[] = [
  { path: '/login', component: LoginPage, meta: { requiresGuest: true } },
  { path: '/signup', component: SignupPage, meta: { requiresGuest: true } },
  { path: '/', component: MainApp, meta: { requiresAuth: true } },
  { path: '/jobs/:id', component: JobDetailPage, meta: { requiresAuth: true } },
  { path: '/templates', component: TemplateGalleryPage, meta: { requiresAuth: true } },
  // Phase 5C — public Benchmark dashboard (no auth required).
  { path: '/benchmarks', component: BenchmarkDashboardPage },
  // Findings: aggregated experiment results. Specific route MUST come
  // before the splat-style suite route, otherwise vue-router treats
  // "findings" as a suite_id.
  { path: '/benchmarks/findings', component: BenchmarkFindingsPage },
  { path: '/benchmarks/suites/:pathMatch(.*)', component: BenchmarkSuitePage },
  { path: '/benchmarks/solvers/:solverName', component: BenchmarkSolverPage },
  { path: '/benchmarks/instances/:pathMatch(.*)', component: BenchmarkInstancePage },
  // Phase 7 (/admin) bolts on here.
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

const guard: NavigationGuardWithThis<undefined> = async (to) => {
  const auth = useAuthStore()

  // The first navigation may race the App.vue onBeforeMount checkAuth() call;
  // wait for it here too so guards don't see a stale unauth'd state.
  if (!auth.bootstrapped) {
    await auth.checkAuth()
  }

  if (to.meta.requiresAuth && !auth.user) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  if (to.meta.requiresGuest && auth.user) {
    return { path: '/' }
  }
  if (to.meta.requiresAdmin && auth.user?.role !== 'admin') {
    return { path: '/' }
  }
  return true
}

router.beforeEach(guard)

export default router
