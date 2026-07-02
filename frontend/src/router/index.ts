import {
  createRouter,
  createWebHistory,
  type NavigationGuardWithThis,
  type RouteRecordRaw,
} from 'vue-router'
import MainApp from '@/views/MainApp.vue'
import LandingPage from '@/views/LandingPage.vue'
import JobDetailPage from '@/views/JobDetailPage.vue'
import TemplateGalleryPage from '@/views/TemplateGalleryPage.vue'
import BenchmarkDashboardPage from '@/views/BenchmarkDashboardPage.vue'
import BenchmarkSuitePage from '@/views/BenchmarkSuitePage.vue'
import BenchmarkSolverPage from '@/views/BenchmarkSolverPage.vue'
import BenchmarkInstancePage from '@/views/BenchmarkInstancePage.vue'
import BenchmarkFindingsPage from '@/views/BenchmarkFindingsPage.vue'
import Quantum101Page from '@/views/Quantum101Page.vue'
import PlaygroundPage from '@/views/PlaygroundPage.vue'
import AdminPage from '@/views/AdminPage.vue'
import QmlLandingPage from '@/views/QmlLandingPage.vue'
import QmlDatasetDetailPage from '@/views/QmlDatasetDetailPage.vue'
import QmlJobDetailPage from '@/views/QmlJobDetailPage.vue'
import QmlLearnPage from '@/views/QmlLearnPage.vue'
import QmlBenchmarkDashboardPage from '@/views/QmlBenchmarkDashboardPage.vue'
import QmlBenchmarkRecordPage from '@/views/QmlBenchmarkRecordPage.vue'
import QldpcLandingPage from '@/views/QldpcLandingPage.vue'
import QldpcLearnPage from '@/views/QldpcLearnPage.vue'
import QldpcCodeFamilyDetailPage from '@/views/QldpcCodeFamilyDetailPage.vue'
import SettingsPage from '@/views/SettingsPage.vue'
import { useAuthStore } from '@/stores/auth'

declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    requiresGuest?: boolean
    requiresAdmin?: boolean
  }
}

const routes: RouteRecordRaw[] = [
  // Single-login policy — auth lives only on ``/``. These legacy paths
  // stay in the routing table so old bookmarks and external links keep
  // working, but they redirect to the landing page with a query flag
  // that auto-opens the appropriate AuthDialog mode.
  { path: '/login', redirect: { path: '/', query: { auth: 'login' } } },
  { path: '/signup', redirect: { path: '/', query: { auth: 'signup' } } },
  // Public landing page — Phase 5D. Doubles as the entry point for
  // unauthenticated visitors.
  { path: '/', component: LandingPage },
  // The solve app moved here in Phase 5D so the landing page can own /.
  { path: '/solve', component: MainApp, meta: { requiresAuth: true } },
  { path: '/jobs/:id', component: JobDetailPage, meta: { requiresAuth: true } },
  // Central account/settings page — API keys + change password. The
  // API keys tab was previously inside MainApp; centralising it here
  // matches the platform's single-user model where credentials apply
  // across every module (Optimization, QML, qLDPC).
  { path: '/settings', component: SettingsPage, meta: { requiresAuth: true } },
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
  // Phase 10B — interactive QAOA explainer. Public (no auth needed):
  // the demo is self-contained and the goal is education, not solving.
  { path: '/learn/quantum', component: Quantum101Page },
  // Circuit Playground — self-hosted Quirk-E build under public/quirk/.
  // Public so the sandbox is reachable from the Learn menu without forcing
  // a login, like /learn/quantum and /qml/learn.
  { path: '/learn/playground', component: PlaygroundPage },
  // QML-1 — sister application's landing page. Public so the dataset
  // gallery is browsable without auth, like the benchmark dashboard.
  { path: '/qml', component: QmlLandingPage },
  // QML-4 — primer page (Bloch sphere demo + VQC walkthrough). Public
  // for the same reason as Quantum 101 on the optimization side.
  { path: '/qml/learn', component: QmlLearnPage },
  // QML-4 — dataset "study before training" surface. Public so a student
  // can read the dataset + see the circuit + understand the baselines
  // before deciding to log in and spend simulator time.
  { path: '/qml/datasets/:id', component: QmlDatasetDetailPage },
  // QML-2 — live training job detail (loss curve, confusion matrix).
  // requiresAuth because the SSE stream is per-user.
  { path: '/qml/jobs/:id', component: QmlJobDetailPage, meta: { requiresAuth: true } },
  // QML-7 — public benchmark archive dashboard + per-record detail.
  // Public so the scoreboard is browsable without auth, mirroring the
  // optimization-side benchmark dashboard.
  { path: '/qml/benchmarks', component: QmlBenchmarkDashboardPage },
  { path: '/qml/benchmarks/:id', component: QmlBenchmarkRecordPage },
  // qLDPC Sprint 0 — code-family gallery + primer + family detail.
  // All public for the same reason QML's gallery + primer are public:
  // the pitched research customers should be able to evaluate the
  // module surface without an account.
  { path: '/qldpc', component: QldpcLandingPage },
  { path: '/qldpc/learn', component: QldpcLearnPage },
  { path: '/qldpc/codes/:id', component: QldpcCodeFamilyDetailPage },
  // Phase 7 — Admin read-only views. Operator visibility into users,
  // jobs, and BYOK provider distribution. requiresAdmin pushes
  // non-admins back to /.
  { path: '/admin', component: AdminPage, meta: { requiresAuth: true, requiresAdmin: true } },
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
    // Single-login policy — every protected route bounces back to
    // ``/``, where the landing page's AuthDialog opens automatically
    // (see LandingPage.vue's watcher on the ``auth`` query param).
    // ``redirect`` remembers where the user was headed so we can
    // continue there after they authenticate.
    return { path: '/', query: { auth: 'login', redirect: to.fullPath } }
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
