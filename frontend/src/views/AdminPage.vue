<script setup lang="ts">
/**
 * Phase 7 — Admin read-only views.
 *
 * Three tabs:
 *   - **Overview**: headline counters (user count, jobs by status,
 *     last-24 h activity, top LLM providers, pending cloud jobs)
 *   - **Users**: every registered user + their stored BYOK providers
 *     (provider names only — never plaintext) + total job count
 *   - **Jobs**: paginated cross-user job list with status filter
 *     and drill-into-detail links
 *
 * Gated by ``requiresAdmin`` in the router. Hits the new
 * ``/api/admin/*`` endpoints; anyone non-admin who reaches this
 * URL gets bounced back to ``/`` by the global guard.
 */
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import { useAuthStore } from '@/stores/auth'
import CiraLogo from '@/components/CiraLogo.vue'

const api = axios.create({ withCredentials: true })
const router = useRouter()
const auth = useAuthStore()

const tab = ref<'overview' | 'users' | 'jobs'>('overview')

// ----- Overview --------------------------------------------------------------

interface OverviewPayload {
  users: { total: number; active: number; admins: number }
  jobs: {
    total: number
    by_status: Record<string, number>
    last_24h: number
    top_providers: { provider: string; count: number }[]
  }
  pending_cloud_jobs: number
}

const overview = ref<OverviewPayload | null>(null)
const overviewLoading = ref(false)
const overviewError = ref<string | null>(null)

async function loadOverview() {
  overviewLoading.value = true
  overviewError.value = null
  try {
    const r = await api.get<OverviewPayload>('/api/admin/overview')
    overview.value = r.data
  } catch (e: any) {
    overviewError.value = e?.response?.data?.error || e?.message || 'Failed to load overview'
  } finally {
    overviewLoading.value = false
  }
}

// ----- Users -----------------------------------------------------------------

interface AdminUserRow {
  id: number
  username: string
  email: string | null
  display_name: string | null
  role: string
  is_active: boolean
  created_at: string
  last_login: string | null
  providers_on_file: string[]
  total_jobs: number
}

const users = ref<AdminUserRow[]>([])
const usersLoading = ref(false)
const usersError = ref<string | null>(null)

async function loadUsers() {
  usersLoading.value = true
  usersError.value = null
  try {
    const r = await api.get<{ users: AdminUserRow[]; total: number }>('/api/admin/users')
    users.value = r.data.users
  } catch (e: any) {
    usersError.value = e?.response?.data?.error || e?.message || 'Failed to load users'
  } finally {
    usersLoading.value = false
  }
}

// ----- Jobs ------------------------------------------------------------------

interface AdminJobRow {
  id: string
  user_id: number
  username: string | null
  problem_statement: string
  provider: string
  status: string
  created_at: string
  completed_at: string | null
  solve_time_ms: number | null
  template_id: string | null
}

const jobsPage = ref(1)
const jobsPageSize = ref(30)
const jobsStatusFilter = ref<string>('all')
const jobs = ref<AdminJobRow[]>([])
const jobsTotal = ref(0)
const jobsLoading = ref(false)
const jobsError = ref<string | null>(null)

const jobsTotalPages = computed(() =>
  Math.max(1, Math.ceil(jobsTotal.value / jobsPageSize.value)),
)

async function loadJobs() {
  jobsLoading.value = true
  jobsError.value = null
  try {
    const params: Record<string, string> = {
      page: String(jobsPage.value),
      page_size: String(jobsPageSize.value),
    }
    if (jobsStatusFilter.value !== 'all') {
      params.status = jobsStatusFilter.value
    }
    const r = await api.get<{
      jobs: AdminJobRow[]
      total: number
      page: number
      page_size: number
    }>('/api/admin/jobs', { params })
    jobs.value = r.data.jobs
    jobsTotal.value = r.data.total
  } catch (e: any) {
    jobsError.value = e?.response?.data?.error || e?.message || 'Failed to load jobs'
  } finally {
    jobsLoading.value = false
  }
}

function statusColor(status: string): string {
  if (status === 'complete') return 'success'
  if (status === 'error') return 'error'
  if (status === 'queued') return 'info'
  if (status === 'solving' || status === 'compiling' || status === 'validating' || status === 'formulating') return 'primary'
  return 'default'
}

function fmtDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}
function fmtTime(ms: number | null): string {
  if (ms === null) return '—'
  if (ms < 10) return `${ms.toFixed(1)} ms`
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

onMounted(() => {
  void loadOverview()
})
</script>

<template>
  <v-app-bar color="surface" flat aria-label="Admin app bar">
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/solve')" />
    <div
      class="d-flex align-center logo-link"
      role="button"
      tabindex="0"
      @click="router.push('/')"
      @keydown.enter="router.push('/')"
    >
      <CiraLogo :size="28" />
    </div>
    <v-app-bar-title class="text-medium-emphasis ml-2">
      <span class="font-weight-medium">Admin</span>
    </v-app-bar-title>
    <v-spacer />
    <v-chip
      v-if="auth.user"
      size="x-small"
      color="accent"
      variant="tonal"
      prepend-icon="mdi-shield-account"
    >
      {{ auth.user.display_name || auth.user.username }} (admin)
    </v-chip>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-tabs v-model="tab" color="primary" align-tabs="start" class="mb-4">
        <v-tab value="overview" @click="loadOverview">
          <v-icon icon="mdi-view-dashboard" start /> Overview
        </v-tab>
        <v-tab value="users" @click="loadUsers">
          <v-icon icon="mdi-account-group" start /> Users
        </v-tab>
        <v-tab value="jobs" @click="loadJobs">
          <v-icon icon="mdi-history" start /> Jobs
        </v-tab>
      </v-tabs>

      <v-window v-model="tab">
        <!-- ============ OVERVIEW ============ -->
        <v-window-item value="overview">
          <v-skeleton-loader v-if="overviewLoading && !overview" type="article" />
          <v-alert v-else-if="overviewError" type="error" variant="tonal">
            {{ overviewError }}
          </v-alert>
          <template v-else-if="overview">
            <v-row class="mb-3">
              <v-col cols="12" sm="6" md="3">
                <v-card class="pa-4" variant="tonal" color="primary">
                  <div class="text-overline">Users</div>
                  <div class="text-h4">{{ overview.users.total }}</div>
                  <div class="text-caption text-medium-emphasis">
                    {{ overview.users.active }} active · {{ overview.users.admins }} admin
                  </div>
                </v-card>
              </v-col>
              <v-col cols="12" sm="6" md="3">
                <v-card class="pa-4" variant="tonal" color="success">
                  <div class="text-overline">Total jobs</div>
                  <div class="text-h4">{{ overview.jobs.total }}</div>
                  <div class="text-caption text-medium-emphasis">
                    {{ overview.jobs.last_24h }} in last 24 h
                  </div>
                </v-card>
              </v-col>
              <v-col cols="12" sm="6" md="3">
                <v-card class="pa-4" variant="tonal" color="info">
                  <div class="text-overline">Pending cloud</div>
                  <div class="text-h4">{{ overview.pending_cloud_jobs }}</div>
                  <div class="text-caption text-medium-emphasis">
                    queued at Origin / IBM
                  </div>
                </v-card>
              </v-col>
              <v-col cols="12" sm="6" md="3">
                <v-card class="pa-4" variant="tonal" color="warning">
                  <div class="text-overline">Jobs in flight</div>
                  <div class="text-h4">
                    {{
                      (overview.jobs.by_status.solving || 0)
                      + (overview.jobs.by_status.formulating || 0)
                      + (overview.jobs.by_status.compiling || 0)
                      + (overview.jobs.by_status.validating || 0)
                    }}
                  </div>
                  <div class="text-caption text-medium-emphasis">
                    actively running now
                  </div>
                </v-card>
              </v-col>
            </v-row>

            <v-row>
              <v-col cols="12" md="6">
                <v-card class="pa-4">
                  <div class="text-subtitle-1 mb-2">Jobs by status</div>
                  <div
                    v-for="(count, status) in overview.jobs.by_status"
                    :key="status"
                    class="d-flex align-center mb-1"
                  >
                    <v-chip
                      size="small"
                      :color="statusColor(String(status))"
                      variant="tonal"
                      class="mr-3"
                    >
                      {{ status }}
                    </v-chip>
                    <strong>{{ count }}</strong>
                  </div>
                </v-card>
              </v-col>
              <v-col cols="12" md="6">
                <v-card class="pa-4">
                  <div class="text-subtitle-1 mb-2">Top LLM providers</div>
                  <div
                    v-for="p in overview.jobs.top_providers"
                    :key="p.provider"
                    class="d-flex align-center mb-1"
                  >
                    <code class="mr-3">{{ p.provider }}</code>
                    <strong>{{ p.count }}</strong>
                  </div>
                </v-card>
              </v-col>
            </v-row>
          </template>
        </v-window-item>

        <!-- ============ USERS ============ -->
        <v-window-item value="users">
          <v-skeleton-loader v-if="usersLoading && !users.length" type="table" />
          <v-alert v-else-if="usersError" type="error" variant="tonal">
            {{ usersError }}
          </v-alert>
          <v-card v-else>
            <v-table density="compact">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Username</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Active</th>
                  <th>Created</th>
                  <th>Last login</th>
                  <th>Providers on file</th>
                  <th class="text-right">Jobs</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="u in users" :key="u.id">
                  <td>{{ u.id }}</td>
                  <td><code>{{ u.username }}</code></td>
                  <td>{{ u.email || '—' }}</td>
                  <td>
                    <v-chip
                      size="x-small"
                      :color="u.role === 'admin' ? 'accent' : 'default'"
                      variant="tonal"
                    >
                      {{ u.role }}
                    </v-chip>
                  </td>
                  <td>
                    <v-icon
                      :icon="u.is_active ? 'mdi-check-circle' : 'mdi-pause-circle'"
                      :color="u.is_active ? 'success' : 'grey'"
                      size="small"
                    />
                  </td>
                  <td>{{ fmtDate(u.created_at) }}</td>
                  <td>{{ fmtDate(u.last_login) }}</td>
                  <td>
                    <v-chip
                      v-for="p in u.providers_on_file"
                      :key="p"
                      size="x-small"
                      variant="tonal"
                      class="mr-1"
                    >
                      {{ p }}
                    </v-chip>
                    <span v-if="!u.providers_on_file.length" class="text-medium-emphasis text-caption">none</span>
                  </td>
                  <td class="text-right">{{ u.total_jobs }}</td>
                </tr>
              </tbody>
            </v-table>
          </v-card>
        </v-window-item>

        <!-- ============ JOBS ============ -->
        <v-window-item value="jobs">
          <div class="d-flex align-center mb-3 ga-2">
            <v-select
              v-model="jobsStatusFilter"
              :items="['all', 'queued', 'formulating', 'compiling', 'validating', 'solving', 'complete', 'error']"
              label="Status filter"
              density="compact"
              hide-details
              style="max-width: 220px"
              @update:model-value="() => { jobsPage = 1; loadJobs() }"
            />
            <v-btn
              variant="outlined"
              size="small"
              prepend-icon="mdi-refresh"
              :loading="jobsLoading"
              @click="loadJobs"
            >
              Refresh
            </v-btn>
            <v-spacer />
            <span class="text-caption text-medium-emphasis">
              {{ jobsTotal }} total · page {{ jobsPage }} of {{ jobsTotalPages }}
            </span>
          </div>

          <v-skeleton-loader v-if="jobsLoading && !jobs.length" type="table" />
          <v-alert v-else-if="jobsError" type="error" variant="tonal">
            {{ jobsError }}
          </v-alert>
          <v-card v-else>
            <v-table density="compact">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>User</th>
                  <th>Provider</th>
                  <th>Problem</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th class="text-right">Solve time</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="j in jobs"
                  :key="j.id"
                  class="job-row"
                  @click="router.push(`/jobs/${j.id}`)"
                >
                  <td><code>{{ j.id.slice(0, 8) }}</code></td>
                  <td><code>{{ j.username || `(user ${j.user_id})` }}</code></td>
                  <td>{{ j.provider }}</td>
                  <td class="problem-cell">{{ j.problem_statement }}</td>
                  <td>
                    <v-chip
                      size="x-small"
                      :color="statusColor(j.status)"
                      variant="tonal"
                    >
                      {{ j.status }}
                    </v-chip>
                  </td>
                  <td>{{ fmtDate(j.created_at) }}</td>
                  <td class="text-right">{{ fmtTime(j.solve_time_ms) }}</td>
                </tr>
              </tbody>
            </v-table>
          </v-card>
          <div class="d-flex justify-center mt-3">
            <v-pagination
              v-model="jobsPage"
              :length="jobsTotalPages"
              density="compact"
              @update:model-value="loadJobs"
            />
          </div>
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
.logo-link:hover { opacity: 0.8; }
.logo-link:focus-visible {
  outline: 2px solid currentColor;
  outline-offset: 4px;
  border-radius: 4px;
}
.problem-cell {
  max-width: 360px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.job-row {
  cursor: pointer;
}
.job-row:hover {
  background: rgba(255, 255, 255, 0.03);
}
</style>
