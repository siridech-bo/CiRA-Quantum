<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useSolveStore } from '@/stores/solve'
import { useTemplatesStore, type TemplateDetail, type TemplateSummary } from '@/stores/templates'
import TemplateCard from '@/components/TemplateCard.vue'
import TemplateDetailModal from '@/components/TemplateDetailModal.vue'

const auth = useAuthStore()
const solve = useSolveStore()
const templates = useTemplatesStore()
const router = useRouter()

const view = ref<'gallery' | 'modules'>('gallery')
const categoryFilter = ref<string>('all')
const difficultyFilter = ref<'all' | 'beginner' | 'intermediate' | 'advanced'>('all')
const search = ref('')

const modalOpen = ref(false)
const modalTemplate = ref<TemplateDetail | null>(null)

const filtered = computed<TemplateSummary[]>(() => {
  const needle = search.value.trim().toLowerCase()
  return templates.list.filter((t) => {
    if (categoryFilter.value !== 'all' && t.category !== categoryFilter.value) return false
    if (difficultyFilter.value !== 'all' && t.difficulty !== difficultyFilter.value) return false
    if (needle) {
      const hay = `${t.title} ${t.summary} ${(t.tags || []).join(' ')}`.toLowerCase()
      if (!hay.includes(needle)) return false
    }
    return true
  })
})

async function openTemplate(t: TemplateSummary) {
  modalTemplate.value = null
  modalOpen.value = true
  const full = await templates.loadDetail(t.id)
  modalTemplate.value = full
}

async function logout() {
  solve.reset()
  await auth.logout()
  router.push('/login')
}

onMounted(async () => {
  await Promise.all([
    templates.loadList(),
    templates.loadModules(),
    solve.loadKeys(),
  ])
})
</script>

<template>
  <v-app-bar color="surface" flat>
    <v-btn icon="mdi-arrow-left" variant="text" @click="router.push('/')" />
    <v-app-bar-title class="text-primary font-weight-bold">
      CiRA Quantum
    </v-app-bar-title>
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
      <div class="d-flex align-center mb-2">
        <div>
          <div class="text-h5">Example problems</div>
          <div class="text-body-2 text-medium-emphasis">
            Curated optimization problems you can try with one click.
          </div>
        </div>
        <v-spacer />
        <v-btn-toggle
          v-model="view"
          mandatory
          density="compact"
          variant="outlined"
          divided
        >
          <v-btn value="gallery" prepend-icon="mdi-view-grid">Gallery</v-btn>
          <v-btn value="modules" prepend-icon="mdi-school">Modules</v-btn>
        </v-btn-toggle>
      </div>

      <!-- Gallery view -->
      <template v-if="view === 'gallery'">
        <v-row class="my-2" align="center">
          <v-col cols="12" sm="6">
            <v-text-field
              v-model="search"
              prepend-inner-icon="mdi-magnify"
              label="Search by title or tag"
              density="comfortable"
              hide-details
              clearable
            />
          </v-col>
          <v-col cols="12" sm="6" class="d-flex ga-2 flex-wrap align-center">
            <v-chip-group v-model="categoryFilter" selected-class="bg-primary text-white" mandatory>
              <v-chip value="all" size="small">All</v-chip>
              <v-chip
                v-for="c in templates.categories"
                :key="c.name"
                :value="c.name"
                size="small"
              >
                {{ c.name }} ({{ c.count }})
              </v-chip>
            </v-chip-group>
          </v-col>
        </v-row>

        <v-chip-group
          v-model="difficultyFilter"
          selected-class="bg-primary text-white"
          mandatory
          class="mb-3"
        >
          <v-chip value="all" size="small">All levels</v-chip>
          <v-chip value="beginner" size="small">Beginner</v-chip>
          <v-chip value="intermediate" size="small">Intermediate</v-chip>
          <v-chip value="advanced" size="small">Advanced</v-chip>
        </v-chip-group>

        <v-row>
          <v-col
            v-for="t in filtered"
            :key="t.id"
            cols="12"
            sm="6"
            md="4"
            lg="3"
          >
            <TemplateCard :template="t" @open="openTemplate" />
          </v-col>
        </v-row>
        <v-card v-if="!filtered.length" class="pa-6 text-center" variant="tonal">
          <v-icon icon="mdi-magnify-close" size="large" class="mb-2" />
          <div>No examples match your filters. Try clearing them.</div>
        </v-card>
      </template>

      <!-- Modules view -->
      <template v-else>
        <div v-if="!Object.keys(templates.modules).length" class="text-medium-emphasis py-6 text-center">
          No structured modules yet.
        </div>
        <div
          v-for="(lessons, modId) in templates.modules"
          :key="modId"
          class="mb-6"
        >
          <div class="text-h6 mb-1">
            <v-icon icon="mdi-school" start color="accent" />
            {{ modId }}
          </div>
          <div class="text-caption text-medium-emphasis mb-3">
            {{ lessons.length }} lessons · prerequisites shown per card
          </div>
          <v-timeline align="start" density="compact" side="end">
            <v-timeline-item
              v-for="(lesson, idx) in lessons"
              :key="lesson.id"
              dot-color="accent"
              :icon="`mdi-numeric-${Math.min(9, idx + 1)}-circle`"
              size="small"
            >
              <v-card
                class="lesson-card pa-3"
                hover
                @click="openTemplate(lesson)"
              >
                <div class="d-flex align-center mb-1">
                  <span class="text-subtitle-1 font-weight-medium flex-grow-1">
                    {{ lesson.title }}
                  </span>
                  <v-chip size="x-small" variant="tonal">{{ lesson.difficulty }}</v-chip>
                </div>
                <div class="text-body-2 text-medium-emphasis mb-1">
                  {{ lesson.summary }}
                </div>
                <ul class="text-caption pl-4 my-1">
                  <li
                    v-for="obj in (lesson.module && lesson.module.learning_objectives) || []"
                    :key="obj"
                  >
                    {{ obj }}
                  </li>
                </ul>
                <div
                  v-if="lesson.module && (lesson.module.prerequisites || []).length"
                  class="text-caption text-medium-emphasis mt-1"
                >
                  Requires:
                  <v-chip
                    v-for="p in lesson.module.prerequisites"
                    :key="p"
                    size="x-small"
                    variant="outlined"
                    class="ml-1"
                  >
                    {{ p }}
                  </v-chip>
                </div>
              </v-card>
            </v-timeline-item>
          </v-timeline>
        </div>
      </template>
    </v-container>
  </v-main>

  <TemplateDetailModal v-model="modalOpen" :template="modalTemplate" />
</template>

<style scoped>
.lesson-card {
  cursor: pointer;
  transition: transform 0.12s ease-in-out;
}
.lesson-card:hover {
  transform: translateY(-1px);
}
</style>
