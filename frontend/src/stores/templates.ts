/**
 * Templates store — backs the template gallery, the modules library,
 * and the "Try this example" flow.
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

const api = axios.create({ withCredentials: true })

export interface TemplateSummary {
  id: string
  title: string
  category: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  summary: string
  tags: string[]
  estimated_solve_time_seconds: number
  module_id?: string
  module_order?: number
}

export interface TemplateDetail extends TemplateSummary {
  problem_statement: string
  real_world_example: string
  expected_pattern: {
    variables?: string
    objective?: string
    constraints?: string
  }
  expected_optimum: number | null
  expected_solution_summary: string
  learning_notes: string
  min_solver_tier: string
  module?: {
    module_id: string
    lesson_id: string
    order: number
    learning_objectives: string[]
    prerequisites?: string[]
    assessment_hooks?: Record<string, any>
  }
}

export interface CategoryCount {
  name: string
  count: number
}

export const useTemplatesStore = defineStore('templates', () => {
  const list = ref<TemplateSummary[]>([])
  const categories = ref<CategoryCount[]>([])
  const modules = ref<Record<string, TemplateDetail[]>>({})
  const current = ref<TemplateDetail | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function loadList(): Promise<void> {
    loading.value = true
    error.value = null
    try {
      const r = await api.get<{
        templates: TemplateSummary[]
        categories: CategoryCount[]
      }>('/api/templates')
      list.value = r.data.templates
      categories.value = r.data.categories
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || 'failed to load templates'
    } finally {
      loading.value = false
    }
  }

  async function loadDetail(id: string): Promise<TemplateDetail | null> {
    loading.value = true
    error.value = null
    try {
      const r = await api.get<{ template: TemplateDetail }>(`/api/templates/${id}`)
      current.value = r.data.template
      return r.data.template
    } catch (e: any) {
      error.value = e?.response?.status === 404
        ? 'template not found'
        : e?.response?.data?.error || e?.message || 'failed to load template'
      return null
    } finally {
      loading.value = false
    }
  }

  async function loadModules(): Promise<void> {
    loading.value = true
    error.value = null
    try {
      const r = await api.get<{ modules: Record<string, TemplateDetail[]> }>(
        '/api/templates/modules',
      )
      modules.value = r.data.modules
    } catch (e: any) {
      error.value = e?.response?.data?.error || e?.message || 'failed to load modules'
    } finally {
      loading.value = false
    }
  }

  async function solveFromTemplate(
    id: string,
    payload: {
      provider: 'claude' | 'openai' | 'local'
      api_key?: string
      use_stored_key?: boolean
    },
  ): Promise<{ id: string } & Record<string, any>> {
    const r = await api.post<{ success: boolean; job: any }>(
      `/api/solve/from-template/${id}`,
      payload,
    )
    return r.data.job
  }

  return {
    list,
    categories,
    modules,
    current,
    loading,
    error,
    loadList,
    loadDetail,
    loadModules,
    solveFromTemplate,
  }
})
