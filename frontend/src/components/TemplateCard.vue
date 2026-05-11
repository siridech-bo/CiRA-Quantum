<script setup lang="ts">
import type { TemplateSummary } from '@/stores/templates'

defineProps<{ template: TemplateSummary }>()
defineEmits<{ (e: 'open', t: TemplateSummary): void }>()

const DIFF_COLOR: Record<string, string> = {
  beginner: 'success',
  intermediate: 'warning',
  advanced: 'error',
}
</script>

<template>
  <v-card
    class="template-card"
    @click="$emit('open', template)"
    elevation="2"
    hover
  >
    <v-card-text class="pa-4">
      <div class="d-flex align-center mb-2 ga-2">
        <v-chip
          size="x-small"
          :color="DIFF_COLOR[template.difficulty] || 'grey'"
          variant="tonal"
        >
          {{ template.difficulty }}
        </v-chip>
        <v-chip size="x-small" variant="tonal" color="info">
          {{ template.category }}
        </v-chip>
        <v-spacer />
        <v-chip
          v-if="template.module_id"
          size="x-small"
          variant="tonal"
          color="accent"
          prepend-icon="mdi-school"
        >
          Module
        </v-chip>
      </div>

      <div class="text-subtitle-1 font-weight-medium mb-1">
        {{ template.title }}
      </div>
      <div class="text-body-2 text-medium-emphasis summary">
        {{ template.summary }}
      </div>

      <div class="mt-3 d-flex flex-wrap ga-1">
        <v-chip
          v-for="tag in (template.tags || []).slice(0, 3)"
          :key="tag"
          size="x-small"
          variant="outlined"
        >
          {{ tag }}
        </v-chip>
        <v-chip
          v-if="(template.tags || []).length > 3"
          size="x-small"
          variant="outlined"
        >
          +{{ template.tags.length - 3 }} more
        </v-chip>
      </div>
    </v-card-text>
  </v-card>
</template>

<style scoped>
.template-card {
  height: 100%;
  cursor: pointer;
  transition: transform 0.12s ease-in-out;
}
.template-card:hover {
  transform: translateY(-2px);
}
.summary {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
