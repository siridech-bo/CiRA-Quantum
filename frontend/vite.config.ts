import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [vue(), vuetify({ autoImport: true })],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    // Spec default is 3011 but ports 3010/3011 are occupied on this host;
    // 3070 picked as the next clear band. CORS_ORIGINS in the backend
    // config must include this port for the cookie round-trip to work.
    port: 3070,
    // Proxy /api/* to the Flask backend on :5009. `changeOrigin: true` is
    // essential — without it Flask sees the wrong Host header and CORS
    // logic gets confused on cookies.
    proxy: {
      '/api': {
        target: 'http://localhost:5009',
        changeOrigin: true,
      },
    },
  },
})
