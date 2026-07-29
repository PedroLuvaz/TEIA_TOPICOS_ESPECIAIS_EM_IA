import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    // Em desenvolvimento, o Vite encaminha /api para o FastAPI na porta 8000.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    // O chunk `graficos` (recharts) passa de 500 kB, mas so e baixado quando
    // uma pagina com grafico abre — nao pesa no carregamento inicial.
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        // Separa as bibliotecas pesadas para que o carregamento inicial nao
        // pague por graficos e LaTeX antes de precisar deles.
        manualChunks: {
          react: ['react', 'react-dom'],
          graficos: ['recharts'],
          latex: ['katex'],
          animacao: ['motion'],
        },
      },
    },
  },
})
