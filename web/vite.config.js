import { defineConfig } from 'vite';

export default defineConfig({
  base: process.env.VITE_BASE_PATH || '/rtttl-generative-transformer/',
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 1800 },
  server: { port: 5173, strictPort: true },
  preview: { port: 4173, strictPort: true },
});
