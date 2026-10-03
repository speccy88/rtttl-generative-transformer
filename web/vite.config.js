import { defineConfig } from 'vite';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const runtimeRevision = createHash('sha256').update(readFileSync(new URL('./package-lock.json', import.meta.url))).digest('hex').slice(0, 16);

export default defineConfig({
  base: process.env.VITE_BASE_PATH || '/rtttl-generative-transformer/',
  define: { __RUNTIME_REVISION__: JSON.stringify(runtimeRevision) },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 1800 },
  server: { port: 5173, strictPort: true },
  preview: { port: 4173, strictPort: true },
});
