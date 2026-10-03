import { createRequire } from 'node:module';
import { mkdir, rm, copyFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

// Both worker bundles use matching local binaries. GitHub Pages needs no
// cross-origin isolation headers because each WASM runtime uses one thread.
const require = createRequire(import.meta.url);
const transformerEntry = require.resolve('@huggingface/transformers');
const transformerRequire = createRequire(transformerEntry);
for (const [name, resolver] of [['ort', require], ['transformers', transformerRequire]]) {
  const entry = resolver.resolve('onnxruntime-web');
  const source = dirname(entry);
  const target = resolve('public/runtime', name);
  await rm(target, { recursive: true, force: true });
  await mkdir(target, { recursive: true });
  // Both locked WebGPU entrypoints use asyncify for WebGPU and WASM execution.
  // Do not copy unused WebGL/all-backend bundles into the public distribution.
  const files = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];
  for (const file of files) await copyFile(resolve(source, file), resolve(target, file));
  console.log(`Prepared ${files.length} ${name} runtime assets`);
}
