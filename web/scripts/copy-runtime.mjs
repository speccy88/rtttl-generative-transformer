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
  // Asyncify supports modern WebGPU/CPU execution. Older Safari without WebGPU
  // needs the plain CPU build, matching Transformers.js' runtime selection.
  // Each pair must come from its own worker's locked ORT version.
  const files = ['', '.asyncify'].flatMap(suffix => ['mjs', 'wasm']
    .map(extension => `ort-wasm-simd-threaded${suffix}.${extension}`));
  for (const file of files) await copyFile(resolve(source, file), resolve(target, file));
  console.log(`Prepared ${files.length} ${name} runtime assets`);
}
