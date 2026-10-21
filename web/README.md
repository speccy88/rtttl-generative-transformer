# RTTTL Studio

**[Open the complete studio on GitHub Pages](https://speccy88.github.io/rtttl-generative-transformer/)**

Convert supported audio files to RTTTL, select a melody, edit notes, generate
variations or repair suggestions, compose new music, and export from one webpage.
All inference runs on the visitor's device. No local server, Python installation,
account or inference API is required.

## Browser workflow

**Convert audio** accepts local MP3, WAV, FLAC, M4A, OGG, AAC, Opus and WebM files
when the browser supports their codec. Files are limited to 32 MiB, source length to 10 minutes and
selected excerpts to 0.25–30 seconds. Short clips are recommended for decode memory
and responsiveness. Use the waveform or numeric fields to select an excerpt, listen
to it, enter known BPM if available, and choose a lead preference/rhythm grid.

The bundled Spotify Basic Pitch model runs in a dedicated single-thread WASM worker.
The browser compares several melody interpretations from the same inference. Lead
preference changes decoding; it is not vocal/instrument stem separation. Candidate
percentages describe agreement with model evidence, not recognition accuracy.

**Select and edit** notes directly in the piano roll. Shift-select a phrase, or use
numeric selection boundaries. Manual edits and model proposals create new versions;
the original remains available. The latest 64 scores persist in this browser.

**Vary selection** changes selected pitches while locking rhythm, rests, tempo,
event count and unselected notes. **Repair suggestions** compares bounded alternatives
against the original using forward model likelihood across the selection and up to
eight following events. It may retain the original. This is approximate causal
candidate search, not BERT-trained masked reconstruction or verified audio correction.
Selections are limited to 16 events and edit batches to four alternatives.

**Compose** retains the existing style, key, tempo and sampling controls. **Continue
selected tune** feeds the score's real token prefix to the transformer and appends
new events; its length control counts additional events. Imported 64th notes can
play, but model operations explicitly reject unsupported 64th-note rhythms.

**Export** RTTTL text, WAV or MP3 from the same selected score. Source audio remains
in browser memory for the session and is never posted to a backend. Export important
scores before clearing browser data.

See [the feature specification](../docs/RTTTL_STUDIO.md) for behavior, acceptance
criteria, architecture and the distinction between variations, repair and continuation.
The Python [mp3_to_rtttl](https://github.com/speccy88/mp3_to_rtttl) project remains an
independent desktop/CLI tool and decoding reference, not a runtime dependency.

## Development only

Use Node.js 22.12 or newer on the Node 22 release line, and npm. These commands run from `web/`:

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/rtttl-generative-transformer/`. The dev/build scripts copy the matching ONNX runtime assets from the locked npm packages into `public/runtime/`; those generated assets are ignored by Git. Use HTTPS or localhost so the browser can expose WebGPU.

```sh
npm test
npm run build
npm run preview
```

For production UI checks, install Playwright's browser once with
`npx playwright install chromium firefox webkit`, then run `npm run test:e2e` after building.
The mandatory browser inference tests use the actual bundled Basic Pitch model
in Chromium, Firefox and WebKit. An MP3-to-variation-to-repair check downloads the
pinned generator files, verifies their hashes, then runs them in a real browser
worker. Separate UI and cache fixtures use mocks or block external downloads.
To explicitly test the real melody model and your available GPU, use
`PLAYWRIGHT_REAL_MODEL=1 npm run test:e2e -- --grep 'optional real'`.

The cache regression suite checks Chromium, Firefox and WebKit using normal
browser profiles, including page reloads and browser restarts. The separate
optional title check downloads the real title model once:
`PLAYWRIGHT_REAL_TITLES=1 npm run test:e2e -- --grep 'optional real title model'`.
To verify actual CPU inference and cache reuse in all three engines, use
`PLAYWRIGHT_REAL_PORTABLE=1 npm run test:e2e -- --grep 'optional portable'`.
The multi-song mobile naming check runs four songs, reloads the page, restores
the library and names a second batch with model downloads blocked:
`PLAYWRIGHT_REAL_MOBILE_TITLES=1 npm run test:e2e -- --grep 'optional mobile'`.

The production preview uses port `4173` and the same repository path. `npm test` covers the JavaScript music and naming helpers; it is not proof of WebGPU compatibility on every device. Browser inference, downloads, cancellation, audio playback and optional naming need browser checks as well. The [aggregate release checks](../results/browser_validation.json) record the tested scope; detailed local evidence lives in `../evidence/web_pages_v1/` when available.

## Local inference and downloads

The melody model is an approximately **8.8 MB ONNX download**, plus its vocabulary and browser runtime. Downloads begin when generation is requested. Automatic mode tries **WebGPU**, then falls back to **WebAssembly on the CPU** if GPU initialization or inference fails. Safari/WebKit uses local CPU inference for both models: the tested WebKit engine crashes when ONNX's worker GPU device is torn down, including after normal session and device disposal. The GPU-only option is disabled there. Other engines retain automatic and explicit GPU selection. CPU inference uses one worker thread so the static site does not require cross-origin isolation headers. Older Safari without WebGPU receives the matching plain WASM runtime instead of the newer asyncify variant.

Five handcrafted melody guides—pop hook, chiptune, cinematic, dance and lullaby—adjust sampling. Mixed mode varies guides, tempos and keys. Custom tempo, key, length and sampling controls are available. The guides describe melodic preferences; the model has not been trained on labeled genres. Playback offers sine, triangle and square sounds.

Automatic titles are **off by default**. Enabling them starts an additional download of about **140 MB** for the quantized `onnx-community/SmolLM2-135M-Instruct-ONNX` model when the generated melodies are ready. The q8 weights are 135,658,354 bytes and the tokenizer is about 3.5 MB. This replaces the earlier roughly 800 MB title model to reduce download size and memory pressure on phones. Naming runs locally through Transformers.js and follows the browser compatibility policy above. The title worker releases its model after every completed batch; the next batch reads saved files without downloading them again. Cancelling or failing naming keeps the melodies available.

After a complete first download, melody weights, vocabulary, title model files and both browser runtimes are saved in small storage blocks. The page owns the storage connection, and dedicated inference workers exchange one block at a time. IndexedDB stores ArrayBuffer blocks, avoiding large Blob records on iOS. The Cache API provides a fallback, and files from earlier app versions remain readable. Completed writes are checked before the interface reports that a file was saved. Generation reuses loaded models within a tab and saved files after page reloads and normal browser restarts, with no model network request on a cache hit.

Model cache keys include the immutable model revision; runtime keys include the locked build revision. Cancelled partial downloads and unsuccessful HTTP responses are never reused. Cache writes failing due to quota or privacy settings do not prevent generation; a bounded memory fallback retains smaller files for the open tab without retaining another complete title model.

Website storage is still managed by the browser: clearing site data, private browsing and storage eviction can require another download. Safari uses its normal website-data storage for these files; adding the page to the Home Screen can improve persistence eligibility. See [WebKit's storage policy](https://webkit.org/blog/14403/updates-to-storage-policy/). This is model-file caching; the site shell still needs to be reachable after a reload.

The [mobile repair validation record](../results/mobile_browser_fix_validation.json) covers multi-song naming, library recovery, chunked persistence and real inference with model networking blocked after reload in Chromium, Firefox and WebKit. The [earlier cache validation](../results/browser_cache_validation.json) describes the previous implementation. WebKit runs at a mobile viewport; this does not replace testing on a physical iPhone.

Each visitor downloads and runs generation models on their own device. This static app has no hosted inference endpoint, API key, account requirement, or analytics code. Generated notes, imported RTTTL and title prompts are processed in the browser rather than uploaded for inference. Selected audio is decoded and transcribed in the browser; no audio or score content is uploaded. GitHub Pages and Hugging Face receive the ordinary requests needed to serve the site, runtime and model files.

WAV and MP3 are synthesized locally from the same score and selected waveform. MP3 loads its small encoder only on export. Exports are mono, 44.1 kHz; MP3 uses 128 kbps and can contain encoder padding. Audio is limited to five minutes per song.

The latest library (up to 64 scores) and completed titles are saved on the device as soon as each song is ready, before the optional title model runs. Reloading restores the scores and selection, including an interrupted batch, without starting models automatically. Naming updates only each song's title and RTTTL name; its notes, identity and place in the batch are preserved. Recent title history prevents repeated names across batches and reloads. If the small model gives unusable or repeated suggestions twice, a distinct title is derived from the melody's measured character and recorded as a fallback. Download songs for a durable backup, since clearing website data also clears the saved library.

## Model versions

### Verify browser transcription quality

`node scripts/verify-browser-transcription.mjs --asyncify` runs the actual bundled
Basic Pitch model through ONNX Runtime Web/WASM on six generated audio fixtures.
It evaluates pitches, onsets, rests and elapsed timing using known BPM and no time
stretching. The aggregate record is
[`../results/studio_browser_transcription_native.json`](../results/studio_browser_transcription_native.json).
This Node-hosted check exercises the browser conversion algorithm, while the
mandatory browser tests separately cover decoding, workers and the static site.
Synthetic fixture results do not establish accuracy on arbitrary songs.

### Verify continuation and selected edits against actual weights

The optional `scripts/verify-continuation.mjs` check uses native ONNX Runtime CPU
inference with the same immutable published checkpoint. Supply a directory
containing `model.onnx` and `tokenizer.json` from the manifest's pinned URLs:

```sh
node scripts/verify-continuation.mjs /path/to/model-files /path/to/result.json
```

It verifies hashes before running and never downloads files itself. It checks
source-conditioned continuation, selected-pitch variations, and repair scoring
against the original baseline and known suffix. Exact rhythm, rests, tempo and
unselected events are checked, along with seeded reproducibility. Recorded
evidence is in [`../results/studio_edits_native.json`](../results/studio_edits_native.json);
the earlier continuation-only result remains available separately. This native
check does not establish browser WebGPU/WASM compatibility, musical quality or
audio recognition accuracy. Browser evidence is recorded by the Pages workflow.

### Pinned downloads

[`public/model-manifest.json`](public/model-manifest.json) pins the melody ONNX model and tokenizer to an immutable Hugging Face revision, records their sizes/hashes, and identifies the local runtime path. [`src/title-utils.js`](src/title-utils.js) pins the optional title model separately. Changes to the published model's default branch do not silently change the browser app's selected weights; update these pins deliberately after validation.

The melody export is produced by [`../scripts/export_browser_model.py`](../scripts/export_browser_model.py). From the repository root, install `pip install -e '.[browser-export]'` alongside the project's normal requirements, then run:

```sh
python scripts/export_browser_model.py --model-dir /path/to/hf-snapshot --output /path/to/new-browser-export
```

The model snapshot must contain `config.json`, `model.safetensors` and `tokenizer.json`. Export verifies ONNX against PyTorch at five sequence lengths, without training or a corpus. Runtime versions and the MP3 encoder are pinned in `package.json` and `package-lock.json`. No training runs in the website or the deployment workflow.

## GitHub Pages deployment

The repository's Pages source must be **GitHub Actions**. [`../.github/workflows/pages.yml`](../.github/workflows/pages.yml) installs locked dependencies, runs unit and production browser checks, builds the static files, and deploys `web/dist`. It runs on `main` changes to `web/**` or the workflow itself, and supports manual dispatch. Publishing is restricted to `main`; the build and deployment jobs use separate minimal permissions. Official Actions are pinned to full commit hashes.

The build uses the Pages base path supplied by `actions/configure-pages`. For another host, override `VITE_BASE_PATH` when building, for example `VITE_BASE_PATH=/ npm run build` for a domain root. Serve the complete `dist/` folder, including runtime and third-party notice files. Model downloads require access to the pinned Hugging Face URLs.

The workflow follows [GitHub's custom Pages workflow guidance](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages). The Node requirement follows [Vite's supported versions](https://vite.dev/guide/).

## Player reuse and licenses

Playback and RTTTL parsing adapt code from [Adam Rahwane's RTTTL Play](https://github.com/adamonsoon/rtttl-play) and its `rtttl-parse` library, both declared MIT by their upstream package metadata. Scheduling, envelopes, progress, validation and exports extend that foundation. MP3 uses the LGPL `@breezystack/lamejs` encoder.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for exact revisions, attribution, licenses, preserved upstream code and the encoder's corresponding source archive. These notices and source downloads ship with the site under `vendor/`.
