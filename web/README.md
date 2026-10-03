# Pocket Composer

A browser music studio for the [RTTTL Generative Transformer](https://huggingface.co/charlie88agent/rtttl-generative-transformer): make melodies, listen with a moving piano roll, optionally name them, and download RTTTL text, WAV, or MP3.

**Site:** [speccy88.github.io/rtttl-generative-transformer](https://speccy88.github.io/rtttl-generative-transformer/)

## Run locally

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
`npx playwright install chromium`, then run `npm run test:e2e` after building.
The ordinary suite blocks external downloads. To explicitly test the real
melody model and your available GPU, use
`PLAYWRIGHT_REAL_MODEL=1 npm run test:e2e -- --grep 'optional real'`.

The production preview uses port `4173` and the same repository path. `npm test` covers the JavaScript music and naming helpers; it is not proof of WebGPU compatibility on every device. Browser inference, downloads, cancellation, audio playback and optional naming need browser checks as well. The [aggregate release checks](../results/browser_validation.json) record the tested scope; detailed local evidence lives in `../evidence/web_pages_v1/` when available.

## Local inference and downloads

The melody model is an approximately **8.8 MB ONNX download**, plus its vocabulary and browser runtime. Downloads begin when generation is requested. Automatic mode tries **WebGPU**, then falls back to **WebAssembly on the CPU** if GPU initialization or inference fails. Advanced settings can explicitly select either backend. CPU inference uses one worker thread so the static site does not require cross-origin isolation headers.

Five handcrafted melody guides—pop hook, chiptune, cinematic, dance and lullaby—adjust sampling. Mixed mode varies guides, tempos and keys. Custom tempo, key, length and sampling controls are available. The guides describe melodic preferences; the model has not been trained on labeled genres. Playback offers sine, triangle and square sounds.

Automatic titles are **off by default**. Enabling them starts an additional download of about **800 MB** for the quantized `onnx-community/Qwen2.5-0.5B-Instruct` model when the generated melodies are ready. Naming runs locally through Transformers.js, tries WebGPU, and can use the CPU if GPU loading fails. Cancelling or failing naming keeps the melodies available. Browser caches may avoid repeat downloads, but caches can be cleared or evicted by the browser.

Each visitor downloads and runs models on their own device. This static app has no hosted inference endpoint, API key, account requirement, or analytics code. Generated notes, imported RTTTL and title prompts are processed in the browser rather than uploaded for inference. GitHub Pages and Hugging Face receive the ordinary requests needed to serve the site, runtime and model files.

WAV and MP3 are synthesized locally from the same score and selected waveform. MP3 loads its small encoder only on export. Exports are mono, 44.1 kHz; MP3 uses 128 kbps and can contain encoder padding. Audio is limited to five minutes per song. The browser library is temporary: download songs you want to keep before closing or reloading the page.

## Model versions

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
