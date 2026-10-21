# RTTTL Studio development agreements

- The complete end-user studio runs on GitHub Pages. No Python server, hosted
  transcription endpoint, localhost connection, API key or user audio upload is
  part of the browser workflow.
- Keep audio decoding, transcription, generation, editing, audition and export on
  the visitor's device. Model/runtime downloads are ordinary static GET requests.
- Preserve existing composer controls, model pins, version library, downloads,
  runtime compatibility policy and third-party notices.
- Treat the selected score and note range as the shared input for manual editing,
  variations, repair suggestions and continuation. Preserve originals as versions.
- Keep unselected notes, rhythm and tempo exact when an operation promises to lock
  them. Validate that contract with meaningful tests.
- The current checkpoint is causal. Suffix-scored repair is an approximate search
  method, not a BERT-trained denoiser and not proof of the original audio notes.
- Bound device work, support cancellation and report codec/model errors clearly.
- Record real inference separately from mocked UI tests. Never label a synthetic
  fixture score as recognition accuracy on arbitrary full-mix songs.
- Keep model-generated melodies out of public evidence; retain hashes and metrics.
  Never commit user recordings or model caches.
- The reviewed, licensed Basic Pitch asset under `web/public/transcription/` is an
  explicit static-distribution exception to the model-artifact ignore rule; retain
  its license, hash and upstream provenance.
- Behavior, acceptance criteria and plan live in `docs/RTTTL_STUDIO.md`.
