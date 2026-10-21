# Unified RTTTL Studio

The feature joins the existing browser generator to the converter at
[`speccy88/mp3_to_rtttl`](https://github.com/speccy88/mp3_to_rtttl). The full
[specification](https://github.com/speccy88/mp3_to_rtttl/blob/feat/rtttl-studio/docs/RTTTL_STUDIO_SPEC.md)
and [launch guide](https://github.com/speccy88/mp3_to_rtttl/blob/feat/rtttl-studio/START_HERE.md)
live with the local service.

## Responsibilities

| Layer | Responsibility |
| --- | --- |
| Converter Python core | Audio decode, one Basic Pitch analysis, source evidence, melody candidates |
| Converter HTTP service | Local static serving, upload bounds, asynchronous jobs and cleanup |
| `converter-client.js` | Same-origin service discovery, input validation, upload and polling |
| `main.js` | Shared library, original preview, candidate selection, editing and explicit continuation |
| `studio-score.js` | Pure tempo/transpose transformations and bounded append semantics |
| `sampler.js` | Validated source prefix, preserved events, additional-event budget, constrained sampling |
| `melody-worker.js` | Existing model initialization/cache/backend lifecycle; source forwarded to sampler |

## Continuation contract

Pass `options.promptSong = {name, bpm, events}` with events containing `pitch`
(MIDI 60–107 or null), `duration` (1/2/4/8/16/32), and boolean `dotted`.
`minEvents` and `maxEvents` count **additional** events. Source BPM overrides
composition tempo choices. Unsupported values are rejected before model loading.

The source becomes BOS + BPM + pitch/rest-duration pairs, without EOS. The first
model decision extends the source. Long prompts use the existing event-aligned
context window while retaining the full source in the returned score. Existing
repetition controls inspect source history but only constrain new sampled events.

Returned metadata includes `continuation`, `prefixEventCount` and
`generatedEventCount`. The combined score stays within the player's existing event
and five-minute limits. UI versions have a `source` label and optional `parentId`.
Creating a batch appends scores; no first-result reset discards prior imports.

## Checks

- `npm test`: existing tests plus prefix-conditioning, upload/polling, score editing
  and workspace preservation tests.
- `npm run build`: production build, retaining the existing runtime-copy step.
- `npm run test:e2e`: existing browser suite plus `tests/unified-studio.spec.js`.
  API and worker fixtures test UI contracts; they are not evidence of actual audio
  or model execution. Use `VITE_BASE_PATH=/` for both root-path builds and tests.
- `scripts/verify-continuation.mjs`: optional real-checkpoint native CPU check.
- Audio-inference quality and timing regressions are tested in the converter repo.

Existing model pins, browser cache keys, license notices, MP3 export and WebKit
fallback policy remain in use. No training or generated-weight commit is involved.
