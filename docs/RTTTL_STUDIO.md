# RTTTL Studio — GitHub Pages specification

## User goal

One webpage for audio files to RTTTL, choosing and editing a melody, generating new
melodies, making variations of a converted score, and suggesting repairs to selected
notes. The entire end-user application runs on GitHub Pages, without installing or
starting a local server. This supersedes the initial local-Python integration.

## Required workflow

1. Open the public Studio URL. Choose **Convert audio** or the existing **Compose**
   controls. Audio import supports the formats the browser can decode; an unsupported
   codec has an actionable error rather than a cloud-upload fallback.
2. Choose an excerpt using a waveform or numeric fields and audition the original.
   Select tempo, rhythmic grid and lead preference as needed.
3. Run Basic Pitch once per conversion in a dedicated browser worker and compare
   several lead interpretations derived from that evidence. Keep the selected one
   in the same library as imported and generated RTTTL.
4. Select a note or phrase in the piano roll. Manual pitch/rest/duration edits create
   a new version. Keep tempo and transpose copies available.
5. **Vary selection** samples alternative pitches while preserving source rhythm,
   rests, tempo, event count and all unselected notes. Variation strength controls
   the pitch-change budget and range.
6. **Repair suggestions** compares bounded candidate replacements, including the
   original, using the causal model's likelihood over the edited span and a known
   suffix. An unchanged winner is explicitly identified. This is not a trained
   bidirectional masked model and does not establish acoustic correctness.
7. **Continue** preserves the score and appends events. **Compose** still generates
   new music with the existing style/key/tempo controls.
8. Compare versions, audition, and export RTTTL, WAV or MP3 from the same page.

## Architecture

| Component | Responsibility |
| --- | --- |
| GitHub Pages | Static HTML, JavaScript, CSS, licensed transcription model and WASM assets |
| Web Audio | Decode local files, prepare mono excerpts, waveform preview and audio playback |
| Transcription worker | Basic Pitch ONNX inference, note evidence and melody candidates |
| Melody worker | Existing checkpoint/cache/backend lifecycle and constrained symbolic inference |
| Sampler | New-song generation, exact prefix continuation, selected-pitch variations, suffix-scored repair |
| Shared editor | Source/phrase selection, manual edits, version comparison and export |

Audio and score contents are not sent to an inference API. The browser downloads
existing pinned generator weights when needed. Basic Pitch is distributed with the
site and hash-verified. CPU WASM uses one thread because GitHub Pages does not
provide application-controlled cross-origin isolation headers. Existing generation
WebGPU/WASM fallback and WebKit policy remain in use.

The Python converter repository remains useful as a desktop/CLI tool and as the
reference for timing, attack, rest and melody-selection behavior. It is not a
runtime dependency of this webpage.

## Editing contract

`promptSong` contains a parsed score. `selection: {start, end}` is a zero-based,
half-open event range. `editMode` is `variation` or `repair`. A selection is bounded
to 16 events and must include a pitched note; an edit batch has at most four
versions. Duration, dots, rests, event count, BPM and events outside the range are
preserved. Unsupported rhythms or invalid ranges fail before downloading a model.

Repair uses forward likelihood for the candidate span and up to eight unchanged
suffix events. The 256-token context limits useful surrounding material. It does
not gain bidirectional attention by forcing or appending a suffix. A genuine
masked-note denoiser would require a versioned tokenizer, corruption/masking
training objective, new checkpoint and separate held-out evaluation.

## Acceptance criteria

- The deployed page has no converter API/localhost health check, setup instructions
  or dependency on a running Python process.
- A real generated audio fixture is decoded and transcribed by the actual model in
  a browser, producing a valid nonempty RTTTL score; input audio is not uploaded.
- Static model and runtime assets work under the GitHub Pages repository prefix.
- Silence stays silent, supported short notes and rests survive, and candidate
  timing is evaluated against the source without stretching away tempo mistakes.
- Excerpt selection, candidate selection and note/phrase selection are available.
- Variations preserve all promised locks and change at least one selected pitch;
  repair may retain the original and reports that outcome honestly.
- Real generator inference is verified for variations/repair in addition to unit
  tests with controlled logits. UI mocks are labelled separately.
- Cancellation, model/codec failures and subsequent retries preserve saved scores.
- Existing composition, continuation, browser cache, playback, export and mobile
  layout tests continue to pass.

## Ordered implementation and validation

1. Replace local-server assumptions in both repositories and record this contract.
2. Port transcription to a browser worker with the existing licensed ONNX model,
   bounded decoding and onset-aware melody selection.
3. Add constrained selected-note sampling and suffix-scored repair to the existing
   sampler, with exact preservation and cancellation tests.
4. Connect waveform selection, interactive notes and edit actions to the shared UI.
5. Run unit tests, production build, real-model browser checks and native reference
   checks. Preserve measured results and explicit limitations.
6. Publish through the repository's existing GitHub Pages workflow and verify the
   deployed static assets and workflow result.

## Quality limits

Dense recordings can still lead the transcriber toward accompaniment. Lead
preference is a decoding choice, not stem separation. Browser codec support differs
by device. Very long files may exceed browser decode memory even when their encoded
size is small, so use short source clips. Repair scores reflect the symbolic model's
preferences; listening and source evidence remain necessary when fidelity matters.
