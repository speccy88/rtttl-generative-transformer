---
library_name: pytorch
tags:
- music-generation
- symbolic-music
- rtttl
- transformer
- from-scratch
- safetensors
- mps
---

# RTTTL Generative Transformer (2M)

A compact decoder-only Transformer trained from random initialization to generate
monophonic melodies in **Ring Tone Text Transfer Language (RTTTL)**. It models
tempo, pitch/rest, and duration events. The current weights include a three-epoch
FP32 continuation on Apple M2 MPS.

The October 3, 2026 update adds repetition controls, five handcrafted melody
guides, mixed batches with varied tempos and keys, Apple GPU inference, local
WAV previews, and optional local LLM titles. This is a dataset-free inference
release: weights, fixed vocabulary, source, and aggregate measurements. The
training corpus and generated music/audio are omitted.

Original code and documentation retain the existing scoped MIT license in
[LICENSE_CODE](LICENSE_CODE). Weight and corpus scope is described in
[LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [PROVENANCE.md](PROVENANCE.md). The
optional Qwen title model is separately downloaded under its own Apache-2.0
license. No title-model weights are bundled here.

Full training, baselines, data preparation, evaluation, and release builder:
[GitHub source at this release's commit](https://github.com/speccy88/rtttl-generative-transformer/tree/{{SOURCE_COMMIT}}).

## Try it in your browser

**[Open Pocket Composer](https://speccy88.github.io/rtttl-generative-transformer/)** —
generate melodies locally with WebGPU, or WebAssembly on the CPU when needed.
Choose a guide, tempo, key and length, listen with the piano-roll player, and
export RTTTL text, WAV or MP3. No installation or account is required.
Optional Qwen-generated titles are off by default and require a separate
approximately 800 MB download; model inference stays on the visitor's device.

The browser uses a pinned 8.8 MB ONNX export of the same melody weights, plus
the browser runtime. Browser assets and export parity evidence live in
[`browser/`](browser/); the safetensors and Python interface remain available.
See [web app source and documentation](https://github.com/speccy88/rtttl-generative-transformer/tree/main/web)
for the player attribution, runtime versions and build instructions.

## Download and run

Download this repository into a local directory, then create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_inference
python inference.py --profile mixed --num-songs 25 --min-events 16 --max-events 48 --seed 4000 --output-dir mixed_melodies --audio
```

Open `mixed_melodies/listening/index.html`. The output directory must be new.
Each song's guide, key preference, and actual BPM appear on its card. All previews
use the same gentle sine sound. On Windows, activate with
`.\.venv\Scripts\Activate.ps1`, or invoke `.\.venv\Scripts\python.exe` directly.

`--device auto` selects CUDA, then Apple MPS, then CPU. On Apple Silicon use the
normal macOS PyTorch wheel installed by the command above; `--device mps`
requires the Apple GPU explicitly. For CPU-only Linux/Windows or CUDA wheels,
consult the [official PyTorch 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/).
Both melody and optional title inference run locally. Small sampling, rendering,
and file operations use the CPU.

Existing file/stdout usage remains available:

```bash
python inference.py --device cpu --num-songs 5 --bpm 120 --output melodies.rtttl
python inference.py --profile cinematic --num-songs 3
```

This custom PyTorch model loads local JSON and safetensors. It does not require
Transformers for melody inference and does not unpickle checkpoint objects.
Review the supplied source before running it. The inference package does not
require a training dataset; training-reference similarity is explicitly unavailable.

## Variety and repetition

| `--profile` | Default BPM range | Melody preference |
|---|---:|---|
| `pop-hook` | 88–132 | Compact hooks, smaller leaps, eighth/quarter notes |
| `chiptune` | 120–180 | Higher register, quick notes, wider jumps |
| `cinematic` | 60–104 | Lower sustained phrases, more space, natural minor |
| `dance` | 118–150 | Brisk even rhythm, fewer gaps |
| `lullaby` | 60–84 | Gentle slow phrases, small steps, longer notes |

`mixed` balances all five profiles and shuffles tonic preferences. Tempos are
sampled within inclusive ranges using an independent seeded plan. Single
profiles default to C; cinematic defaults to natural minor and the others to
major. `--tonic Bb` and `--mode natural-minor` override key preferences.

Use `--bpm-range 70 160` for a custom range or `--bpm 112` for one fixed tempo;
these options are mutually exclusive. With neither a profile nor a tempo option,
BPM is sampled from the learned distribution. Profiles are handcrafted soft
preferences applied during decoding. Learned genre/mood conditioning and
instrument arrangements would require different training data and representation.

The default sampler applies a 1.2 recent-pitch/rest penalty, caps consecutive
identical pitch/rest events at four, and breaks after three consecutive copies
of a nonconstant pitch motif of up to 16 events. These operate before top-k/top-p.
Repeated rhythm alone is unaffected. Adjust `--repetition-penalty`,
`--max-pitch-run`, and `--max-motif-repeats`; setting them to 1, 0, and 0 disables
the controls. Saved records include intervention counts and repetition statistics.

## Optional local titles

```bash
python -m pip install -r requirements-naming.txt
python inference.py --profile mixed --num-songs 25 --min-events 16 --max-events 48 --name-songs --output-dir named_melodies --audio
```

First naming use downloads about 1 GB of
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) safetensors
weights, pinned to `7ae557604adf67be50417f59c2c2f167def9a775`. No Ollama, separate
server, API key, or external LLM application is required. Add `--name-offline`
after the model is cached. `--name-model`, `--name-revision`, and `--name-cache`
allow local files or an explicitly chosen compatible model.

Naming runs after melody generation from measured symbolic features. It preserves
musical tokens and timing, stores full titles in `songs.jsonl`, and uses unique
ASCII names of at most 11 characters in RTTTL. Naming failures are reported and
saved melodies remain usable. Automatic naming selects CUDA, MPS, native XPU,
then CPU, and records any accelerator-to-CPU fallback. NPU/Vulkan require a
different runtime and are outside this Transformers integration.

## Python API

```python
from inference import load_local_model
from generation import generate_tokens
from rtttl import encode_rtttl

model, tokenizer = load_local_model(".", device="mps")
result = generate_tokens(model, tokenizer, profile="cinematic",
                         min_events=16, max_events=48, seed=42, device="mps")
print(encode_rtttl(tokenizer.decode(result["token_ids"])))
```

## Architecture and training

The model has **2,009,472 unique parameters**, four decoder blocks, width 192,
six heads, feed-forward width 768, and context 256. It uses pre-layer normalization,
causal scaled-dot-product attention, GELU, learned positions, and tied input/output
embeddings. Safetensors stores the tied matrix once. The vocabulary has 940 fixed
tokens: `BOS BPM (PITCH|REST DURATION)+ EOS`, MIDI pitches 60–107, integer BPM
25–900, and durations 1/2/4/8/16/32 with optional dots. Song titles are excluded
from melody-model inputs.

The original 60-epoch RTX 3090 run used 7,828 training, 980 validation, and 978 test
songs, grouped by detected musical families. It took 312.24 seconds including
validation. Corpus content matched historical collections linked by PICAXE;
the provenance and its limits remain documented in [PROVENANCE.md](PROVENANCE.md).

The new warm start used the unchanged splits, transposition ±2, batch 16 with
accumulation 2, learning rate 3e-5, three FP32 epochs, and 735 optimizer updates
on Apple M2 MPS. It took 208.41 seconds. Full validation covered 980 songs and
95,600 unmasked target tokens. Perplexity changed from 3.73227158 to
**3.71257212**; current cross entropy is **1.31172493 nats**. The held-out test
was not used to select this continuation. Historical test measurements in
`original_evaluation.json` describe the earlier weights.

## Measured decoding checks

A paired 100-seed MPS comparison of the original weights found expanded
degeneracy flags in 55/100 legacy outputs and 0/100 with repetition controls;
maximum pitch run changed from 64 to 4. On the continuation checkpoint the
corresponding counts were 53/100 and 0/100. Caps are enforced by design, and
length changes affect repetition statistics; these are not independent
musicality scores.

A separate mixed continuation batch retained all 25 attempts, five per profile:
22 tempos spanning 66–174 BPM, all 12 tonic preferences, 25 valid/distinct songs,
zero repetition flags, and four length-limit endings. All received titles
locally on MPS. There is no controlled human listening assessment.

Aggregate settings, denominators, and hashes are in `mps_finetune.json`,
`repetition_comparison_mps.json`, `naming_validation.json`, and
`variety_validation.json`. Historical original results remain in
`original_evaluation.json` and `original_training_config.json`.

## Release verification and limits

`VALIDATION.json` records checks of this packaged release. The safetensors export
is checked against the fine-tuned checkpoint's state tensors and forward logits.
The source project passed 341 tests including the release-builder checks;
publication measurements are in `publication_validation.json`. The bundled
dataset-free tests verify loading, tied weights, generation, tempo metadata,
repetition controls, and device selection. `SHA256SUMS` covers every other release
file. Optimizer state, RNG, and training records are omitted, so this is a
weights-only inference export.

Current weights SHA256: `{{WEIGHTS_SHA256}}`.

These short monophonic outputs cannot encode chords, instrument tracks, drums,
lyrics, velocities, or expressive timing. Historical ringtone material shapes
the distribution. Seed equivalence across devices/PyTorch versions is not
guaranteed. Novelty heuristics and grammar validity do not establish originality
or musical quality. The previous weights and inference release remain available
at [revision 4be28d83](https://huggingface.co/charlie88agent/rtttl-generative-transformer/tree/4be28d83ec077b84ebf5f8556b6d5075b54efe6b).
