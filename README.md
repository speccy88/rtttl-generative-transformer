# RTTTL Generative Transformer

A small, real **decoder-only Transformer trained from random weights** to generate monophonic melodies in Ring Tone Text Transfer Language (RTTTL). Includes GRU and interpolated n-gram baselines, an audited parser, family-aware dataset splits, reproducible training, similarity checks, and WAV rendering. Melody generation uses no pretrained model or remote generation API. An optional, separate local language model can name completed melodies.

**[Open Pocket Composer →](https://speccy88.github.io/rtttl-generative-transformer/)**
Make melodies directly in your browser with WebGPU or a local CPU fallback.
Choose a melody guide, tempo, key and length; listen with the animated piano
roll; and download RTTTL text, WAV or MP3. Optional automatic titles use a
separate local model (about 800 MB), downloaded only after you enable naming.
No installation, account or inference server is needed. See the
[web app documentation](web/README.md) for development and deployment.

## Public release scope

This repository contains **source code, tests, configurations, English documentation, and aggregate experiment results**. It does not include the original music collection, processed records, trained checkpoints, or generated melodies/audio: the supplied corpus has no established redistribution license. Bring a corpus you are entitled to use, or run the synthetic demonstration below. Exact reproduction of the reported music experiment requires the original, non-public inputs and checkpoints.

See [data and rights](docs/DATA_AND_RIGHTS.md) and [license notice](LICENSE_NOTICE.md). Public visibility is not a license grant.

The separate [Hugging Face model release](https://huggingface.co/charlie88agent/rtttl-generative-transformer)
provides dataset-free safetensors inference, including the MPS continuation,
repetition controls, mixed melody guides, audio previews, and optional titles.
Its existing scoped license and provenance are documented in that repository.

## Features

- Strict RTTTL parsing, canonical round trips, and per-record rejection/normalization audit
- Fixed 940-token pitch/rest, duration, tempo, and boundary vocabulary
- Exact deduplication and transposition/rhythm-aware family grouping before splitting
- 2,009,472-parameter causal Transformer: 4 layers, width 192, 6 heads, context 256
- Training-only transposition, mixed precision, gradient accumulation, early stopping, and full-state resume
- Grammar-constrained sampling with temperature, top-k, top-p, optional BPM, and explicit length-limit flags
- Pitch and phrase repetition controls, plus five optional melody guides with varied tempos and keys
- Apple Silicon MPS inference and FP32 fine-tuning from an existing checkpoint
- Optional local song titles using a small Hugging Face safetensors model; no separate LLM application or server
- Full-split unmasked perplexity, repetition/degeneracy checks, nearest-training similarity, distribution plots, and browser-based listening pages

## Installation

Use Python 3.10–3.13; Python 3.11 was used for the reported GPU experiment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m pytest -q
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`, or use `.\.venv\Scripts\python.exe` directly if activation is restricted. Do not change your system's execution policy for this project.

On Apple Silicon, replace the CPU-wheel install line with
`python -m pip install torch==2.6.0` to get the macOS wheel with MPS support.

For a compatible NVIDIA GPU, replace the CPU install line with:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Use a working NVIDIA driver and consult the [official PyTorch versioned installation instructions](https://pytorch.org/get-started/previous-versions/). Newer GPUs may require a separately tested newer PyTorch/CUDA build. The CPU installation is sufficient for tests and the synthetic demonstration.

## Quick start without a music dataset

Create deterministic artificial RTTTL examples, prepare family-disjoint splits, and run a tiny CPU training job:

```bash
python scripts/create_demo_dataset.py --output data/demo
python scripts/make_smoke_subset.py --input data/demo --output data/smoke
python train.py --config configs/smoke_test.yaml
```

Training prints `RUN_DIR=...`. Replace `YOUR_RUN` below with that actual directory name:

```bash
python generate.py --checkpoint runs/YOUR_RUN/checkpoint_best.pt --data data/smoke --device cpu --num-songs 10 --min-events 8 --max-events 32 --audio
python evaluate.py --checkpoint runs/YOUR_RUN/checkpoint_best.pt --data data/smoke --device cpu --num-songs 10 --max-events 32
```

Open the generated output's `listening/index.html` to listen. This artificial demonstration checks the pipeline; its output and scores are not musical-quality evidence and are not the reported experiment.

## Train on your own authorized collection

Input can be a ZIP, directory, or text RTTTL file. Preparation never modifies the source:

```bash
python scripts/inspect_dataset.py --input /path/to/RTTTL.zip --output data/audit
python scripts/prepare_dataset.py --input /path/to/RTTTL.zip --output data/processed --seed 42
python scripts/verify_package.py --data data/processed
python train.py --config configs/local_2060s.yaml
```

Use at least three distinct musical families. Output directories must be new or empty. The local preset uses batch 16 with accumulation 4. For the measured RTX 3090 configuration:

```bash
python train.py --config configs/runpod_verified_cuda.yaml
```

This configuration requires CUDA and uses batch 64, BF16 where supported, and a 60-epoch ceiling. It creates files under `runs_runpod/`. It does not provision, purchase, or stop cloud resources. See [training and reproducibility](docs/TRAINING.md) for resume, monitoring, and baselines.

## Generate and evaluate your trained model

The dataset must match the checkpoint's stored split/tokenizer hashes, including when generating. It is needed for honest training-match comparisons.

```bash
python generate.py --checkpoint runs/YOUR_RUN/checkpoint_best.pt --data data/processed --num-songs 20 --temperature 0.8 --top-k 20 --top-p 0.95 --bpm 120 --audio
python evaluate.py --checkpoint runs/YOUR_RUN/checkpoint_best.pt --data data/processed --num-songs 100 --min-events 8 --max-events 96 --temperature 0.8 --top-k 20 --top-p 0.95 --seed 42 --audio
```

Without a profile, omit `--bpm` to sample tempo. Automatic device selection supports CUDA, Apple MPS, and CPU; use `--device cpu` to require CPU. Existing output folders are refused. Only load trusted PyTorch checkpoints; this project loads Python objects from them.

### Apple Silicon and less repetitive melodies

`--device auto` selects CUDA, then Apple MPS, then CPU. On an Apple Silicon Mac,
use `--device mps` to require Metal acceleration and fail explicitly if unavailable.
All model forward/backward passes use that backend; the small sampling RNG and
RTTTL/audio/file processing run on the CPU. No cloud service is involved.

Repetition controls are now enabled by default: a 1.2 penalty on recently used
pitches/rests, at most four identical consecutive pitches/rests, and a break after
three contiguous copies of a nonconstant pitch motif of up to 16 events. They
operate **before** top-k/top-p, so an overconfident loop cannot hide every escape
candidate. Rhythm alone is not penalized. These adjustable heuristics preserve
some repetition but can also interrupt intentional ostinatos.

For a varied batch of short melody ideas:

```bash
python generate.py --checkpoint runs/YOUR_RUN/checkpoint_best.pt --data data/processed --device mps --profile mixed --min-events 16 --max-events 48 --num-songs 25 --seed 4000 --name-songs --name-offline --audio
```

`mixed` balances the following profiles across a batch, samples a tempo for each
song, and rotates through shuffled keys. A seed reproduces the plan and sampling
on the same backend. Omit the naming options if the optional model is not cached
yet; see the naming instructions below.

| `--profile` | Default BPM range | Melody preference |
|---|---:|---|
| `pop-hook` | 88–132 | Compact hooks, smaller leaps, eighth/quarter notes |
| `chiptune` | 120–180 | Higher register, quicker notes, wider jumps |
| `cinematic` | 60–104 | Lower sustained phrases, more space, natural minor |
| `dance` | 118–150 | Brisk even rhythm, fewer gaps |
| `lullaby` | 60–84 | Slow gentle phrases, small steps, longer notes |

Select one profile for a consistent direction, still with varying BPM. Use
`--bpm-range 70 160` for a custom inclusive range or `--bpm 112` for one fixed
tempo; these options are mutually exclusive and override profile tempo ranges.
Without a profile or tempo option, the model samples its learned BPM distribution.
`--tonic Bb` and `--mode natural-minor` override key preferences for every song.
Single profiles default to C; cinematic defaults to natural minor and the others
to major. Each record saves its resolved `generation_settings`; listening cards
show style, key preference, and actual BPM.

These are handcrafted generation guides, **not learned genre or mood
conditioning**. They nudge sampling without guaranteeing scale adherence or
phrase structure. All previews use the same sine-wave sound; chiptune and
cinematic describe melodic direction, not instrumentation or full arrangements.

Use `--repetition-penalty 1 --max-pitch-run 0 --max-motif-repeats 0` without a
profile to reproduce the original sampler on the same backend and seeds. Every
generated record retains intervention counts; no bad songs are silently discarded.
To compare both samplers with identical seeds and all attempts retained:

```bash
python scripts/compare_sampling.py --checkpoint runs/YOUR_RUN/checkpoint_best.pt --data data/processed --output evaluations/repetition_comparison --device mps
```

To continue training from an inference-only checkpoint on your Mac, start a
separate experiment with fresh optimizer state:

```bash
python train.py --config configs/mac_mps.yaml --init-checkpoint /path/to/checkpoint_best_inference.pt
```

The preset uses three FP32 epochs on MPS, with a smaller learning rate and the
same architecture. Set `data.processed_dir` in a copy of the config to your exact
original splits. See [training](docs/TRAINING.md) for the distinction from resume,
and [model limitations and next steps](docs/MODEL.md) for learned style controls.

### Optional local song names

Install the optional Python library once (already installed in the supplied Mac
inference environment), then add one option to a normal generation command:

```bash
python -m pip install -r requirements-naming.txt
python generate.py --checkpoint runs/YOUR_RUN/checkpoint_best.pt --data data/processed --name-songs --audio
```

Alternatively install the package extra with `python -m pip install '.[naming]'`.
Naming runs after generation and cannot change notes, timing, tempo, token IDs,
or similarity scores. It uses measured tempo, register, pitch movement, rests,
and rhythm to suggest titles. It reads symbolic features, not the audio itself.
Original source/training song names are not sent to the title model.

The default is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`. First use downloads
about **1 GB** of safetensors weights into the ordinary Hugging Face cache.
Inference is entirely local; no account, API key, Ollama, LM Studio, server,
compiler, or separate LLM application is needed. Transformers 4.57.6 runs inside
the existing Python environment. Weights are not included in the source ZIP.

`--name-device auto` detects CUDA, Apple MPS, then native Intel XPU when supported
by the installed PyTorch build, otherwise CPU. If the selected accelerator fails,
auto mode retries on CPU and records the reason. An explicit `--name-device mps`
or `cpu` does not silently switch devices. This implementation does not use NPU
or Vulkan backends: those require a different runtime/model export. Core melody
device selection remains independent through `--device`.

Use `--name-offline` for cached/local files only, `--name-cache PATH` for a custom
cache, or `--name-model /path/to/model` for a local safetensors directory. A model
override must be a chat-capable architecture supported by the installed
Transformers version; remote model code is disabled. `--name-revision` can pin a
different model. Full titles are stored in `songs.jsonl` and displayed in the
listening page; RTTTL receives unique ASCII names within its 11-character limit.
Duplicate/invalid suggestions get one retry. Failures are reported, and music
is saved before naming so a missing dependency/model never discards a batch.

Existing songs can receive titles in a **new** directory without regeneration:

```bash
python scripts/name_songs.py --input generated/YOUR_BATCH --output generated/YOUR_BATCH_named --name-offline --audio
```

Omit `--name-offline` for the initial download. The original batch is preserved.
Model identity, resolved backend, fallback reasons, and per-song naming status
are recorded in metadata. Naming is disabled by default.

### Rebuild the Hugging Face inference package

The model release is built separately from the source-only GitHub archive. Use
the original Hub snapshot at revision `4be28d83ec077b84ebf5f8556b6d5075b54efe6b`
as `--base-release`; it supplies the existing license and provenance files.
The trusted checkpoint must match the SHA256 in `results/mps_finetune.json`:

```bash
python -m pip install '.[release]'
python scripts/build_hf_release.py --checkpoint /path/to/mps/checkpoint_best.pt --base-release /path/to/original-hub-snapshot --output /path/to/new-release
```

The builder exports safetensors, fixed vocabulary, standalone inference source,
templates, and aggregate results. It omits optimizer/RNG state and the corpus,
and makes no network or upload calls. Validate the new directory with
`python -m unittest -v test_inference`, add a measured `VALIDATION.json`, and
refresh its `SHA256SUMS` before publication.

## Measured experiment: October 3, 2026

| Measurement | Result |
|---|---:|
| Family-disjoint train / validation / test songs | 7,828 / 980 / 978 |
| Parameters | 2,009,472 |
| Training | 60 epochs, 7,380 optimizer updates |
| Hardware | NVIDIA RTX 3090, 24 GB class |
| Training wall time, including validation | 312.24 seconds |
| Peak allocated CUDA tensors | 776.87 MiB |
| Validation cross entropy / perplexity | 1.31702 / 3.73227 |
| Full test cross entropy / perplexity | 1.34540 / 3.83973 |
| Test target tokens | 93,358 |
| Evaluation generations | 100 |
| Valid / distinct / exact training matches | 100 / 100 / 0 |
| Degeneracy flags / forced length endings / near matches | 19 / 31 / 3 |

Time excludes installation, provisioning, transfers, and other operational waiting. Allocated tensor memory is not total GPU or driver memory. Syntax validity comes partly from grammar constraints; it is not evidence of musical quality. Similarity checks do not prove legal originality. See [results and limitations](docs/RESULTS.md) and machine-readable [aggregate results](results/experiment.json).

## Project map

- `src/rtttl_gen/`: parser, tokenizer, corpus preparation, models, training, generation, evaluation, audio
- `train.py`, `generate.py`, `evaluate.py`: command-line entry points
- `scripts/`: preparation, inspection, synthetic demo, audio, environment and integrity tools
- `configs/`: GPU, CPU smoke, and controlled baseline presets
- `tests/`: parser/model/training/evaluation/regression tests using synthetic fixtures
- `docs/`: architecture, data policy, reproduction, results, references
- `results/`: sanitized aggregate measurements and training curve

## Important limitations

Monophonic events only: no chords, instruments, velocities, lyrics, expressive timing, or genre labels. The model has short context; long melodies are windowed. Family detection is heuristic and may miss rearrangements. One trained run is not a statistically controlled comparison. Musical quality needs listening and independent assessment. CUDA Flash Attention may remain nondeterministic despite deterministic settings. Windows and RTX 2060 Super performance have not been independently verified in this release.
