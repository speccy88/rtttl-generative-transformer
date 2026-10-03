# RTTTL Generative Transformer

A small, real **decoder-only Transformer trained from random weights** to generate monophonic melodies in Ring Tone Text Transfer Language (RTTTL). Includes GRU and interpolated n-gram baselines, an audited parser, family-aware dataset splits, reproducible training, similarity checks, and WAV rendering. No pretrained model or remote generation API is used.

## Public release scope

This repository contains **source code, tests, configurations, English documentation, and aggregate experiment results**. It does not include the original music collection, processed records, trained checkpoints, or generated melodies/audio: the supplied corpus has no established redistribution license. Bring a corpus you are entitled to use, or run the synthetic demonstration below. Exact reproduction of the reported music experiment requires the original, non-public inputs and checkpoints.

See [data and rights](docs/DATA_AND_RIGHTS.md) and [license notice](LICENSE_NOTICE.md). Public visibility is not a license grant.

## Features

- Strict RTTTL parsing, canonical round trips, and per-record rejection/normalization audit
- Fixed 940-token pitch/rest, duration, tempo, and boundary vocabulary
- Exact deduplication and transposition/rhythm-aware family grouping before splitting
- 2,009,472-parameter causal Transformer: 4 layers, width 192, 6 heads, context 256
- Training-only transposition, mixed precision, gradient accumulation, early stopping, and full-state resume
- Grammar-constrained sampling with temperature, top-k, top-p, optional BPM, and explicit length-limit flags
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

Omit `--bpm` to sample tempo. Add `--device cpu` to run without CUDA. Existing output folders are refused. Only load trusted PyTorch checkpoints; this project loads Python objects from them.

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
