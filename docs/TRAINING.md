# Training and reproducibility

## Setup and own data

Follow the root README to install Python dependencies and prepare a collection you may use. Run `python scripts/verify_package.py --data data/processed` after preparation. Keep the exact train/val/test JSONL and tokenizer files; generation, evaluation, and resume verify their SHA256 hashes against checkpoints.

Every training invocation creates a new run directory containing its configuration, dataset hashes, model information, CSV metrics, best/last checkpoints, sample text, and summary. Do not publish these automatically: samples and paths can contain material that needs separate review.

## Presets

- `mac_mps.yaml`: three-epoch warm start on Apple Silicon; explicit MPS, FP32, batch 16 × accumulation 2, learning rate 3e-5, full validation
- `local_2060s.yaml`: full model, batch 16 × accumulation 4, automatic CUDA/MPS/CPU selection, CUDA FP16/BF16 as supported
- `runpod.yaml`: same model, batch 64 × accumulation 1, automatic device choice
- `runpod_verified_cuda.yaml`: measured configuration; explicit CUDA, 60 epochs, two workers, training-only transposition ±2 semitones
- `smoke_test.yaml`, `smoke_gru.yaml`, `smoke_ngram.yaml`: small functional checks using `data/smoke`
- `transformer_comparison.yaml`, `gru_comparison.yaml`, `ngram.yaml`: no-augmentation architecture baselines
- `gru.yaml`: augmented GRU experiment

Use the same splits, evaluation settings, and multiple seeds for meaningful comparisons. GRU and Transformer parameter counts differ, so these are not equal-parameter or equal-compute experiments. The full music experiment only establishes the main Transformer run described in RESULTS, not a controlled superiority claim over the baselines.

## Resume

```bash
python train.py --config configs/runpod_verified_cuda.yaml --resume runs_runpod/YOUR_RUN/checkpoint_last.pt
```

Preserve the original architecture, schedule and batching configuration. Resume creates a new child run rather than replacing the parent. Full checkpoints preserve optimizer, scheduler and random-number states. Inference-only checkpoints lack those states and cannot reproduce an exact resume. Checkpoints contain Python objects; only open trusted files.

The measured run was stopped after its first epoch, then resumed to epoch 60 using the same dataset/configuration. Its final summary reports 312.242753 cumulative training seconds. The best checkpoint's evaluation metadata captured 311.243146 seconds slightly earlier in the final save sequence; these are different instrumentation points, not independent training runs.

## Apple Silicon fine-tuning

```bash
python scripts/check_environment.py --device mps
python train.py --config configs/mac_mps.yaml --init-checkpoint /path/to/checkpoint_best_inference.pt
```

`--init-checkpoint` copies compatible weights into a new run with a fresh AdamW
optimizer, scheduler, and seeded RNG stream. It accepts the inference-only
checkpoint; `--resume` requires full optimizer/scheduler/RNG state. The two options
are mutually exclusive. Initialization validates the model architecture, tokenizer,
and all original split hashes. Each saved checkpoint records its initialization
source, and all runs receive new directories under `runs_mps/`.

Use a copied config to set `data.processed_dir` to the existing private dataset.
The Mac preset uses MPS explicitly and FP32, not CUDA autocast. Automatic selection
also recognizes MPS. Explicit `mps` never silently selects CPU; CPU processing still
handles input files, metrics, audio and the portable sampling RNG. MPS RNG state is
preserved for training and sampling isolation. Exact cross-device reproduction is
not promised. Synthetic regression tests also exercise CPU mechanics.

For a shorter invocation, `--stop-after-steps N` saves a resumable checkpoint at an
optimizer-update boundary without changing the planned learning-rate schedule.
The default preset is a bounded experiment, not a promise of musical improvement.
Compare full validation loss and same-seed raw/guarded generations against the
original checkpoint before choosing a new default. Keep the original available
even if a new checkpoint lowers cross entropy.

## Monitoring and cloud operation

CSV logs work by default. Optional TensorBoard:

```bash
python -m pip install -r requirements-tensorboard.txt
```

Set `training.tensorboard: true` in your own copied YAML, then run `tensorboard --logdir runs` (or `runs_runpod`). TensorBoard itself was not independently exercised in the reported experiment.

On a GPU host, install the appropriate official PyTorch build, upload your cleared code/data, verify CUDA with `scripts/check_environment.py`, and run the desired config. This repository does not create cloud accounts, provision GPUs, manage credentials, or stop paid instances. Save required results and explicitly stop resources through your provider when finished; persistent storage may continue to incur charges.

## Limits of reproducibility

The reported GPU stack was Python 3.11.10, PyTorch 2.6.0+cu124, RTX 3090, BF16, seed 42. NumPy/PyYAML/matplotlib have bounded dependency ranges rather than a full environment lock. Use the checked-in experiment configuration and metadata to compare environments.

CPU resume equality is tested. CUDA Flash Attention warned that it can remain nondeterministic even with deterministic settings, so bit-for-bit cross-hardware replay is not guaranteed. The original music corpus and weights are withheld; a public clone can reproduce the software workflow on an authorized replacement corpus, not the exact published scores.
