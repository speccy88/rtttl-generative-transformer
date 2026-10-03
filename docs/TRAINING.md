# Training and reproducibility

## Setup and own data

Follow the root README to install Python dependencies and prepare a collection you may use. Run `python scripts/verify_package.py --data data/processed` after preparation. Keep the exact train/val/test JSONL and tokenizer files; generation, evaluation, and resume verify their SHA256 hashes against checkpoints.

Every training invocation creates a new run directory containing its configuration, dataset hashes, model information, CSV metrics, best/last checkpoints, sample text, and summary. Do not publish these automatically: samples and paths can contain material that needs separate review.

## Presets

- `local_2060s.yaml`: recommended full model, batch 16 × accumulation 4, automatic CPU/CUDA selection, CUDA FP16/BF16 as supported
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
