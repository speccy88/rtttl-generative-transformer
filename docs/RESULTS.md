# Executed results and limitations

The main Transformer was trained on October 3, 2026 using the configuration in `configs/runpod_verified_cuda.yaml`. Aggregate measurements are in `results/experiment.json`, with the recorded training curve in `results/training_metrics.csv`.

## Executed experiment

- Randomly initialized 2,009,472-parameter Transformer; 940-token vocabulary; context 256
- 7,828 train / 980 validation / 978 test songs; detected musical families disjoint
- 60 epochs and 7,380 optimizer updates, seed 42, training transposition ±2 semitones
- RTX 3090 24 GB class, PyTorch 2.6.0+cu124, Python 3.11.10, BF16
- 312.24 seconds cumulative training including validation, excluding setup/transfer/waiting
- 776.87 MiB peak allocated CUDA tensors, not total reserved/driver memory
- Validation: cross entropy 1.31701714 nats; perplexity 3.73227192; 95,600 targets
- Full test: cross entropy 1.34540137 nats; perplexity 3.83972739; 93,358 targets across 978 songs

Predictive evaluation uses the whole supplied validation/test splits without grammar masking. These are per-token scores for this exact representation. Validation selected the checkpoint; test was used for the reported final assessment.

## Generation batch

Sampling used 100 attempts, temperature 0.8, top-k 20, top-p 0.95, seed 42, minimum 8 and maximum 96 events, and sampled BPM. All 100 were syntactically valid and distinct. None was an exact training match under the event-based check. However:

- 19/100 had heuristic degeneracy flags
- 31/100 reached the length limit and forced EOS
- 3/100 were labelled near matches to training melodies
- 23/100 had moderate similarity and 74/100 low similarity
- Mean generated pitch range was 5.15 semitones, compared with 12.78 for test songs
- Mean generated length was 70.9 events, compared with 46.73 in the test split; length constraints affect that comparison

These measurements support functional generation and nontrivial prediction; they do not prove that the model composes good or legally original music. Repetition and reduced melodic range remain visible limitations. Generated audio is not included in this public release.

## Checks and scope

The original GPU environment passed 89 tests and complete parser/tokenizer/integrity checks covering all 9,786 retained songs. Public-release test results are reported separately in `results/release_validation.json`; they use synthetic fixtures and do not need the withheld corpus.

There is only one full Transformer training experiment here. No claim of statistically significant improvement over GRU/n-gram, equal-compute comparison, human listening score, hardware-player compatibility, or broad musical generalization is made. No new full GPU training was performed merely to publish the repository.
