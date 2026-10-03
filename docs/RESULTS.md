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

## Repetition-control comparison on Apple M2

A new paired test uses the original unchanged inference checkpoint on MPS,
PyTorch 2.6.0, 100 seeds (1000–1099), fixed 120 BPM, 16–64 events, temperature
0.8, top-k 20, and top-p 0.95. Every attempt is retained. The legacy arm disables
all new repetition controls; the guarded arm uses their defaults without a
musical profile. The first ten legacy outputs exactly reproduce the user's
saved MPS batch. Machine-readable settings, checkpoint/data hashes, timings,
and measurements are in `results/repetition_comparison_mps.json`.

| Measurement | Legacy | Repetition controls |
|---|---:|---:|
| Valid / distinct songs | 100 / 100 | 100 / 100 |
| Songs with expanded degeneracy flags | 55 / 100 | 0 / 100 |
| Maximum consecutive identical pitches/rests | 64 | 4 |
| Maximum consecutive copies of a nonconstant pitch motif | 32 | 3 |
| Mean distinct pitches per song | 3.88 | 9.49 |
| Mean events per song | 57.18 | 39.73 |
| Mean repeated pitch four-gram fraction | 0.67185 | 0.19252 |
| Forced length-limit endings | 61 / 100 | 6 / 100 |
| Exact training matches | 0 / 100 | 0 / 100 |

Both arms use the expanded checks for local loops and motifs after an intro;
their degeneracy percentages must not be compared directly with the historical
19/100 measured above. Run and motif maxima are constrained by design, so zero
flags is not independent proof of musical quality. The shorter outputs also
affect length-sensitive repetition statistics. No human listening study was
performed. This comparison establishes a decoding improvement, not an
improvement in learned weights, genre control, or subjective composition.

## Three-epoch MPS continuation

The original inference weights were then initialized into a new three-epoch
FP32 experiment on the same Apple M2 (16 GiB unified memory). This used the full
7,828-song training split, batch 16 with accumulation 2, transposition ±2, learning
rate 3e-5, 735 optimizer updates, and 208.41 seconds including validation and
training samples. The original files were preserved. This is a fresh-optimizer
warm start, not an exact continuation of the former CUDA optimizer state.

Both checkpoints were re-evaluated on MPS against all 980 original validation
songs / 95,600 target tokens. Cross entropy decreased from 1.31701705 to
1.31172493 nats, and perplexity from 3.73227158 to 3.71257212. The held-out test
split was not used for this checkpoint choice. See `results/mps_finetune.json`.

The same 100-seed generation comparison on the new checkpoint found 53/100
flagged songs with legacy decoding and 0/100 with the controls. This small raw
change from 55/100 does **not** establish that extra training fixed repetition.
The guarded new checkpoint averaged 9.10 distinct pitches and a 0.18267 repeated
pitch four-gram fraction. Pop guidance was disabled for these measurements.

The new checkpoint is a candidate with a modest predictive gain. It has not
learned pop labels, and there is no human assessment establishing that it makes
better music than the original. The separate pop-hook audio examples combine
this candidate with the documented handcrafted guidance and repetition controls.

## Optional local title model

The embedded Qwen2.5-0.5B-Instruct title model named all 20 existing pop-hook
examples with no failed titles, using automatic MPS selection and cached files
only. The run took 19.44 seconds including WAV rendering. All 20 full titles and
compact RTTTL names are distinct; one compact-name collision required a suffix.
Every original musical token, RTTTL defaults/melody section, generation metric,
and WAV file was unchanged (WAVs checked byte-for-byte).

A separate generation-plus-naming integration verified real CPU naming with
FP32, while melody generation still used MPS. Automatic accelerator failure
fallback and CUDA/XPU selection were tested with mocks; those devices were not
available for physical hardware validation. A CPU offline check blocked HTTP
requests. The source test suite passed 255 tests and dependency checks passed.
The official 988,097,824-byte safetensors file's SHA256 was verified. Details
are in `results/naming_validation.json`.

These are functional naming checks, not judgments of title quality or originality.
Titles are creative suggestions from measured symbolic features, not audio
interpretations. NPU/Vulkan are outside this simple Transformers integration.

## Varied melody batch on Apple M2

A 25-song mixed batch used the unchanged MPS continuation checkpoint, seed 4000,
16–48 events, and the default repetition controls. All attempts were retained:
five each of pop-hook, chiptune, cinematic, dance, and lullaby. There were 22
distinct tempos from 66 to 174 BPM and all 12 tonic preferences. Actual BPM agreed
between the saved plan, tokens, RTTTL, audio manifest, and listening cards.

All 25 songs were valid and distinct, with zero degeneracy flags; four reached
the length limit. All received unique full titles from the cached local title
model. Melody inference and title inference both used MPS. The run took 37.02
seconds including naming and WAV rendering. All 25 WAV files were checked for
duration and playable PCM structure. Original checkpoints and earlier batches
were preserved. The updated source suite passed 339 tests.

Observed mean note density was 3.83 notes/second for chiptune, 3.85 for dance,
and 0.98 for cinematic. These five-song subsets illustrate variety, not a
controlled comparison or listening score. The profiles are handcrafted sampling
preferences; no genre labels or new weights were trained. All previews share the
same sine-wave renderer. Settings, measurements, and verification are in
`results/variety_validation.json`.

## Checks and scope

The updated publication package passed 341 source tests and eight standalone
inference tests on Apple M2. Exported safetensors preserve all 53 checkpoint state
entries and CPU forward logits exactly. A five-profile named MPS smoke batch from
the standalone export reproduced the existing musical tokens and WAV bytes.
See `results/publication_validation.json`. No training was rerun for publication.

The original GPU environment passed 89 tests and complete parser/tokenizer/integrity checks covering all 9,786 retained songs. Public-release test results are reported separately in `results/release_validation.json`; they use synthetic fixtures and do not need the withheld corpus.

There is only one full Transformer training experiment here. No claim of statistically significant improvement over GRU/n-gram, equal-compute comparison, human listening score, hardware-player compatibility, or broad musical generalization is made. No new full GPU training was performed merely to publish the repository.
