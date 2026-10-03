# Model and representation

The main model is a randomly initialized causal Transformer with 2,009,472 trainable parameters: four pre-layer-normalized residual blocks, width 192, six attention heads, GELU feed-forward width 768, dropout 0.1, learned positional embeddings, and tied input/output embeddings. Its context is 256 input tokens. No external model API or pretrained weights are used.

## Fixed event vocabulary

| Family | Tokens |
|---|---:|
| PAD, BOS, EOS | 3 |
| Exact integer BPM 25–900 | 876 |
| MIDI pitches 60–107 plus REST | 49 |
| Six duration denominators, plain or dotted | 12 |
| Total | 940 |

Each event has two tokens: pitch/rest followed by duration. The model learns P(pitch | history) × P(duration | history, pitch). Sharps and octaves are represented through the MIDI pitch; a dot multiplies duration by 1.5. Vocabulary values are fixed before data splitting.

For `Example:d=8,o=5,b=120:c,e,g,4c6`, the event sequence is C5 eighth, E5 eighth, G5 eighth, C6 quarter. Its tokens are BOS, BPM_120, PITCH_72, DUR_8_plain, PITCH_76, DUR_8_plain, PITCH_79, DUR_8_plain, PITCH_84, DUR_4_plain, EOS.

## Training objective and windows

Teacher-forced cross entropy uses the complete 940-token softmax, with padding excluded. Perplexity is exp(mean cross entropy in nats). BPM, event components, and EOS count as targets; BOS does not. Predictive evaluation does not apply the generation grammar mask, and scores are not directly comparable to character-token or compound-event models.

A window contains at most 127 complete musical events with BOS/BPM context. Long songs split at event boundaries. Repeated BPM targets in later windows are masked, and EOS is learned only at the true ending, so each original target is counted once. Continuity across window boundaries is an approximation. Attention never crosses between songs.

The measured corpus has 7,840 training windows and 759,018 unmasked targets per epoch. The final shorter accumulation group is token-weight normalized correctly.

## Baselines and decoding

The two-layer width-192 GRU has 806,572 parameters and resets hidden state between windows. The order-5 n-gram uses additive unigram smoothing and Witten–Bell interpolation. It fits full songs once and evaluates with the same target-window policy.

Generation enforces `BOS BPM (PITCH/REST DURATION)+ EOS`, then applies temperature, top-k and top-p sampling. Minimum event count is enforced; reaching the maximum forces EOS and is explicitly reported. The resulting RTTTL is re-parsed. This improves structural validity but does not establish composition quality, learned music theory, or originality.

## Repetition-aware generation

Teacher-forced prediction sees real prefixes during training. Autoregressive
generation sees its own outputs; a locally likely note or phrase can become a
self-reinforcing loop. The saved failing 64-event melody had two C6 notes followed
by 62 D-sharp6 notes. Its prefix fits within the model context, so sliding-window
truncation does not explain that failure.

The updated sampler penalizes recently used pitch/rest tokens, caps runs, and
breaks continuations of repeated pitch motifs before probability truncation.
Duration tokens are deliberately excluded: repeated rhythmic values are normal
music. Both hard checks use the full generated pitch history, including rests,
and catch loops after an introduction even if durations change. The controls
are optional and do not change model weights or predictive perplexity.

Diagnostics now include the longest local pitch/event run, number of distinct
pitches, maximum contiguous nonconstant motif repeats (periods 2–16), and the
fraction of overlapping pitch four-grams that repeat an earlier four-gram.
Degeneracy flags include pitch runs of at least eight and four or more copies of
a pitch motif. These new definitions must be applied to both comparison arms;
historical percentages using the old definitions are not directly comparable.

## Melody profiles and future learned control

The optional `pop-hook`, `chiptune`, `cinematic`, `dance`, and `lullaby` profiles
are small sets of soft decoding preferences, not separately trained models.
They nudge scale, register, interval size, rhythm, and rests while leaving
alternatives possible. Each profile has a seeded tempo range; `mixed` balances
profiles and varies keys within a batch. Explicit tempo, range, key, and mode
options override the defaults. Per-song settings are saved alongside outputs.
A major/minor setting is a scale
preference, not a validated happy/sad mood label. Audio remains a sine-wave
audition of the same monophonic RTTTL melody.

The current corpus has no reviewed genre, mood, instrumentation, or time-signature
labels. A few filenames contain source categories, but inferring genres from
artist/title keywords would create unreliable supervision. For genuinely learned
pop control, the next data step is a reviewed set of pop/hook annotations with
enough distinct musical families for separate train/validation/test subsets.
Available event data can already support objective labels for register, note
density, rest fraction, range, and rhythmic values; bin boundaries must be fit
on the training split only.

Adding learned control tokens requires a versioned tokenizer, variable-prefix
dataset/window handling, checkpoint migration, and conditioning training. It
cannot be implemented honestly by attaching untrained genre names to this
checkpoint. Future runs should measure requested-control adherence, raw and
guarded repetition, held-out prediction, training similarity, and blinded
listening separately. Improving cross entropy alone does not prove better hooks.

RTTTL cannot encode a full pop arrangement with chord, bass, drum, instrument,
velocity, and vocal tracks. Such a model needs a richer multitrack representation
and licensed training material; the current work improves melody generation.
