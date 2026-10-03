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
