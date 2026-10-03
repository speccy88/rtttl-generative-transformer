# Data preparation, parser policy, and rights

## Public boundaries

Source and aggregate research measurements are published. The original archive, canonical song records, rejected raw text, per-song source metadata, deduplication manifests, trained weights, generated melodies, and listening audio are omitted because redistribution rights have not been established. The training run remains documented, but a clean clone cannot reconstruct that private experiment's exact inputs or weights.

The tokenizer is a fixed grammar defined in code, not fitted to corpus statistics. Preparation regenerates its JSON locally. There is no hidden dataset download or pretrained-weight fetch.

## Observed collection

The supplied collection contained 11,144 candidate RTTTL records. The parser accepted 10,935 and rejected 209. Exact deduplication retained 9,786 songs; family-aware splitting produced 7,828 training, 980 validation, and 978 test songs. Separately, 119 PICAXE BASIC tunes and four other non-RTTTL files were excluded. PICAXE bytecode is a different representation and is not guessed into RTTTL.

## Accepted representation

- Scientific pitches C4–B7 (MIDI 60–107), plus rests
- Whole-note denominators 1, 2, 4, 8, 16, 32, with at most one dot
- Integer BPM 25–900
- Missing defaults resolve to `d=4,o=6,b=63`, with warnings
- Explicit defaults are always emitted when exporting
- Safe whitespace, line wrapping, accidental placement, and dot placement are normalized with warnings
- Entire malformed melodies are rejected; bad notes are not silently dropped
- UTF-8 is preferred, then CP1252 and Latin-1 without lossy decoding

Source titles are metadata, not model input; exports sanitize names to at most 11 ASCII characters. RTTTL historical dialects vary, so these limits describe this project's policy rather than universal player compatibility. No physical RTTTL playback hardware was validated.

A corpus-specific rule treats `_` as sharp only for the historical `rtttl3` source directory. The original evidence included 17 exact matches with sharp-spelled versions. This is a documented assumption; the general parser rejects underscores unless explicitly opted in. Review this rule before preparing an unrelated collection with a similarly named directory.

## Deduplication and splitting

Exact identity uses tempo plus resolved events, excluding title. Family grouping is invariant to global transposition, global rhythmic scaling, and tempo, then conservatively joins near/contained patterns. Entire connected families are assigned to one split. These checks reduce detected leakage; they cannot prove that every cover or rearrangement is identified.

Only training songs are randomly transposed, within legal bounds and the configured semitone range. All windows of the same song use the same epoch-specific shift. Validation and test are unchanged.

Preparation emits checksums, source inventory, parse audit, manifest, deduplication report, statistics, and split files. Those generated files can contain original melodies and private source metadata. Keep them local unless separately cleared for publication; `.gitignore` excludes the data directory.
