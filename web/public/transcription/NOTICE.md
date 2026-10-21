# Browser transcription model

`basic-pitch.onnx` is the unmodified ICASSP 2022 ONNX model distributed in
Spotify's `basic-pitch` Python package, version **0.4.0**.

- Upstream project: https://github.com/spotify/basic-pitch
- Original package path: `basic_pitch/saved_models/icassp_2022/nmp.onnx`
- File size: **230,444 bytes**
- SHA-256: `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`
- License: Apache License 2.0; see `APACHE-2.0.txt`.
- Upstream notice is preserved verbatim in `SPOTIFY-NOTICE.txt`.

The browser's windowing, overlap removal, inferred-onset calculation, and
polyphonic note-event extraction in `src/transcription-core.js` are adapted from
Spotify's Apache-2.0 `inference.py` and `note_creation.py`. The adaptation uses
typed JavaScript arrays and a sorted remaining-energy traversal. Browser inference
uses ONNX Runtime Web; it does not ship the Python, TensorFlow, or dataset packages
mentioned in the original package notice.

The recording is decoded and analyzed on the user's device. This model is a
polyphonic note estimator, not a vocal separator. The monophonic decoder must
still select a lead voice, and dense mixes may follow accompaniment instead.
