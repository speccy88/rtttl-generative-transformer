# Browser music studio: third-party notices

## RTTTL Play and RTTTL Parse — Adam Rahwane

This application reuses and adapts code from **[adamonsoon/rtttl-play](https://github.com/adamonsoon/rtttl-play)**, the player requested for this project, and its parser **[adamonsoon/rtttl-parse](https://github.com/adamonsoon/rtttl-parse)**. Both upstream `package.json` files declare the MIT license and name Adam Rahwane as author. Neither repository contains a separate license file at the revisions below.

- Player revision: `3611762347eb62eadeecbe2b55838f436a9c834a`.
- Parser revision: `4159483e64cf22dec6f11df03187183618e5219d`.
- `src/audio.js` adapts the player's oscillator-per-note playback, triangle sound, frequency assignment, and note-duration sequencing. It adds scheduling on the Web Audio clock, envelopes, rests, volume control, cancellation, progress, and a shared offline renderer.
- `src/rtttl.js` adapts the parser's section/default handling, duration formula, note interpretation, and frequency-to-pitch representation. It adds whole-token validation, bounded input, and model-compatible pitch limits.
- Exact upstream source files are preserved as `public/vendor/rtttl-play-original.js.txt` and `public/vendor/rtttl-parse-original.js.txt`, with provenance in `public/vendor/rtttl-sources.json`.

MIT License

Copyright (c) Adam Rahwane

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Optional MP3 encoder — LAME / lamejs

MP3 export uses **[@breezystack/lamejs 1.2.7](https://www.npmjs.com/package/@breezystack/lamejs)**, a packaging fork of Alex Zhukov's JavaScript port of **[LAME](https://lame.sourceforge.net/)**, under **LGPL-3.0**. The encoder is not modified. It is dynamically imported only when the user requests an MP3, and runs in a separate worker. WAV and playback do not load the MP3 encoder.

Exact corresponding source: [shijinyu/lamejs at `1fb0ef5fa177413107e2e107d054a9b994e3f79c`](https://github.com/shijinyu/lamejs/tree/1fb0ef5fa177413107e2e107d054a9b994e3f79c), matching the package registry's `gitHead`. A complete source archive is distributed alongside the deployed site at [`vendor/lamejs-1.2.7-source.tar.gz`](public/vendor/lamejs-1.2.7-source.tar.gz). The package's original license notice and the full [LGPL-3.0](public/vendor/LGPL-3.0.txt) and [GPL-3.0](public/vendor/GPL-3.0.txt) texts are distributed there too.

To rebuild or replace the encoder, extract the corresponding source archive, install its package dependencies, and run its documented build command (`npm run build`). The application's `src/mp3-worker.js` is the small integration boundary; its dynamic `@breezystack/lamejs` import can be replaced with a compatible modified module, and the app can be rebuilt with `npm run build` from `web/`. The application imposes no restriction on reverse engineering to debug modifications to the LGPL component. The source archive is offered for downloading and is not loaded during normal app startup.

Original package notice:

> Can I use LAME in my commercial program?
>
> Yes, you can, under the restrictions of the LGPL. The easiest way to do this is to:
>
> 1. Link to LAME as separate jar (lame.min.js or lame.all.js)
> 2. Fully acknowledge that you are using LAME, and give a link to our web site, lame.sourceforge.net
> 3. If you make modifications to LAME, you *must* release these these modifications back to the LAME project, under the LGPL.

The detailed LGPL text controls the terms. The rest of this repository retains its existing license.

## Browser inference libraries — Hugging Face and Microsoft

The site bundles **`@huggingface/transformers` 4.3.0**, copyright Hugging Face, under Apache-2.0. Its complete, unmodified [license](public/vendor/Transformers-js-LICENSE.txt) is distributed with the site. The bundled chat-template and tokenizer dependencies preserve their own notices: [`@huggingface/jinja` 0.5.10, MIT](public/vendor/Hugging-Face-Jinja-LICENSE.txt) and [`@huggingface/tokenizers` 0.2.0, Apache-2.0](public/vendor/Hugging-Face-Tokenizers-LICENSE.txt). These library licenses are separate from the individual model weights' licenses.

**ONNX Runtime**, copyright Microsoft Corporation, is distributed under the [MIT license](public/vendor/ONNX-Runtime-LICENSE.txt). The melody runtime uses `onnxruntime-web` 1.30.0; Transformers.js resolves its own runtime version, `1.31.0-dev.20260914-8d85527a0`. Each runtime's JavaScript and WebAssembly assets remain paired with that version.

The complete upstream third-party notices are preserved at the matching source revisions:

- [ONNX Runtime 1.30.0 notices](public/vendor/ONNX-Runtime-1.30.0-ThirdPartyNotices.txt), from [`f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7`](https://github.com/microsoft/onnxruntime/tree/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7).
- [ONNX Runtime 1.31 development notices](public/vendor/ONNX-Runtime-1.31-dev-ThirdPartyNotices.txt), from [`8d85527a010e294a26b274749f74294b2a32cec5`](https://github.com/microsoft/onnxruntime/tree/8d85527a010e294a26b274749f74294b2a32cec5).
- [Additional JavaScript dependency notices](public/vendor/ONNX-JavaScript-ThirdPartyNotices.txt) for FlatBuffers, long.js, platform.js, protobuf.js and guid-typescript included in upstream runtime distributions.

The upstream ONNX notices cover multiple build targets and are retained in full; entries for desktop-only components do not imply those components are loaded by this browser app. Exact license-file provenance and SHA-256 checksums are recorded in [`runtime-license-sources.json`](public/vendor/runtime-license-sources.json).

## Fonts

Instrument Serif and DM Sans are locally served under their SIL Open Font Licenses. Their complete license texts and source metadata are distributed under `fonts/` as `Instrument-Serif-OFL.txt`, `DM-Sans-OFL.txt`, and `sources.json`.
