// The LGPL encoder is isolated in a lazily loaded module; see THIRD_PARTY_NOTICES.md.
export async function encodeMp3Pcm(samples, sampleRate, onProgress = () => {}) {
  if (!(samples instanceof Int16Array) || samples.length === 0) throw new Error('MP3 input must be nonempty 16-bit PCM.');
  if (sampleRate !== 44100) throw new Error('MP3 export expects 44.1 kHz audio.');
  const { Mp3Encoder } = await import('@breezystack/lamejs');
  const encoder = new Mp3Encoder(1, sampleRate, 128);
  const chunks = [];
  const blockSize = 1152;
  for (let offset = 0; offset < samples.length; offset += blockSize) {
    const encoded = encoder.encodeBuffer(samples.subarray(offset, offset + blockSize));
    if (encoded.length) chunks.push(new Uint8Array(encoded));
    if (offset % (blockSize * 64) === 0) onProgress(offset / samples.length);
  }
  const last = encoder.flush();
  if (last.length) chunks.push(new Uint8Array(last));
  onProgress(1);
  return new Blob(chunks, { type: 'audio/mpeg' });
}

if (typeof self !== 'undefined' && typeof document === 'undefined') {
  self.onmessage = async ({ data }) => {
    try {
      const blob = await encodeMp3Pcm(data.samples, data.sampleRate, progress => self.postMessage({ type: 'progress', progress }));
      self.postMessage({ type: 'done', blob });
    } catch (error) {
      self.postMessage({ type: 'error', message: error.message || 'MP3 export failed.' });
    }
  };
}
