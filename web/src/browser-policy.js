// WebKit can crash the entire page while tearing down ORT's WebGPU device in
// a dedicated worker, even after session.dispose() and GPUDevice.destroy().
// Keep local inference on CPU in this engine until that lifecycle is reliable.
// Chrome/Firefox on iOS also use WebKit, despite their different browser names.
export function isWebKitEngine(userAgent = globalThis.navigator?.userAgent || '') {
  return /AppleWebKit\//i.test(userAgent) && !/(?:Chrome|Chromium|Edg|OPR)\//i.test(userAgent);
}

export const WEBKIT_CPU_REASON = 'This browser uses local CPU inference to avoid a WebKit GPU cleanup crash.';
