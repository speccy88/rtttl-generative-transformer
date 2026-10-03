import test from 'node:test';
import assert from 'node:assert/strict';
import { isWebKitEngine } from './browser-policy.js';

test('WebKit worker compatibility includes Safari and other iOS browser shells', () => {
  for (const suffix of ['Version/26.0 Mobile/15E148 Safari/604.1', 'CriOS/140.0 Mobile/15E148 Safari/604.1', 'FxiOS/143.0 Mobile/15E148 Safari/605.1.15', 'EdgiOS/140.0 Mobile/15E148 Safari/605.1.15']) {
    assert.equal(isWebKitEngine(`Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 ${suffix}`), true);
  }
  assert.equal(isWebKitEngine('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/26.0 Safari/605.1.15'), true);
});

test('Chromium and Gecko keep hardware GPU support, including on the Mac', () => {
  for (const browser of ['Chrome/140.0.0.0', 'Chromium/140.0.0.0', 'Chrome/140.0.0.0 Edg/140.0', 'Chrome/140.0.0.0 OPR/122.0']) {
    assert.equal(isWebKitEngine(`Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ${browser} Safari/537.36`), false);
  }
  assert.equal(isWebKitEngine('Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0'), false);
});
