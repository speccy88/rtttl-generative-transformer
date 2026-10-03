import { defineConfig } from '@playwright/test';

// Run against the built static site, including its GitHub Pages project prefix.
// `npm run build` must precede this suite, locally and in CI.
const siteOrigin = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:4173';
const sitePath = process.env.VITE_BASE_PATH || '/rtttl-generative-transformer/';
const realModel = process.env.PLAYWRIGHT_REAL_MODEL === '1';

export default defineConfig({
  testDir: './tests',
  timeout: 30000,
  expect: { timeout: 7000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['json', { outputFile: 'test-results/ui-results.json' }]],
  use: {
    baseURL: new URL(sitePath, siteOrigin).href,
    browserName: 'chromium',
    viewport: { width: 1440, height: 1000 },
    acceptDownloads: true,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    launchOptions: realModel && process.platform === 'darwin' ? {
      args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'],
    } : {},
  },
  webServer: {
    command: 'npm run preview',
    url: new URL(sitePath, siteOrigin).href,
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});
