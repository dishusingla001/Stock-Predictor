import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';
const systemEdge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || (existsSync(systemEdge) ? systemEdge : undefined);
const baseURL = process.env.E2E_BASE_URL || 'http://127.0.0.1:5173';
export default defineConfig({
  testDir: './tests/e2e', fullyParallel: false, workers: 1, timeout: 30000,
  reporter: 'list', outputDir: 'test-results',
  use: { baseURL, browserName: 'chromium', headless: true, viewport: { width: 1440, height: 1080 },
    launchOptions: executablePath ? { executablePath } : {}, screenshot: 'only-on-failure' },
  webServer: process.env.E2E_BASE_URL ? undefined : {
    command: 'npm run dev', url: baseURL, reuseExistingServer: !process.env.CI, timeout: 30000,
  },
});
