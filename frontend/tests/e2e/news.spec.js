import { test, expect } from '@playwright/test';

function news(symbol = 'RELIANCE', overrides = {}) {
  return {
    symbol, provider: 'Google News RSS', window_days: 7,
    fetched_at: '2026-11-14T10:00:00Z', is_cached: false, is_stale: false,
    articles: [{
      id: symbol + '-article', title: symbol + ' reports excellent growth',
      source: 'Example Financial News', url: 'https://example.com/' + symbol,
      published_at: '2026-11-14T09:00:00Z', sentiment: 'positive', sentiment_score: 0.7,
    }],
    sentiment_counts: { positive: 1, neutral: 0, negative: 0 }, detail: null,
    ...overrides,
  };
}
async function respond(route, payload, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(payload) });
}
test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/stocks/*/news', route => respond(route, news(route.request().url().split('/').at(-2))));
});

test('shows dated linked headlines with tone labels separately from forecast probabilities', async ({ page }) => {
  await page.goto('/');
  const panel = page.getByRole('region', { name: 'News & sentiment' });
  await expect(panel.getByRole('link', { name: /RELIANCE reports excellent growth/ })).toHaveAttribute('href', 'https://example.com/RELIANCE');
  await expect(panel.getByText('Example Financial News', { exact: true })).toBeVisible();
  await expect(panel.locator('time[datetime="2026-11-14T09:00:00Z"]')).toContainText('14:30');
  await expect(panel.getByText('Sentiment is separate from the forecast', { exact: false })).toBeVisible();
  await expect(panel.getByTestId('news-item').getByText('positive', { exact: true })).toBeVisible();
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  const probability = await page.getByTestId('up-probability').textContent();
  await panel.getByRole('button', { name: 'Refresh stock news' }).click();
  await expect(panel.getByTestId('news-item')).toHaveCount(1);
  await expect(page.getByTestId('up-probability')).toHaveText(probability);
  await panel.screenshot({ path: 'test-results/news-panel-desktop.png' });
});

test('news outage has a retry state while forecasts remain usable', async ({ page }) => {
  await page.route('**/api/v1/stocks/RELIANCE/news', route => respond(route, { detail: 'Provider temporarily offline.' }, 503));
  await page.goto('/');
  const panel = page.getByRole('region', { name: 'News & sentiment' });
  await expect(panel.getByText('News is unavailable', { exact: true })).toBeVisible();
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  await page.unroute('**/api/v1/stocks/RELIANCE/news');
  await panel.getByRole('button', { name: 'Retry news', exact: true }).click();
  await expect(panel.getByTestId('news-item')).toHaveCount(1);
});

test('empty and stale responses are explicit and mobile headlines do not overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route('**/api/v1/stocks/RELIANCE/news', route => respond(route, news('RELIANCE', { articles: [], sentiment_counts: { positive: 0, neutral: 0, negative: 0 } })));
  await page.goto('/');
  const panel = page.getByRole('region', { name: 'News & sentiment' });
  await expect(panel.getByText('No matching headlines', { exact: false })).toBeVisible();
  await page.route('**/api/v1/stocks/TCS/news', route => respond(route, news('TCS', {
    is_cached: true, is_stale: true, detail: 'News refresh failed. Showing the last successful fetch.',
    articles: [{ ...news('TCS').articles[0], title: 'TCS ' + 'longword'.repeat(40) }],
  })));
  await page.getByRole('combobox', { name: 'Select stock' }).selectOption('TCS');
  await expect(panel.getByText('Stale cache', { exact: false })).toBeVisible();
  await expect(panel.getByText('Cached headlines may be out of date.', { exact: false })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('late previous-stock news cannot replace the current stock and loading clears old headlines', async ({ page }) => {
  await page.goto('/');
  const panel = page.getByRole('region', { name: 'News & sentiment' });
  await expect(panel.getByTestId('news-item')).toHaveCount(1);
  let started, release;
  const pending = new Promise(resolve => { started = resolve; });
  const blocked = new Promise(resolve => { release = resolve; });
  await page.route('**/api/v1/stocks/TCS/news', async route => {
    started();
    await blocked;
    await respond(route, news('TCS')).catch(() => {});
  });
  await page.getByTestId('watchlist-TCS').click();
  await pending;
  await expect(panel.getByText('Loading headlines for TCS...')).toBeVisible();
  await expect(panel.getByTestId('news-item')).toHaveCount(0);
  await page.getByTestId('watchlist-INFY').click();
  await expect(panel.getByRole('link', { name: /INFY reports excellent growth/ })).toBeVisible();
  release();
  await expect(panel.getByRole('link', { name: /TCS reports excellent growth/ })).toHaveCount(0);
});

test('global refresh also reloads news', async ({ page }) => {
  let count = 0;
  await page.route('**/api/v1/stocks/RELIANCE/news', route => {
    count += 1;
    return respond(route, news());
  });
  await page.goto('/');
  await expect(page.getByTestId('news-item')).toHaveCount(1);
  const before = count;
  await page.getByRole('button', { name: 'Refresh market data' }).click();
  await expect.poll(() => count).toBeGreaterThan(before);
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
});
