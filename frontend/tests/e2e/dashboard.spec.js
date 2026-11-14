import { test, expect } from '@playwright/test';

// Keep dashboard checks independent of the external provider.
test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/stocks/*/news', route => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ symbol: 'RELIANCE', fetched_at: '2026-11-14T10:00:00Z',
      is_cached: false, is_stale: false, articles: [], sentiment_counts: { positive: 0, neutral: 0, negative: 0 } }),
  }));
});

test('loads real forecasts and keeps probabilities separate from historical accuracy', async ({ page, request }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const actual = await (await request.get('/api/v1/stocks/RELIANCE/prediction')).json();
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Market overview', exact: true })).toBeVisible();
  await expect(page.getByTestId('selected-stock-name')).toHaveText('Reliance Industries');
  await expect(page.getByTestId('forecast-direction')).toHaveText(actual.direction);
  await expect(page.getByTestId('up-probability')).toHaveText((actual.up_probability * 100).toFixed(2) + '%');
  await expect(page.getByText('Probability is a model output, not measured accuracy.', { exact: false })).toBeVisible();
  await expect(page.getByText('5 trading days', { exact: false }).first()).toBeVisible();
  await expect(page.getByTestId('chart-session-count')).toHaveText('126 sessions');
  await page.screenshot({ path: 'test-results/dashboard-desktop.png', fullPage: true });
  expect(errors).toEqual([]);
});
test('switches stocks without retaining the previous forecast', async ({ page, request }) => {
  await page.goto('/');
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  const expected = await (await request.get('/api/v1/stocks/TCS/prediction')).json();
  await page.getByTestId('watchlist-TCS').click();
  await expect(page.getByTestId('selected-stock-name')).toHaveText('Tata Consultancy Services');
  await expect(page.getByTestId('forecast-direction')).toHaveText(expected.direction);
  await expect(page.getByTestId('up-probability')).toHaveText((expected.up_probability * 100).toFixed(2) + '%');
  await expect(page).toHaveURL(/stock=TCS/);
  await expect(page.getByTestId('watchlist-TCS')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('price-chart')).toHaveAttribute('aria-label', /^TCS closing-price chart/);
});
test('chart ranges and keyboard navigation select actual data points', async ({ page, request }) => {
  const actualHistory = await (await request.get('/api/v1/stocks/RELIANCE/history?limit=3000')).json();
  await page.goto('/');
  await expect(page.getByTestId('price-chart')).toBeVisible();
  await page.getByRole('button', { name: '1M', exact: true }).click();
  await expect(page.getByTestId('chart-session-count')).toHaveText('22 sessions');
  const chart = page.getByTestId('price-chart');
  await chart.focus();
  await chart.press('Home');
  const first = await page.getByTestId('chart-point-date').textContent();
  await chart.press('End');
  const last = await page.getByTestId('chart-point-date').textContent();
  expect(first).not.toBe(last);
  await chart.press('ArrowLeft');
  await expect(page.getByTestId('chart-point-date')).not.toHaveText(last);
  await page.getByRole('button', { name: 'All', exact: true }).click();
  await expect(page.getByTestId('chart-session-count')).toHaveText(actualHistory.returned_rows + ' sessions');
});
test('2026 experiment is explicitly separate from the active historical models', async ({ page, request }) => {
  const performance = await (await request.get('/api/v1/performance')).json();
  const experiment = performance.experiments.find(item => item.evaluation_kind === 'New_2026_Test');
  await page.goto('/');
  await expect(page.getByRole('tab', { name: 'Historical test', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText((performance.mean_active_historical_accuracy * 100).toFixed(2) + '%', { exact: true })).toBeVisible();
  await page.getByRole('tab', { name: '2026 experiment', exact: true }).click();
  await expect(page.getByTestId('experimental-accuracy')).toHaveText((experiment.selected_mean_accuracy * 100).toFixed(2) + '%');
  await expect(page.getByText('EXPERIMENTAL MODELS', { exact: true })).toBeVisible();
  await expect(page.getByText('These results describe separate experimental models', { exact: false })).toBeVisible();
  await expect(page.getByText((experiment.selected_mean_balanced_accuracy * 100).toFixed(2) + '%', { exact: true })).toBeVisible();
});
test('mobile layout supports stock selection with no page overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  await page.getByRole('combobox', { name: 'Select stock' }).selectOption('INFY');
  await expect(page.getByTestId('selected-stock-name')).toHaveText('Infosys');
  await expect(page.getByTestId('price-chart')).toHaveAttribute('aria-label', /^INFY closing-price chart/);
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/dashboard-mobile.png', fullPage: true });
});
test('unavailable service shows a retry state and recovers without fake prices', async ({ page }) => {
  await page.route('**/health', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'The local data service is temporarily unavailable.' }) }));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Market data is unavailable' })).toBeVisible();
  await expect(page.getByTestId('forecast-direction')).toHaveCount(0);
  await page.unroute('**/health');
  await page.getByRole('button', { name: 'Try again', exact: true }).click();
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
});
test('stock failure clears old results and a delayed previous stock cannot replace the current one', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  await page.route('**/api/v1/stocks/TCS/prediction', route => route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"This stock is temporarily unavailable."}' }));
  await page.getByTestId('watchlist-TCS').click();
  await expect(page.getByRole('heading', { name: 'Unable to analyze TCS' })).toBeVisible();
  await expect(page.getByTestId('forecast-direction')).toHaveCount(0);
  await page.unroute('**/api/v1/stocks/TCS/prediction');
  await page.getByRole('button', { name: 'Retry this stock' }).click();
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  await page.getByTestId('watchlist-RELIANCE').click();
  await expect(page.getByTestId('selected-stock-name')).toHaveText('Reliance Industries');
  await expect(page.getByTestId('forecast-direction')).toBeVisible();
  let started;
  const pending = new Promise(resolve => { started = resolve; });
  await page.route('**/api/v1/stocks/TCS/prediction', async route => {
    started();
    await new Promise(resolve => setTimeout(resolve, 700));
    await route.continue().catch(() => {});
  });
  await page.getByTestId('watchlist-TCS').click();
  await pending;
  await page.getByTestId('watchlist-INFY').click();
  await expect(page.getByTestId('selected-stock-name')).toHaveText('Infosys');
  await expect(page.getByTestId('price-chart')).toHaveAttribute('aria-label', /^INFY closing-price chart/);
  await page.waitForTimeout(800);
  await expect(page.getByTestId('selected-stock-name')).toHaveText('Infosys');
  await expect(page.getByTestId('price-chart')).toHaveAttribute('aria-label', /^INFY closing-price chart/);
});
