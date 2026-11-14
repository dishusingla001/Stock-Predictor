export const RANGES = [
  { id: '1M', sessions: 22 }, { id: '3M', sessions: 66 }, { id: '6M', sessions: 126 },
  { id: '1Y', sessions: 252 }, { id: 'ALL', sessions: Infinity },
];
export function visiblePrices(prices, range) {
  const count = RANGES.find(item => item.id === range)?.sessions ?? 126;
  return prices.slice(-count);
}
export function chartGeometry(prices, width = 800, height = 290) {
  const left = 60, right = 16, top = 20, bottom = height - 78;
  const values = prices.map(point => point.close);
  const minimum = Math.min(...values), maximum = Math.max(...values);
  const padding = Math.max((maximum - minimum) * 0.15, maximum * 0.002, 1);
  const low = minimum - padding, high = maximum + padding;
  const plotWidth = Math.max(1, width - left - right);
  const x = index => left + index / Math.max(1, prices.length - 1) * plotWidth;
  const y = value => top + (high - value) / (high - low) * (bottom - top);
  const points = prices.map((point, index) => [x(index), y(point.close)]);
  const line = points.map(([a, b], index) => (index ? 'L' : 'M') + a.toFixed(2) + ',' + b.toFixed(2)).join(' ');
  const area = line + ' L' + x(prices.length - 1).toFixed(2) + ',' + bottom + ' L' + left + ',' + bottom + ' Z';
  return { left, right, top, bottom, width, height, x, y, line, area, low, high, plotWidth };
}
export function closestIndex(pointerX, geometry, count) {
  return Math.max(0, Math.min(count - 1, Math.round((pointerX - geometry.left) / geometry.plotWidth * Math.max(1, count - 1))));
}
