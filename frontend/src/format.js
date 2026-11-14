const moneyFormatter = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', minimumFractionDigits: 2, maximumFractionDigits: 2 });
const numberFormatter = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 });
export const money = value => Number.isFinite(value) ? moneyFormatter.format(value) : '—';
export const number = value => Number.isFinite(value) ? numberFormatter.format(value) : '—';
export const percent = (value, digits = 2) => Number.isFinite(value) ? (value * 100).toFixed(digits) + '%' : '—';
export const compact = value => Number.isFinite(value) ? new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) : '—';
export function dateLabel(value, options = {}) {
  if (!value) return '—';
  const day = new Date(value.length === 10 ? value + 'T00:00:00Z' : value);
  if (Number.isNaN(day.getTime())) return '—';
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC', ...options }).format(day);
}
export function priceChange(prices) {
  if (!prices || prices.length < 2) return null;
  const previous = prices.at(-2).close;
  const current = prices.at(-1).close;
  return { amount: current - previous, fraction: previous ? (current - previous) / previous : 0 };
}
export const signed = value => value > 0 ? '+' : value < 0 ? '−' : '';

export function newsTime(value) {
  const timestamp = new Date(value);
  if (!value || Number.isNaN(timestamp.getTime())) return 'Unknown time';
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
    hourCycle: 'h23', timeZone: 'Asia/Kolkata',
  }).format(timestamp) + ' IST';
}
