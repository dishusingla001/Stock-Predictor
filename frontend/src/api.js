const BASE = (import.meta.env?.VITE_API_BASE_URL || '').replace(/\/$/, '');
export class ApiError extends Error {}
export async function request(path, { signal, fetcher = globalThis.fetch, timeout = 12000 } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  if (signal?.aborted) cancel();
  signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
  try {
    const response = await fetcher(BASE + path, { signal: controller.signal, headers: { Accept: 'application/json' } });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new ApiError(typeof body?.detail === 'string' ? body.detail : 'The local data service could not complete this request.');
    }
    if (!body) throw new ApiError('The data service returned an unreadable response.');
    return body;
  } catch (error) {
    if (signal?.aborted) throw new DOMException('Request cancelled', 'AbortError');
    if (timedOut) throw new ApiError('The data service took too long to respond. Please try again.');
    if (error instanceof ApiError) throw error;
    throw new ApiError('Unable to reach the local data service. Check that it is running and try again.');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
export async function loadOverview(signal) {
  const [health, stocks, performance] = await Promise.all([
    request('/health', { signal }), request('/api/v1/stocks', { signal }), request('/api/v1/performance', { signal }),
  ]);
  return { health, stocks: stocks.stocks, performance };
}
export async function loadStock(symbol, signal) {
  const prefix = '/api/v1/stocks/' + encodeURIComponent(symbol);
  const [prediction, history] = await Promise.all([
    request(prefix + '/prediction', { signal }), request(prefix + '/history?limit=3000', { signal }),
  ]);
  return { prediction, history };
}

export async function loadNews(symbol, signal) {
  return request('/api/v1/stocks/' + encodeURIComponent(symbol) + '/news', { signal, timeout: 10000 });
}
