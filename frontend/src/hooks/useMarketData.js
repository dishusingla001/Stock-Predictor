import { useEffect, useState } from 'react';
import { loadOverview, loadStock } from '../api.js';

export function useMarketData() {
  const initial = new URLSearchParams(window.location.search).get('stock')?.toUpperCase() || 'RELIANCE';
  const [symbol, setSymbol] = useState(initial);
  const [version, setVersion] = useState(0);
  const [retry, setRetry] = useState(0);
  const [overview, setOverview] = useState({ loading: true, error: null, stocks: [], health: null, performance: null });
  const [stock, setStock] = useState({ symbol: null, loading: true, error: null, prediction: null, history: null });
  useEffect(() => {
    const controller = new AbortController();
    setOverview(previous => ({ ...previous, loading: true, error: null }));
    loadOverview(controller.signal).then(data => {
      if (controller.signal.aborted) return;
      setOverview({ ...data, loading: false, error: null });
      setSymbol(current => data.stocks.some(item => item.symbol === current) ? current : data.stocks[0]?.symbol || '');
    }).catch(error => {
      if (!controller.signal.aborted) setOverview(previous => ({ ...previous, loading: false, error: error.message }));
    });
    return () => controller.abort();
  }, [version]);
  useEffect(() => {
    if (overview.loading || overview.error || !overview.stocks.length || !symbol) return;
    const controller = new AbortController();
    setStock({ symbol, loading: true, error: null, prediction: null, history: null });
    loadStock(symbol, controller.signal).then(data => {
      if (!controller.signal.aborted) setStock({ ...data, symbol, loading: false, error: null });
    }).catch(error => {
      if (!controller.signal.aborted) setStock({ symbol, loading: false, error: error.message, prediction: null, history: null });
    });
    return () => controller.abort();
  }, [symbol, overview.loading, overview.error, overview.stocks, retry]);
  function selectStock(value) {
    setSymbol(value);
    const url = new URL(window.location.href);
    url.searchParams.set('stock', value);
    window.history.replaceState(null, '', url);
  }
  const currentStock = stock.symbol === symbol ? stock : { symbol, loading: true, error: null, prediction: null, history: null };
  return { symbol, selectStock, overview, stock: currentStock, newsVersion: version, refresh: () => setVersion(value => value + 1), retryStock: () => setRetry(value => value + 1) };
}
