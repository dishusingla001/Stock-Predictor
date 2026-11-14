import { useEffect, useState } from 'react';
import { loadNews } from '../api.js';

export function useNews(symbol, refreshKey) {
  const [version, setVersion] = useState(0);
  const [state, setState] = useState({ symbol: null, loading: true, data: null, error: null });
  useEffect(() => {
    if (!symbol) return;
    const controller = new AbortController();
    setState({ symbol, loading: true, data: null, error: null });
    loadNews(symbol, controller.signal).then(data => {
      if (!controller.signal.aborted) setState({ symbol, loading: false, data, error: null });
    }).catch(error => {
      if (!controller.signal.aborted) setState({ symbol, loading: false, data: null, error: error.message });
    });
    return () => controller.abort();
  }, [symbol, version, refreshKey]);
  return {
    ...(state.symbol === symbol ? state : { symbol, loading: true, data: null, error: null }),
    retry: () => setVersion(value => value + 1),
  };
}
