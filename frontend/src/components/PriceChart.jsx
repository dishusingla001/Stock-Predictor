import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { chartGeometry, closestIndex, RANGES, visiblePrices } from '../chart-utils.js';
import { compact, dateLabel, money, percent, signed } from '../format.js';

export default function PriceChart({ history, symbol }) {
  const [range, setRange] = useState('6M');
  const [hover, setHover] = useState(null);
  const [width, setWidth] = useState(800);
  const wrapper = useRef(null);
  const gradient = useId().replaceAll(':', '');
  const prices = useMemo(() => visiblePrices(history.prices, range), [history, range]);
  const geometry = useMemo(() => prices.length ? chartGeometry(prices, width) : null, [prices, width]);
  useEffect(() => {
    const observer = new ResizeObserver(entries => setWidth(Math.max(280, Math.round(entries[0].contentRect.width))));
    observer.observe(wrapper.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => setHover(null), [range, symbol]);
  const index = hover === null ? prices.length - 1 : Math.min(hover, prices.length - 1);
  const point = prices[index];
  const change = prices.length > 1 ? (prices.at(-1).close - prices[0].close) / prices[0].close : 0;
  const maxVolume = Math.max(1, ...prices.map(item => item.volume));
  function keyDown(event) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') setHover(0);
    else if (event.key === 'End') setHover(prices.length - 1);
    else setHover(current => Math.max(0, Math.min(prices.length - 1, (current ?? prices.length - 1) + (event.key === 'ArrowLeft' ? -1 : 1))));
  }
  return <section className="card chart-card" aria-labelledby="history-title">
    <div className="card-heading chart-heading">
      <div><h2 id="history-title">Price history</h2><p>Daily closing price · INR</p></div>
      <div className="range-selector" aria-label="Chart time range">{RANGES.map(item => <button key={item.id} type="button" aria-pressed={range === item.id} onClick={() => setRange(item.id)}>{item.id === 'ALL' ? 'All' : item.id}</button>)}</div>
    </div>
    <div className="chart-readout">
      <span data-testid="chart-point-price">{point ? money(point.close) : 'No prices available'}</span>
      <span className={change >= 0 ? 'positive' : 'negative'}>{prices.length > 1 ? signed(change) + percent(Math.abs(change)) + ' in this view' : 'One available session'}</span>
      <time data-testid="chart-point-date">{point ? dateLabel(point.date) : '—'}</time>
    </div>
    <div ref={wrapper} className="chart-wrapper">
      {geometry ? <svg data-testid="price-chart" className="price-chart" width="100%" height="290" viewBox={'0 0 ' + width + ' 290'} tabIndex="0" role="img"
        aria-label={symbol + ' closing-price chart. Use left and right arrow keys to explore sessions.'}
        onKeyDown={keyDown} onPointerLeave={() => setHover(null)}
        onPointerMove={event => {
          const bounds = event.currentTarget.getBoundingClientRect();
          setHover(closestIndex((event.clientX - bounds.left) * width / bounds.width, geometry, prices.length));
        }}>
        <title>{symbol} daily closing prices from {prices[0].date} to {prices.at(-1).date}</title>
        <defs><linearGradient id={gradient} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#287756" stopOpacity=".17"/><stop offset="100%" stopColor="#287756" stopOpacity=".015"/></linearGradient></defs>
        {[0, 1, 2, 3].map(tick => {
          const value = geometry.low + (geometry.high - geometry.low) * tick / 3;
          const y = geometry.y(value);
          return <g key={tick}><line x1={geometry.left} x2={width - geometry.right} y1={y} y2={y} stroke="#e7ece6" strokeDasharray="3 5"/><text x={geometry.left - 12} y={y + 4} textAnchor="end" className="axis-label">{Math.round(value).toLocaleString('en-IN')}</text></g>;
        })}
        <path d={geometry.area} fill={'url(#' + gradient + ')'}/>
        <path d={geometry.line} stroke="#277455" fill="none" strokeWidth="2.2" strokeLinejoin="round"/>
        {prices.map((item, i) => <rect key={item.date} x={geometry.x(i) - Math.max(.5, geometry.plotWidth / prices.length * .28)} y={258 - item.volume / maxVolume * 25}
          width={Math.max(1, geometry.plotWidth / prices.length * .56)} height={Math.max(1, item.volume / maxVolume * 25)} fill={i === index ? '#5d977b' : '#dbe5dd'} rx=".7"/>)}
        {[0, .25, .5, .75, 1].filter((tick, i) => width > 480 || i % 2 === 0).map(tick => {
          const i = Math.round(tick * (prices.length - 1));
          return <text key={tick} x={geometry.x(i)} y="283" textAnchor={tick === 0 ? 'start' : tick === 1 ? 'end' : 'middle'} className="axis-label">{dateLabel(prices[i].date, { year: undefined })}</text>;
        })}
        {point && <g><line x1={geometry.x(index)} x2={geometry.x(index)} y1={geometry.top} y2="260" stroke="#83a58f" strokeDasharray="4 4"/><circle cx={geometry.x(index)} cy={geometry.y(point.close)} r="5" fill="white" stroke="#277455" strokeWidth="2.5"/></g>}
      </svg> : <div className="chart-empty">No price history is available for this stock.</div>}
    </div>
    <div className="chart-footer"><span><i className="legend-dot"/>Closing price <i className="legend-square"/>Volume {point ? compact(point.volume) : '—'}</span><span data-testid="chart-session-count">{prices.length} sessions</span></div>
  </section>;
}
