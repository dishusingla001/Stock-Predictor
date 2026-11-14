import { useState } from 'react';
import Icon from './components/Icon.jsx';
import PriceChart from './components/PriceChart.jsx';
import NewsPanel from './components/NewsPanel.jsx';
import { useMarketData } from './hooks/useMarketData.js';
import { compact, dateLabel, money, number, percent, priceChange, signed } from './format.js';

function EmptyState({ title, message, onRetry, button = 'Try again' }) {
  return <section className="card empty-state" role="alert">
    <span className="empty-icon"><Icon name="alert" size={26}/></span>
    <h2>{title}</h2><p>{message}</p>
    <button className="button button-primary" type="button" onClick={onRetry}><Icon name="refresh" size={16}/>{button}</button>
  </section>;
}
function Skeleton() {
  return <div className="loading-workspace" aria-label="Loading stock analysis" aria-busy="true">
    <div className="summary-grid">{[1, 2, 3].map(item => <div className="card skeleton-stat" key={item}><span className="skeleton short"/><span className="skeleton long"/><span className="skeleton medium"/></div>)}</div>
    <div className="analysis-grid"><div className="card skeleton-chart"><span className="skeleton short"/><div className="skeleton skeleton-plot"/></div><div className="card skeleton-forecast"><span className="skeleton medium"/><span className="skeleton long"/></div></div>
  </div>;
}
function Sidebar({ overview, symbol, selectStock }) {
  return <aside className="sidebar" aria-label="Workspace navigation">
    <a className="brand" href="#overview"><span className="brand-mark"><Icon name="activity" size={25}/></span><span>StockSight<span className="brand-caption">RESEARCH WORKSPACE</span></span></a>
    <div className="sidebar-label">WORKSPACE</div>
    <nav className="workspace-nav">
      <a className="nav-item nav-primary" href="#overview"><Icon name="grid" size={18}/>Overview<span className="nav-pill">5D</span></a>
      <a className="nav-item" href="#history-title"><Icon name="chart" size={18}/>Price history</a>
      <a className="nav-item" href="#news"><Icon name="info" size={18}/>News &amp; sentiment</a>
      <a className="nav-item" href="#evaluation"><Icon name="activity" size={18}/>Model evaluation</a>
    </nav>
    <div className="watchlist-heading"><span className="sidebar-label">YOUR WATCHLIST</span><span>{overview.stocks.length || '—'}</span></div>
    <div className="watchlist">{overview.stocks.map(item => <button key={item.symbol} type="button" data-testid={'watchlist-' + item.symbol}
      className={'watchlist-item ' + (symbol === item.symbol ? 'selected' : '')} aria-pressed={symbol === item.symbol} onClick={() => selectStock(item.symbol)}>
      <span className={'ticker-monogram ticker-' + item.symbol}>{item.symbol === 'RELIANCE' ? 'R' : item.symbol === 'TCS' ? 'T' : item.symbol === 'INFY' ? 'I' : item.symbol === 'HDFCBANK' ? 'H' : 'IC'}</span>
      <span className="watchlist-copy"><strong>{item.symbol}</strong><span>{money(item.close)}</span></span>
      <Icon name="chevron" size={14}/>
    </button>)}</div>
    <div className="sidebar-bottom">
      <span className="local-label"><span className={'status-dot ' + (overview.error ? 'dot-warning' : '')}/>{overview.health ? 'LOCAL SNAPSHOT' : 'LOCAL WORKSPACE'}</span>
      <strong>{overview.health ? dateLabel(overview.health.latest_data_date) : 'Connecting to data'}</strong>
      <p>Five NSE equities.<br/>One clearer perspective.</p>
    </div>
  </aside>;
}
function ForecastCard({ prediction }) {
  const isUp = prediction.direction === 'UP';
  return <section className="forecast-card" aria-labelledby="forecast-title">
    <div className="forecast-heading"><h2 id="forecast-title">5-session forecast</h2><span className="forecast-badge">ACTIVE MODEL</span></div>
    <div className={'forecast-direction ' + (isUp ? 'forecast-up' : 'forecast-down')}><span className="forecast-arrow"><Icon name={isUp ? 'arrowUp' : 'arrowDown'} size={30}/></span><strong data-testid="forecast-direction">{prediction.direction}</strong><span className="pending-label">Outcome pending</span></div>
    <p className="forecast-comparison">Expected closing direction relative to {money(prediction.close)} on {dateLabel(prediction.data_date)}.</p>
    <div className="probability-heading"><span>Model probabilities</span><Icon name="info" size={15}/></div>
    <div className="probability-bar" aria-label={'UP ' + percent(prediction.up_probability) + ', DOWN ' + percent(prediction.down_probability)}>
      <span style={{ width: percent(prediction.up_probability, 6) }}/><span style={{ width: percent(prediction.down_probability, 6) }}/>
    </div>
    <div className="probability-labels"><span><i/>P(UP)<strong data-testid="up-probability">{percent(prediction.up_probability)}</strong></span><span><i/>P(DOWN)<strong>{percent(prediction.down_probability)}</strong></span></div>
    <div className="forecast-details">
      <div><span>Model</span><strong>{prediction.model}</strong></div>
      <div><span>Trained through</span><strong>{dateLabel(prediction.model_trained_through)}</strong></div>
      <div><span>Forecast horizon</span><strong>{prediction.horizon_trading_days} trading days</strong></div>
    </div>
    <p className="threshold-note">UP is selected at P(UP) ≥ {percent(prediction.decision_threshold, 1)}.</p>
    <div className="forecast-note"><Icon name="info" size={16}/><p>Probability is a model output, not measured accuracy. Weekends and market holidays do not count toward the horizon.</p></div>
  </section>;
}
function Indicators({ prediction }) {
  const values = prediction.indicators;
  const items = [
    ['RSI', 'RSI · 14 sessions', number(values.RSI), 'Momentum'],
    ['MACD', 'MACD', number(values.MACD), 'Trend difference'],
    ['ROC', 'ROC · 12 sessions', number(values.ROC) + '%', 'Rate of change'],
    ['ATR', 'ATR · 14 sessions', money(values.ATR), 'Trading range'],
    ['EMA20', 'EMA · 20 sessions', money(values.EMA20), 'Exponential average'],
    ['EMA50', 'EMA · 50 sessions', money(values.EMA50), 'Exponential average'],
    ['Volume', 'Trading volume', compact(values.Volume), 'Shares in the latest session'],
  ].filter(([key]) => Number.isFinite(values[key]));
  return <section className="card indicators-card" aria-labelledby="indicators-title">
    <div className="card-heading"><div><h2 id="indicators-title">What the model sees</h2><p>Technical indicators at the latest data date.</p></div><span className="quiet-tag">{dateLabel(prediction.data_date)}</span></div>
    <div className="indicators-grid">{items.map(([key, title, value, subtitle]) => <div className={'indicator ' + (key === 'Volume' ? 'volume-indicator' : '')} key={key}><span>{title}</span><strong title={String(values[key])}>{value}</strong><small>{subtitle}</small></div>)}</div>
  </section>;
}
function Evaluation({ performance, symbol, selectStock }) {
  const [tab, setTab] = useState('historical');
  if (!performance) return null;
  const rows = performance.active_models;
  const first = rows[0];
  const experiment = performance.experiments.find(item => item.evaluation_kind === 'New_2026_Test');
  const change = experiment ? experiment.selected_mean_accuracy - experiment.stock_only_mean_accuracy : 0;
  return <section id="evaluation" className="card evaluation-card" aria-labelledby="evaluation-title">
    <div className="card-heading evaluation-heading"><div><span className="eyebrow">MEASURE THE EVIDENCE</span><h2 id="evaluation-title">Model evaluation</h2><p>Measured performance, separate from today’s forecast.</p></div>
      <div role="tablist" aria-label="Evaluation scope" className="evaluation-tabs">
        <button role="tab" id="historical-tab" aria-selected={tab === 'historical'} aria-controls="historical-panel" onClick={() => setTab('historical')}>Historical test</button>
        {experiment && <button role="tab" id="experiment-tab" aria-selected={tab === 'experiment'} aria-controls="experiment-panel" onClick={() => setTab('experiment')}>2026 experiment</button>}
      </div>
    </div>
    {tab === 'historical' ? <div id="historical-panel" role="tabpanel" aria-labelledby="historical-tab">
      <div className="evaluation-context"><span className="scope-badge">ACTIVE MODELS</span><span>{first ? dateLabel(first.start) + ' – ' + dateLabel(first.end) : 'No evaluation available'}</span><span className="evaluation-average">Average accuracy <strong>{percent(performance.mean_active_historical_accuracy)}</strong></span></div>
      <div className="table-scroll"><table><caption className="sr-only">Historical test results for the active models. Probabilities in the forecast panel are not accuracy scores.</caption><thead><tr><th scope="col">Stock</th><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Training-majority baseline</th><th scope="col">ROC-AUC</th></tr></thead>
        <tbody>{rows.map(row => <tr key={row.symbol} className={row.symbol === symbol ? 'selected-row' : ''}><th scope="row"><button type="button" onClick={() => selectStock(row.symbol)} aria-label={'Analyze ' + row.symbol}>{row.symbol}{row.symbol === symbol && <span className="table-selected-dot"/>}</button></th><td>{row.model}<small>{row.samples.toLocaleString('en-IN')} test forecasts</small></td><td className="table-number">{percent(row.accuracy)}</td><td>{percent(row.training_majority_baseline_accuracy)}</td><td>{row.roc_auc.toFixed(3)}</td></tr>)}</tbody>
      </table></div>
      <div className="evaluation-note"><Icon name="info" size={17}/><p>This historical period has already been inspected during development. The baseline predicts the constant class chosen from training data. These scores do not measure the latest forecast’s accuracy.</p></div>
    </div> : experiment && <div id="experiment-panel" role="tabpanel" aria-labelledby="experiment-tab">
      <div className="evaluation-context"><span className="scope-badge experimental-badge">EXPERIMENTAL MODELS</span><span>{dateLabel(experiment.start)} – {dateLabel(experiment.end)}</span><span className={'evaluation-average ' + (change >= 0 ? 'positive' : 'negative')}>{signed(change)}{Math.abs(change * 100).toFixed(2)} percentage points</span></div>
      <div className="experiment-metrics">
        <div><span>Stock-only inputs</span><strong>{percent(experiment.stock_only_mean_accuracy)}</strong><small>Matched-period control</small></div>
        <div><span>Validation-selected inputs</span><strong data-testid="experimental-accuracy">{percent(experiment.selected_mean_accuracy)}</strong><small>Mean test accuracy across stocks</small></div>
        <div><span>Balanced accuracy</span><strong>{percent(experiment.selected_mean_balanced_accuracy)}</strong><small>Equal weight for UP and DOWN classes</small></div>
      </div>
      <div className="evaluation-note experiment-note"><Icon name="info" size={17}/><p>These results describe separate experimental models, not the active models or the latest refitted snapshots. The 2026 models were fitted on pre-2026 outcomes. Feature choices were fixed before evaluation.</p></div>
    </div>}
  </section>;
}
export default function App() {
  const { symbol, selectStock, overview, stock, refresh, retryStock, newsVersion } = useMarketData();
  const current = overview.stocks.find(item => item.symbol === symbol);
  const change = priceChange(stock.history?.prices);
  const busy = overview.loading || stock.loading;
  return <div className="app-shell">
    <a className="skip-link" href="#main-content">Skip to market analysis</a>
    <Sidebar overview={overview} symbol={symbol} selectStock={selectStock}/>
    <div className="workspace">
      <header className="topbar"><div className="breadcrumb"><span>WORKSPACE</span><span>/</span><strong>Overview</strong></div>
        <div className="topbar-actions"><span className={'connection-state ' + (overview.error ? 'connection-error' : '')}><i className={'status-dot ' + (overview.error ? 'dot-warning' : '')}/>{overview.error ? 'Data unavailable' : overview.loading ? 'Connecting' : 'Snapshot ready'}</span><button className="button button-secondary refresh-button" type="button" onClick={refresh} disabled={busy && !overview.error && !stock.error} aria-label="Refresh market data" title="Reload the available local snapshot"><Icon name="refresh" size={16} className={busy ? 'spinning' : ''}/><span>Refresh view</span></button></div>
      </header>
      <main id="main-content">
        <section className="page-intro" id="overview"><div><span className="eyebrow">A CLEARER VIEW OF THE MARKET</span><h1>Market overview</h1><p>Price history, model signals, and the evidence behind them.</p></div><span className="exchange-label"><span className="exchange-mark"/>NSE <span>·</span> INR</span></section>
        {overview.error ? <EmptyState title="Market data is unavailable" message={overview.error} onRetry={refresh}/> : <>
          <section className="stock-heading-row" aria-label="Selected stock">
            <div><div className="stock-heading"><span className="symbol-pill">{current?.symbol || 'NSE'}</span><h2 data-testid="selected-stock-name">{current?.name || 'Loading your workspace'}</h2></div><p><Icon name="clock" size={14}/>{current ? 'Data as of ' + dateLabel(current.data_date) : 'Connecting to the local snapshot'}{current?.is_stale && <span className="stale-tag">Older data snapshot</span>}</p></div>
            <label className="stock-select"><span>Select stock</span><select aria-label="Select stock" value={current ? symbol : ''} onChange={event => selectStock(event.target.value)} disabled={!overview.stocks.length}>{!current && <option value="">Loading stocks…</option>}{overview.stocks.map(item => <option value={item.symbol} key={item.symbol}>{item.symbol}</option>)}</select></label>
          </section>
          {overview.loading || stock.loading ? <Skeleton/> : stock.error ? <EmptyState title={'Unable to analyze ' + symbol} message={stock.error} onRetry={retryStock} button="Retry this stock"/> : stock.prediction && stock.history && <>
            <div className="summary-grid">
              <section className="card summary-card"><span>Last closing price</span><strong data-testid="closing-price">{money(stock.prediction.close)}</strong><small>{dateLabel(stock.prediction.data_date)} · INR</small></section>
              <section className="card summary-card"><span>Previous-session change</span><strong className={change && change.amount < 0 ? 'negative' : 'positive'}>{change ? signed(change.amount) + money(Math.abs(change.amount)) : '—'}<small>{change ? signed(change.fraction) + percent(Math.abs(change.fraction)) : ''}</small></strong><small>Compared with the previous available session</small></section>
              <section className="card summary-card"><span>Forecast horizon</span><strong>5 <em>trading days</em></strong><small>One UP or DOWN closing-direction forecast</small></section>
            </div>
            <div className="analysis-grid"><PriceChart key={symbol} history={stock.history} symbol={symbol}/><ForecastCard prediction={stock.prediction}/></div>
            <Indicators prediction={stock.prediction}/>
          </>}
          <NewsPanel symbol={symbol} refreshKey={newsVersion}/>
          <Evaluation performance={overview.performance} symbol={symbol} selectStock={selectStock}/>
        </>}
        <footer className="page-footer"><span><Icon name="activity" size={15}/>StockSight <span>·</span> Research workspace</span><span>Daily snapshots · Horizons count trading sessions</span></footer>
      </main>
    </div>
  </div>;
}
