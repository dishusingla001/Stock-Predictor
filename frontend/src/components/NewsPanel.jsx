import Icon from './Icon.jsx';
import { useNews } from '../hooks/useNews.js';
import { newsTime } from '../format.js';

export default function NewsPanel({ symbol, refreshKey }) {
  const { loading, data, error, retry } = useNews(symbol, refreshKey);
  return <section id="news" className="card news-card" aria-labelledby="news-title" aria-busy={loading}>
    <div className="card-heading news-heading">
      <div><span className="eyebrow">HEADLINE CONTEXT</span><h2 id="news-title">News &amp; sentiment</h2><p>Recent English-language coverage for {symbol}.</p></div>
      <button type="button" className="button button-secondary" onClick={retry} disabled={loading} aria-label="Refresh stock news">
        <Icon name="refresh" size={15}/>{loading ? 'Loading news' : 'Refresh news'}
      </button>
    </div>
    <p className="news-method">VADER labels headline tone only. Sentiment is separate from the forecast and does not change its probabilities. Financial context may be misread.</p>
    {loading ? <div className="news-state" role="status">Loading headlines for {symbol}...</div>
      : error ? <div className="news-state" role="alert"><strong>News is unavailable</strong><p>{error}</p><button type="button" className="button button-secondary" onClick={retry}>Retry news</button></div>
      : data && <>
        <div className="news-meta">
          <span>Google News RSS &middot; Last 7 days &middot; {data.articles.length} headlines</span>
          <span>{data.is_stale ? 'Stale cache' : data.is_cached ? 'Cached' : 'Fetched'} &middot; <time dateTime={data.fetched_at}>{newsTime(data.fetched_at)}</time></span>
        </div>
        {data.is_stale && <p className="news-stale" role="status">{data.detail} Cached headlines may be out of date.</p>}
        {data.articles.length ? <>
          <div className="news-counts" aria-label="Headline sentiment counts">
            {['positive', 'neutral', 'negative'].map(label => <span key={label} className={'sentiment-badge sentiment-' + label}>{label} <strong>{data.sentiment_counts[label]}</strong></span>)}
          </div>
          <ul className="news-list">{data.articles.map(article => <li key={article.id} className="news-item" data-testid="news-item">
            <div className="news-item-copy">
              <a href={article.url} target="_blank" rel="noopener noreferrer" referrerPolicy="no-referrer">{article.title}<span className="sr-only"> (opens in a new tab)</span></a>
              <div className="news-byline"><span>{article.source}</span><span aria-hidden="true">&middot;</span><time dateTime={article.published_at}>{newsTime(article.published_at)}</time></div>
            </div>
            <span className={'sentiment-badge sentiment-' + article.sentiment}>{article.sentiment}</span>
          </li>)}</ul>
        </> : <div className="news-state" role="status">No matching headlines were found in the last seven days. Try again later.</div>}
        <p className="news-footnote">Headlines link through Google News to the publisher. Labels use the headline, not the full article; they are not confidence scores or trading recommendations. Results are cached for 15 minutes.</p>
      </>}
  </section>;
}
