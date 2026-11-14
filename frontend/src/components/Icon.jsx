const paths = {
  activity: <><path d="M3 13h4l3-8 4 14 3-7h4" /></>,
  grid: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
  refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 7a7 7 0 0 1 11-2l3 3M4 16l3 3a7 7 0 0 0 11-2"/></>,
  arrowUp: <path d="M6 17 18 5M6 5h12v12"/>,
  arrowDown: <path d="m6 5 12 12M6 17h12V5"/>,
  chevron: <path d="m9 5 7 7-7 7"/>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></>,
  chart: <><path d="M4 3v17h17M8 14l4-5 4 3 5-7"/></>,
  alert: <><path d="m12 3 10 18H2L12 3Z"/><path d="M12 9v5M12 17h.01"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
};
export default function Icon({ name, size = 20, className = '' }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className}>{paths[name] || paths.info}</svg>;
}
