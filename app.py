"""
Monte Carlo Stock Price Simulator
Streamlit app — UI rendered via components.html for full design control
"""

import streamlit as st
import yfinance as yf
import numpy as np
import json

st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide all Streamlit chrome
st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"],
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #07080b !important; }
iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str):
    data = yf.download(ticker, period="5y", interval="1d",
                       auto_adjust=True, progress=False)
    if data.empty:
        return None, f"No data found for '{ticker}'. Check the symbol and try again."
    closes  = data["Close"].dropna().squeeze()
    returns = closes.pct_change().dropna()
    def s(x): return float(x.iloc[0]) if hasattr(x, "iloc") else float(x)
    mu    = s(returns.mean())
    sigma = s(returns.std())
    last  = s(closes.iloc[-1])
    info  = yf.Ticker(ticker).info
    name  = info.get("longName") or info.get("shortName") or ticker
    hist  = [round(float(v), 2) for v in closes.iloc[-252:].tolist()]
    return {
        "ticker": ticker, "name": name,
        "last_price": round(last, 2), "mu": mu, "sigma": sigma,
        "history": hist,
    }, None


def run_sim(last, mu, sigma, years, n, vol):
    steps = int(years * 252)
    sa    = sigma * vol
    drift = mu - 0.5 * sa ** 2
    rng   = np.random.default_rng()
    Z     = rng.standard_normal((n, steps))
    paths = last * np.exp(np.cumsum(drift + sa * Z, axis=1))
    paths = np.hstack([np.full((n, 1), last), paths])
    idx   = np.linspace(0, steps, min(90, steps + 1), dtype=int)
    samp  = paths[:, idx]
    pcts  = {p: np.percentile(samp, p, axis=0).tolist()
             for p in [5, 25, 50, 75, 95]}
    final = paths[:, -1]
    labels = []
    for i in idx:
        mo = round(i / 252 * 12)
        if mo == 0:   labels.append("Now")
        elif mo < 12: labels.append(f"{mo}m")
        else:
            yr = mo // 12; rm = mo % 12
            labels.append(f"{yr}y" + (f"{rm}m" if rm else ""))
    return {
        "labels": labels,
        "pcts":   pcts,
        "paths":  samp[:100].tolist(),
        "p5":  round(float(np.percentile(final, 5)),  2),
        "p25": round(float(np.percentile(final, 25)), 2),
        "p50": round(float(np.percentile(final, 50)), 2),
        "p75": round(float(np.percentile(final, 75)), 2),
        "p95": round(float(np.percentile(final, 95)), 2),
        "prob_gain":   round(float(np.mean(final > last) * 100),       1),
        "prob_20up":   round(float(np.mean(final > last * 1.2) * 100), 1),
        "prob_loss20": round(float(np.mean(final < last * 0.8) * 100), 1),
    }


# ── State ───────────────────────────────────────────────────────────────────
for k, v in [("asset", None), ("results", None), ("loaded", ""),
             ("years", 1), ("nsims", 600), ("vol", 1.0)]:
    if k not in st.session_state: st.session_state[k] = v

# Handle query params from the UI iframe
params = st.query_params
action = params.get("action", "")

if action == "fetch":
    t = params.get("ticker", "SPY").upper().strip()
    if t != st.session_state.loaded:
        data, err = fetch_data(t)
        if not err:
            st.session_state.asset   = data
            st.session_state.loaded  = t
            st.session_state.results = None

if action == "simulate" or (st.session_state.asset and st.session_state.results is None):
    try:
        st.session_state.years = int(params.get("years",  st.session_state.years))
        st.session_state.nsims = int(params.get("nsims",  st.session_state.nsims))
        st.session_state.vol   = float(params.get("vol",  st.session_state.vol))
    except: pass
    if st.session_state.asset:
        a = st.session_state.asset
        st.session_state.results = run_sim(
            a["last_price"], a["mu"], a["sigma"],
            st.session_state.years, st.session_state.nsims, st.session_state.vol)

# Auto-load SPY on first visit
if not st.session_state.loaded:
    data, _ = fetch_data("SPY")
    if data:
        st.session_state.asset   = data
        st.session_state.loaded  = "SPY"
        st.session_state.results = run_sim(
            data["last_price"], data["mu"], data["sigma"], 1, 600, 1.0)

asset   = st.session_state.asset
results = st.session_state.results

# Prepare JSON for the frontend
asset_json   = json.dumps(asset   or {})
results_json = json.dumps(results or {})
years_val    = st.session_state.years
nsims_val    = st.session_state.nsims
vol_val      = st.session_state.vol

# ── Full HTML UI ─────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,200;0,9..144,300;1,9..144,200;1,9..144,300&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#07080b;
  --surface:#0d0f14;
  --surface2:#13161d;
  --line:rgba(255,255,255,0.06);
  --line2:rgba(255,255,255,0.11);
  --text:#dde1ec;
  --muted:#3d4258;
  --muted2:#5a6070;
  --accent:#e8eaf0;
  --green:#4caf7d;
  --red:#c0392b;
  --yellow:#d4a843;
}}
html,body{{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;
  font-size:14px;min-height:100vh;overflow-x:hidden;}}

.app{{max-width:1200px;margin:0 auto;padding:3.5rem 2.5rem 6rem;}}

/* Header */
.eyebrow{{font-size:10px;letter-spacing:.2em;color:var(--muted);text-transform:uppercase;
  margin-bottom:1.2rem;}}
.headline{{font-family:'Fraunces',Georgia,serif;font-size:clamp(2.2rem,4vw,3.4rem);
  font-weight:200;font-style:italic;color:var(--accent);line-height:1.05;
  margin-bottom:1rem;}}
.subhead{{font-size:12px;color:var(--muted2);line-height:1.9;max-width:500px;
  margin-bottom:2.8rem;}}

/* Rule */
.rule{{height:1px;background:var(--line);margin:2rem 0;}}

/* Search */
.search-row{{display:flex;gap:12px;align-items:center;margin-bottom:.8rem;}}
.ticker-wrap{{position:relative;}}
.ticker-input{{
  background:var(--surface);
  border:1px solid var(--line2);
  border-radius:6px;
  color:var(--accent);
  font-family:'DM Mono',monospace;
  font-size:16px;
  font-weight:500;
  letter-spacing:.1em;
  height:50px;
  width:200px;
  padding:0 1rem;
  outline:none;
  text-transform:uppercase;
  transition:border-color .2s;
}}
.ticker-input::placeholder{{color:var(--muted);text-transform:none;font-weight:300;
  letter-spacing:.02em;font-size:13px;}}
.ticker-input:focus{{border-color:rgba(255,255,255,.3);}}
.load-btn{{
  background:var(--accent);color:var(--bg);
  border:none;border-radius:6px;
  font-family:'DM Mono',monospace;font-size:12px;font-weight:500;
  letter-spacing:.08em;height:50px;padding:0 1.5rem;
  cursor:pointer;white-space:nowrap;
  transition:opacity .15s,transform .1s;
}}
.load-btn:hover{{opacity:.85;}}
.load-btn:active{{transform:scale(.97);}}
.quick{{font-size:10px;color:var(--muted);letter-spacing:.06em;}}
.quick span{{cursor:pointer;transition:color .15s;}}
.quick span:hover{{color:var(--text);}}

/* Error */
.error-msg{{font-size:12px;color:var(--red);padding:.4rem 0 .8rem;display:none;}}

/* Asset strip */
.asset-strip{{display:none;margin:2rem 0;}}
.asset-strip.show{{display:grid;grid-template-columns:auto 1fr repeat(4,auto);
  gap:0;align-items:center;}}
.as-ticker{{
  font-family:'Fraunces',serif;font-size:2rem;font-weight:200;
  color:var(--accent);padding-right:2rem;margin-right:2rem;
  border-right:1px solid var(--line);
}}
.as-name{{font-size:12px;color:var(--muted2);line-height:1.5;padding-right:2rem;}}
.as-stat{{
  border-left:1px solid var(--line);
  padding:0 2rem;text-align:right;
}}
.as-stat:first-of-type{{border-left:none;}}
.as-label{{font-size:9px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);margin-bottom:.3rem;}}
.as-val{{font-family:'Fraunces',serif;font-size:1.4rem;font-weight:200;color:var(--accent);}}

/* Controls */
.controls{{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:2rem;
  align-items:end;margin:2rem 0;}}
.ctrl-label{{font-size:9px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--muted);margin-bottom:.6rem;display:flex;justify-content:space-between;}}
.ctrl-val{{color:var(--muted2);font-size:11px;letter-spacing:.02em;}}
input[type=range]{{
  -webkit-appearance:none;width:100%;height:2px;
  background:var(--line2);border-radius:1px;outline:none;cursor:pointer;
}}
input[type=range]::-webkit-slider-thumb{{
  -webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--accent);cursor:pointer;
  transition:transform .1s;
}}
input[type=range]::-webkit-slider-thumb:hover{{transform:scale(1.25);}}
select{{
  background:var(--surface);border:1px solid var(--line2);border-radius:6px;
  color:var(--text);font-family:'DM Mono',monospace;font-size:12px;
  height:40px;padding:0 .75rem;outline:none;cursor:pointer;width:100%;
  transition:border-color .2s;
}}
select:focus{{border-color:rgba(255,255,255,.3);}}
.run-btn{{
  background:transparent;border:1px solid var(--line2);border-radius:6px;
  color:var(--text);font-family:'DM Mono',monospace;font-size:12px;
  letter-spacing:.06em;height:40px;padding:0 1.5rem;
  cursor:pointer;white-space:nowrap;
  transition:border-color .2s,color .2s;
}}
.run-btn:hover{{border-color:rgba(255,255,255,.3);color:var(--accent);}}
.run-btn:active{{opacity:.7;}}

/* Metrics section */
.section-label{{font-size:9px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--muted);margin-bottom:1.2rem;}}
.metrics-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:8px;
  overflow:hidden;margin-bottom:3rem;}}
.metric{{background:var(--surface);padding:1.4rem 1.25rem 1.2rem;}}
.metric:first-child{{border-radius:8px 0 0 8px;}}
.metric:last-child{{border-radius:0 8px 8px 0;}}
.m-label{{font-size:9px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);margin-bottom:.5rem;}}
.m-val{{font-family:'Fraunces',serif;font-size:1.5rem;font-weight:200;
  color:var(--accent);line-height:1;margin-bottom:.3rem;}}
.m-delta{{font-size:10px;color:var(--muted2);}}
.m-delta.pos{{color:var(--green);}}
.m-delta.neg{{color:var(--red);}}

/* Chart */
.chart-wrap{{position:relative;height:380px;margin-bottom:.75rem;}}
.chart-caption{{font-size:10px;color:var(--muted);letter-spacing:.06em;
  margin-bottom:3rem;}}
.chart-legend{{display:flex;gap:1.5rem;margin-bottom:1rem;}}
.leg{{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted2);}}
.leg-line{{width:18px;height:2px;border-radius:1px;}}

/* Prob row */
.prob-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:8px;
  overflow:hidden;margin-bottom:3rem;}}
.prob-card{{background:var(--surface);padding:1.6rem 1.5rem;}}
.prob-val{{font-family:'Fraunces',serif;font-size:2.2rem;font-weight:200;
  color:var(--accent);line-height:1;margin-bottom:.5rem;}}
.prob-val.green{{color:var(--green);}}
.prob-val.red{{color:var(--red);}}
.prob-label{{font-size:10px;color:var(--muted2);line-height:1.7;}}

/* Methodology */
.meth{{border-top:1px solid var(--line);padding-top:2rem;}}
.meth-toggle{{font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);cursor:pointer;background:none;border:none;
  font-family:'DM Mono',monospace;transition:color .15s;padding:0;}}
.meth-toggle:hover{{color:var(--text);}}
.meth-body{{display:none;margin-top:1.5rem;}}
.meth-body.open{{display:grid;grid-template-columns:1fr 1fr;gap:2rem;}}
.meth-block .meth-title{{font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted2);margin-bottom:.6rem;}}
.meth-block p{{font-size:11px;color:var(--muted2);line-height:1.8;}}
.meth-table{{width:100%;border-collapse:collapse;margin-top:.5rem;}}
.meth-table td{{font-size:11px;color:var(--muted2);padding:.35rem 0;
  border-bottom:1px solid var(--line);line-height:1.6;}}
.meth-table td:first-child{{color:var(--muted);padding-right:1.5rem;white-space:nowrap;}}
.disclaimer{{font-size:10px;color:var(--muted);line-height:1.8;margin-top:1.5rem;
  padding-top:1.5rem;border-top:1px solid var(--line);}}

/* Loading overlay */
.loading{{display:none;position:fixed;inset:0;background:rgba(7,8,11,.85);
  z-index:999;align-items:center;justify-content:center;}}
.loading.show{{display:flex;}}
.loading-text{{font-size:11px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--muted2);animation:pulse 1.4s ease-in-out infinite;}}
@keyframes pulse{{0%,100%{{opacity:.3}}50%{{opacity:1}}}}

@media(max-width:900px){{
  .controls{{grid-template-columns:1fr 1fr;}}
  .asset-strip.show{{grid-template-columns:auto 1fr;row-gap:1rem;}}
  .as-stat{{border-left:none;padding:0;text-align:left;}}
  .metrics-row{{grid-template-columns:repeat(3,1fr);}}
  .prob-row{{grid-template-columns:1fr 1fr;}}
  .meth-body.open{{grid-template-columns:1fr;}}
}}
@media(max-width:600px){{
  .app{{padding:2rem 1.25rem 4rem;}}
  .controls{{grid-template-columns:1fr;}}
  .metrics-row{{grid-template-columns:1fr 1fr;}}
}}
</style>
</head>
<body>
<div class="loading" id="loading"><div class="loading-text">Simulating</div></div>
<div class="app">

  <div class="eyebrow">Monte Carlo Price Simulator</div>
  <div class="headline">Where could this<br>stock end up?</div>
  <div class="subhead">Enter any ticker. Five years of price history, thousands of simulated
  futures, one clear picture of the range of outcomes.</div>

  <div class="rule"></div>

  <div class="search-row">
    <div class="ticker-wrap">
      <input class="ticker-input" id="ticker" type="text"
             placeholder="AAPL, TSLA, BTC-USD…"
             value="{st.session_state.loaded or 'SPY'}" maxlength="12" spellcheck="false">
    </div>
    <button class="load-btn" onclick="loadTicker()">Load data</button>
    <div class="quick">
      Quick:&nbsp;
      <span onclick="setTicker('SPY')">SPY</span> &middot;
      <span onclick="setTicker('AAPL')">AAPL</span> &middot;
      <span onclick="setTicker('TSLA')">TSLA</span> &middot;
      <span onclick="setTicker('MSFT')">MSFT</span> &middot;
      <span onclick="setTicker('NVDA')">NVDA</span> &middot;
      <span onclick="setTicker('BTC-USD')">BTC-USD</span>
    </div>
  </div>
  <div class="error-msg" id="err"></div>

  <!-- Asset strip -->
  <div class="asset-strip" id="asset-strip"></div>
  <div class="rule" id="rule1" style="display:none;"></div>

  <!-- Controls -->
  <div class="controls" id="controls" style="display:none;">
    <div>
      <div class="ctrl-label">Time horizon <span class="ctrl-val" id="lbl-years">1 year</span></div>
      <input type="range" id="sl-years" min="1" max="10" step="1" value="{years_val}"
             oninput="updLabel()">
    </div>
    <div>
      <div class="ctrl-label">Simulations <span class="ctrl-val" id="lbl-sims">{nsims_val:,}</span></div>
      <input type="range" id="sl-sims" min="100" max="3000" step="100" value="{nsims_val}"
             oninput="updLabel()">
    </div>
    <div>
      <div class="ctrl-label">Volatility scenario</div>
      <select id="sel-vol">
        <option value="1.0"  {'selected' if vol_val==1.0  else ''}>Normal (1&times;)</option>
        <option value="1.5"  {'selected' if vol_val==1.5  else ''}>Elevated (1.5&times;)</option>
        <option value="2.0"  {'selected' if vol_val==2.0  else ''}>High (2&times;)</option>
        <option value="2.5"  {'selected' if vol_val==2.5  else ''}>Crisis (2.5&times;)</option>
      </select>
    </div>
    <div>
      <button class="run-btn" onclick="simulate()">Run simulation</button>
    </div>
  </div>

  <!-- Results -->
  <div id="results" style="display:none;">
    <div class="section-label" id="horizon-label">Projected price — 1 year horizon</div>
    <div class="metrics-row" id="metrics-row"></div>

    <div class="chart-legend">
      <div class="leg"><div class="leg-line" style="background:var(--red);border-top:2px dashed var(--red);height:0;"></div>Bear &mdash; 5th pct</div>
      <div class="leg"><div class="leg-line" style="background:var(--text);"></div>Base &mdash; median</div>
      <div class="leg"><div class="leg-line" style="background:var(--green);border-top:2px dashed var(--green);height:0;"></div>Bull &mdash; 95th pct</div>
      <div class="leg"><div class="leg-line" style="background:rgba(255,255,255,0.06);width:18px;height:10px;border-radius:2px;"></div>Simulated paths</div>
    </div>
    <div class="chart-wrap"><canvas id="chart" role="img" aria-label="Monte Carlo price simulation chart"></canvas></div>
    <div class="chart-caption" id="chart-caption"></div>

    <div class="section-label">Probability at end of horizon</div>
    <div class="prob-row" id="prob-row"></div>

    <div class="meth">
      <button class="meth-toggle" onclick="toggleMeth()">+ Methodology &amp; parameters</button>
      <div class="meth-body" id="meth-body">
        <div class="meth-block">
          <div class="meth-title">How it works</div>
          <p>We model daily price returns using Geometric Brownian Motion (GBM), the standard
          framework for equity price simulation. Each simulation draws random daily shocks from a
          normal distribution calibrated to the asset's historical drift and volatility.
          Running hundreds of paths reveals the probability distribution of future prices.</p>
          <p style="margin-top:.8rem;">GBM assumes log-normal returns and constant volatility.
          It does not capture fat tails, mean reversion, or structural regime changes.
          These results are for educational and illustrative purposes only.</p>
        </div>
        <div class="meth-block">
          <div class="meth-title">Parameters</div>
          <table class="meth-table" id="meth-table"></table>
          <div class="disclaimer">
            Data sourced from Yahoo Finance via yfinance. Past performance is not indicative
            of future results. This tool does not constitute financial advice.
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /app -->

<script>
const ASSET   = {asset_json};
const RESULTS = {results_json};
let chart     = null;

// ── Init ─────────────────────────────────────────────────────────────────────
function init() {{
  updLabel();
  if (ASSET && ASSET.ticker) {{
    renderAsset(ASSET);
    if (RESULTS && RESULTS.labels) renderResults(RESULTS, ASSET);
  }}
}}

// ── Ticker utils ─────────────────────────────────────────────────────────────
function setTicker(t) {{
  document.getElementById('ticker').value = t;
  loadTicker();
}}

function loadTicker() {{
  const t = document.getElementById('ticker').value.trim().toUpperCase();
  if (!t) return;
  showErr('');
  setLoading(true);
  const url = new URL(window.location.href);
  url.searchParams.set('action', 'fetch');
  url.searchParams.set('ticker', t);
  window.location.href = url.toString();
}}

function simulate() {{
  const years = document.getElementById('sl-years').value;
  const nsims = document.getElementById('sl-sims').value;
  const vol   = document.getElementById('sel-vol').value;
  setLoading(true);
  const url = new URL(window.location.href);
  url.searchParams.set('action', 'simulate');
  url.searchParams.set('years', years);
  url.searchParams.set('nsims', nsims);
  url.searchParams.set('vol', vol);
  window.location.href = url.toString();
}}

// ── Labels ───────────────────────────────────────────────────────────────────
function updLabel() {{
  const y = +document.getElementById('sl-years').value;
  const s = +document.getElementById('sl-sims').value;
  document.getElementById('lbl-years').textContent = y === 1 ? '1 year' : y + ' years';
  document.getElementById('lbl-sims').textContent  = s.toLocaleString();
}}

// ── Asset strip ──────────────────────────────────────────────────────────────
function renderAsset(a) {{
  const annRet = ((1 + a.mu) ** 252 - 1) * 100;
  const annVol = a.sigma * Math.sqrt(252) * 100;
  const strip  = document.getElementById('asset-strip');
  strip.innerHTML = `
    <div class="as-ticker">${{a.ticker}}</div>
    <div class="as-name">${{a.name}}</div>
    <div class="as-stat">
      <div class="as-label">Price</div>
      <div class="as-val">$${{a.last_price.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}})}}</div>
    </div>
    <div class="as-stat">
      <div class="as-label">Ann. return</div>
      <div class="as-val">${{annRet >= 0 ? '+' : ''}}${{annRet.toFixed(1)}}%</div>
    </div>
    <div class="as-stat">
      <div class="as-label">Ann. volatility</div>
      <div class="as-val">${{annVol.toFixed(1)}}%</div>
    </div>
    <div class="as-stat">
      <div class="as-label">Daily drift</div>
      <div class="as-val">${{(a.mu*100).toFixed(3)}}%</div>
    </div>`;
  strip.classList.add('show');
  document.getElementById('rule1').style.display = 'block';
  document.getElementById('controls').style.display = 'grid';
}}

// ── Results ──────────────────────────────────────────────────────────────────
function renderResults(r, a) {{
  document.getElementById('results').style.display = 'block';
  const last  = a.last_price;
  const years = +document.getElementById('sl-years').value;
  document.getElementById('horizon-label').textContent =
    'Projected price — ' + (years === 1 ? '1 year' : years + ' years') + ' horizon';

  // Metrics row
  const pts = [
    ['5th percentile',  r.p5,  'Bear case'],
    ['25th percentile', r.p25, 'Low case'],
    ['Median',          r.p50, 'Base case'],
    ['75th percentile', r.p75, 'High case'],
    ['95th percentile', r.p95, 'Bull case'],
  ];
  document.getElementById('metrics-row').innerHTML = pts.map(([lbl, v, sub]) => {{
    const pct = ((v - last) / last * 100);
    const cls = pct >= 0 ? 'pos' : 'neg';
    const sgn = pct >= 0 ? '+' : '';
    return `<div class="metric">
      <div class="m-label">${{lbl}}</div>
      <div class="m-val">$${{Math.round(v).toLocaleString()}}</div>
      <div class="m-delta ${{cls}}">${{sgn}}${{pct.toFixed(1)}}% &mdash; ${{sub}}</div>
    </div>`;
  }}).join('');

  // Chart
  drawChart(r, a);

  // Caption
  const volSel = document.getElementById('sel-vol');
  const volLbl = volSel ? volSel.options[volSel.selectedIndex].text : '';
  const nsims  = document.getElementById('sl-sims').value;
  document.getElementById('chart-caption').textContent =
    (+nsims).toLocaleString() + ' simulations · ' + volLbl + ' · Geometric Brownian Motion';

  // Prob row
  const probs = [
    ['Probability of any gain',     r.prob_gain,   false],
    ['Probability of +20% or more', r.prob_20up,   false],
    ['Probability of losing 20%+',  r.prob_loss20, true],
  ];
  document.getElementById('prob-row').innerHTML = probs.map(([lbl, v, isRed]) => `
    <div class="prob-card">
      <div class="prob-val ${{isRed ? 'red' : 'green'}}">${{v.toFixed(1)}}%</div>
      <div class="prob-label">${{lbl}}</div>
    </div>`).join('');

  // Methodology table
  const a2 = ASSET;
  const annRet = ((1 + a2.mu) ** 252 - 1) * 100;
  const annVol = a2.sigma * Math.sqrt(252) * 100;
  document.getElementById('meth-table').innerHTML = [
    ['Ticker', a2.ticker],
    ['Daily drift (μ)', (a2.mu * 100).toFixed(4) + '%'],
    ['Daily volatility (σ)', (a2.sigma * 100).toFixed(4) + '%'],
    ['Annualised return', annRet.toFixed(2) + '%'],
    ['Annualised volatility', annVol.toFixed(2) + '%'],
    ['Volatility multiplier', document.getElementById('sel-vol')?.value + '×'],
    ['Simulations', (+document.getElementById('sl-sims').value).toLocaleString()],
    ['Trading days modelled', (years * 252).toString()],
  ].map(([k,v]) => `<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
}}

// ── Chart ─────────────────────────────────────────────────────────────────────
function drawChart(r, a) {{
  if (chart) {{ chart.destroy(); chart = null; }}
  const labels   = r.labels;
  const pcts     = r.pcts;
  const paths    = r.paths;
  const datasets = [];

  // IQR fill
  datasets.push({{
    data: [...pcts['75'], ...[...pcts['25']].reverse()],
    labels: [...labels, ...[...labels].reverse()],
    fill: true,
    backgroundColor: 'rgba(255,255,255,0.025)',
    borderColor: 'transparent',
    pointRadius: 0,
    showlegend: false,
  }});

  // Paths
  for (const p of paths) {{
    datasets.push({{
      data: p,
      borderColor: 'rgba(255,255,255,0.04)',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    }});
  }}

  // Percentile lines
  const lines = [
    {{ pct:'5',  color:'#c0392b', dash:[5,4] }},
    {{ pct:'50', color:'#dde1ec', dash:[]    }},
    {{ pct:'95', color:'#4caf7d', dash:[5,4] }},
  ];
  for (const l of lines) {{
    datasets.push({{
      label: l.pct === '50' ? 'Median' : l.pct + 'th pct',
      data: pcts[l.pct],
      borderColor: l.color,
      borderWidth: l.pct === '50' ? 2.5 : 1.8,
      borderDash: l.dash,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    }});
  }}

  chart = new Chart(document.getElementById('chart'), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: {{ duration: 500 }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          mode: 'index', intersect: false,
          filter: i => i.datasetIndex >= paths.length + 1,
          backgroundColor: '#0d0f14',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#3d4258',
          bodyColor: '#dde1ec',
          titleFont: {{ family: 'DM Mono', size: 10 }},
          bodyFont:  {{ family: 'DM Mono', size: 12 }},
          padding: 12,
          callbacks: {{
            label: ctx => ' ' + ctx.dataset.label + ':  $' + Math.round(ctx.raw).toLocaleString(),
          }},
        }},
      }},
      scales: {{
        x: {{
          grid: {{ color: 'rgba(255,255,255,0.04)' }},
          ticks: {{ maxTicksLimit: 10, color: '#3d4258', font: {{ family:'DM Mono', size:10 }} }},
        }},
        y: {{
          grid: {{ color: 'rgba(255,255,255,0.04)' }},
          ticks: {{
            color: '#3d4258',
            font: {{ family:'DM Mono', size:10 }},
            callback: v => '$' + Math.round(v).toLocaleString(),
          }},
        }},
      }},
    }},
  }});
}}

function toggleMeth() {{
  const b = document.getElementById('meth-body');
  const btn = document.querySelector('.meth-toggle');
  b.classList.toggle('open');
  btn.textContent = b.classList.contains('open')
    ? '- Methodology & parameters'
    : '+ Methodology & parameters';
}}

function showErr(msg) {{
  const el = document.getElementById('err');
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}}

function setLoading(v) {{
  document.getElementById('loading').classList.toggle('show', v);
}}

document.getElementById('ticker').addEventListener('keydown', e => {{
  if (e.key === 'Enter') loadTicker();
}});
document.getElementById('ticker').addEventListener('input', e => {{
  const p = e.target.selectionStart;
  e.target.value = e.target.value.toUpperCase();
  e.target.setSelectionRange(p, p);
}});

init();
</script>
</body>
</html>"""

st.components.v1.html(HTML, height=2400, scrolling=False)
