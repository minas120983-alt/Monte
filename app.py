"""
Monte Carlo Stock Price Simulator — Streamlit App
"""

import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,200;0,9..144,400;1,9..144,200;1,9..144,400&display=swap');

html, body, [class*="css"], .stApp { background-color: #08090c; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

* { font-family: 'DM Mono', monospace !important; }

/* ── Layout wrapper ── */
.block-container {
    max-width: 1160px !important;
    padding: 3rem 2rem 4rem !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 2rem 0 !important;
}

/* ── Text ── */
p, li, label, div { color: #c9cdd8; }
h1, h2, h3, h4 { color: #e8eaf0 !important; font-weight: 400 !important; }

/* ── Inputs ── */
input[type="text"],
div[data-baseweb="input"] input {
    background: #0f1117 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-size: 15px !important;
    letter-spacing: 0.08em !important;
    font-weight: 500 !important;
    height: 48px !important;
    padding: 0 1rem !important;
    transition: border-color 0.2s !important;
}
input[type="text"]:focus,
div[data-baseweb="input"]:focus-within input {
    border-color: rgba(255,255,255,0.35) !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-baseweb="input"] {
    background: transparent !important;
    border: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #e8eaf0 !important;
    color: #08090c !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    height: 48px !important;
    padding: 0 1.5rem !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.82 !important; }
.stButton > button:active { opacity: 0.65 !important; }

/* ── Sliders ── */
[data-testid="stSlider"] {
    padding: 0 !important;
}
[data-testid="stSlider"] > label {
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5066 !important;
    margin-bottom: 6px !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }
div[data-testid="stSlider"] > div > div > div[role="slider"] {
    background: #e8eaf0 !important;
    border: none !important;
    box-shadow: none !important;
    width: 14px !important;
    height: 14px !important;
}
div[data-testid="stSlider"] > div > div > div:first-child {
    background: rgba(255,255,255,0.08) !important;
    height: 2px !important;
}
div[data-testid="stSlider"] > div > div > div:nth-child(2) {
    background: #e8eaf0 !important;
    height: 2px !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background: #0f1117 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    min-height: 48px !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(255,255,255,0.35) !important;
    box-shadow: none !important;
}
div[data-baseweb="popover"] {
    background: #0f1117 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
li[role="option"] { color: #c9cdd8 !important; }
li[role="option"]:hover { background: rgba(255,255,255,0.05) !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0f1117;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1.1rem 1.25rem 1rem;
}
[data-testid="stMetricLabel"] > div {
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5066 !important;
    font-weight: 400 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.6rem !important;
    font-weight: 200 !important;
    color: #e8eaf0 !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    color: #4a5066 !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── Expander ── */
details {
    background: #0f1117 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 0.25rem 0 !important;
}
summary {
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5066 !important;
    padding: 0.85rem 1.25rem !important;
    cursor: pointer !important;
}
summary:hover { color: #c9cdd8 !important; }

/* ── Label text ── */
.stTextInput label, .stSelectbox label, .stSlider label {
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5066 !important;
    margin-bottom: 6px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #4a5066 !important; }

/* ── Columns gap ── */
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str):
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        return None, f"No data found for '{ticker}'."
    closes = data["Close"].dropna().squeeze()
    returns = closes.pct_change().dropna()
    def scalar(x):
        return float(x.iloc[0]) if hasattr(x, "iloc") else float(x)
    mu    = scalar(returns.mean())
    sigma = scalar(returns.std())
    last  = scalar(closes.iloc[-1])
    info  = yf.Ticker(ticker).info
    name  = info.get("longName") or info.get("shortName") or ticker
    currency = info.get("currency", "USD")
    return {"ticker": ticker, "name": name, "currency": currency,
            "last_price": round(last, 2), "mu": mu, "sigma": sigma}, None


def run_simulation(last_price, mu, sigma, years, n_sims, vol_mult):
    steps     = int(years * 252)
    sigma_adj = sigma * vol_mult
    drift     = mu - 0.5 * sigma_adj ** 2
    rng       = np.random.default_rng()
    Z         = rng.standard_normal((n_sims, steps))
    paths     = last_price * np.exp(np.cumsum(drift + sigma_adj * Z, axis=1))
    paths     = np.hstack([np.full((n_sims, 1), last_price), paths])
    idx       = np.linspace(0, steps, min(80, steps + 1), dtype=int)
    sampled   = paths[:, idx]
    pcts      = {p: np.percentile(sampled, p, axis=0) for p in [5, 25, 50, 75, 95]}
    labels    = []
    for i in idx:
        mo = round(i / 252 * 12)
        if mo == 0:    labels.append("Now")
        elif mo < 12:  labels.append(f"{mo}m")
        else:
            yr = mo // 12; rem = mo % 12
            labels.append(f"{yr}y" + (f" {rem}m" if rem else ""))
    return {"labels": labels, "pcts": pcts, "paths": sampled[:120], "final": paths[:, -1]}


def fmt_price(v): return f"${v:,.2f}"
def fmt_k(v):     return f"${v:,.0f}"
def pct_chg(new, old):
    c = (new - old) / old * 100
    return f"+{c:.1f}%" if c >= 0 else f"{c:.1f}%"


# ── Session state ───────────────────────────────────────────────────────────────
for k, v in [("asset", None), ("results", None), ("ticker_loaded", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom: 2.5rem;">
  <div style="font-size: 10px; letter-spacing: 0.18em; color: #4a5066;
              text-transform: uppercase; margin-bottom: 1rem;">
    Monte Carlo Price Simulator
  </div>
  <div style="font-family: 'Fraunces', Georgia, serif; font-size: clamp(2rem, 4vw, 3rem);
              font-weight: 200; color: #e8eaf0; line-height: 1.1; font-style: italic;">
    Where could this stock end up?
  </div>
  <div style="margin-top: 0.75rem; font-size: 12px; color: #4a5066;
              max-width: 520px; line-height: 1.8;">
    Enter any ticker. We pull 5 years of price history, model thousands of possible futures
    using Geometric Brownian Motion, and show you the realistic range of outcomes.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin-bottom:2rem;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TICKER INPUT ROW
# ══════════════════════════════════════════════════════════════════════════════
c_input, c_btn, c_quick = st.columns([2, 1, 4])

with c_input:
    ticker_raw = st.text_input("Ticker symbol", value="SPY", label_visibility="collapsed",
                                placeholder="Ticker — e.g. AAPL, TSLA, BTC-USD")

with c_btn:
    fetch_clicked = st.button("Load data")

with c_quick:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; height:48px; flex-wrap:wrap;">
      <span style="font-size:10px;letter-spacing:.1em;color:#4a5066;text-transform:uppercase;">Quick load</span>
      <span style="font-size:11px;color:#4a5066;">SPY · AAPL · TSLA · MSFT · NVDA · BTC-USD · GLD</span>
    </div>
    """, unsafe_allow_html=True)

ticker = ticker_raw.upper().strip()

# Auto-load on first visit
if not st.session_state.ticker_loaded:
    fetch_clicked = True

if fetch_clicked and ticker:
    if ticker != st.session_state.ticker_loaded:
        with st.spinner(f"Fetching {ticker}…"):
            data, err = fetch_data(ticker)
        if err:
            st.markdown(f'<div style="font-size:12px;color:#c0392b;padding:.5rem 0;">{err}</div>',
                        unsafe_allow_html=True)
        else:
            st.session_state.asset         = data
            st.session_state.ticker_loaded = ticker
            st.session_state.results       = None


# ══════════════════════════════════════════════════════════════════════════════
# ASSET STRIP
# ══════════════════════════════════════════════════════════════════════════════
asset = st.session_state.asset

if asset:
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:1.5rem 0;"></div>',
                unsafe_allow_html=True)

    ann_ret = ((1 + asset["mu"]) ** 252 - 1) * 100
    ann_vol = asset["sigma"] * (252 ** 0.5) * 100

    a1, a2, a3, a4, a5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2])
    with a1:
        st.markdown(f"""
        <div style="padding: 1.1rem 0 1rem;">
          <div style="font-family:'Fraunces',serif; font-size:1.5rem; font-weight:200;
                      color:#e8eaf0; letter-spacing:.03em;">{asset['ticker']}</div>
          <div style="font-size:11px; color:#4a5066; margin-top:4px;
                      white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                      max-width:260px;">{asset['name']}</div>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.metric("Price", fmt_price(asset["last_price"]))
    with a3:
        st.metric("Ann. return", f"{ann_ret:+.1f}%")
    with a4:
        st.metric("Ann. volatility", f"{ann_vol:.1f}%")
    with a5:
        st.metric("Daily drift (μ)", f"{asset['mu']*100:.3f}%")

    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:1.5rem 0 2rem;"></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONTROLS ROW
# ══════════════════════════════════════════════════════════════════════════════
if asset:
    s1, s2, s3, s4 = st.columns([1.5, 1.5, 1.5, 1])

    with s1:
        years = st.slider("Time horizon (years)", 1, 10, 1, 1)

    with s2:
        n_sims = st.slider("Simulations", 100, 3000, 600, 100)

    with s3:
        vol_label = st.selectbox("Volatility scenario",
            ["Normal (1×)", "Elevated (1.5×)", "High (2×)", "Crisis (2.5×)"])
        vol_map  = {"Normal (1×)": 1.0, "Elevated (1.5×)": 1.5,
                    "High (2×)": 2.0, "Crisis (2.5×)": 2.5}
        vol_mult = vol_map[vol_label]

    with s4:
        st.markdown('<div style="height:1.45rem;"></div>', unsafe_allow_html=True)
        run_clicked = st.button("Run simulation")

    if run_clicked or st.session_state.results is None:
        with st.spinner("Simulating…"):
            st.session_state.results = run_simulation(
                asset["last_price"], asset["mu"], asset["sigma"],
                years, n_sims, vol_mult)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
results = st.session_state.results

if asset and results:
    final = results["final"]
    last  = asset["last_price"]

    p5  = float(np.percentile(final, 5))
    p25 = float(np.percentile(final, 25))
    p50 = float(np.percentile(final, 50))
    p75 = float(np.percentile(final, 75))
    p95 = float(np.percentile(final, 95))

    prob_gain   = float(np.mean(final > last)       * 100)
    prob_20up   = float(np.mean(final > last * 1.2) * 100)
    prob_loss20 = float(np.mean(final < last * 0.8) * 100)

    # ── Outcome metrics ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                color:#4a5066;margin-bottom:1rem;">Projected price at end of horizon</div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("5th percentile",  fmt_k(p5),  pct_chg(p5,  last))
    m2.metric("25th percentile", fmt_k(p25), pct_chg(p25, last))
    m3.metric("Median",          fmt_k(p50), pct_chg(p50, last))
    m4.metric("75th percentile", fmt_k(p75), pct_chg(p75, last))
    m5.metric("95th percentile", fmt_k(p95), pct_chg(p95, last))

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)

    # ── Chart ────────────────────────────────────────────────────────────────
    labels = results["labels"]
    pcts   = results["pcts"]
    paths  = results["paths"]

    fig = go.Figure()

    # IQR fill
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=list(pcts[75]) + list(pcts[25])[::-1],
        fill="toself",
        fillcolor="rgba(255,255,255,0.03)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Individual paths
    for path in paths:
        fig.add_trace(go.Scatter(
            x=labels, y=path, mode="lines",
            line=dict(color="rgba(255,255,255,0.04)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile lines
    line_styles = [
        (5,  "#c0392b", "Bear  —  5th pct",  "dot"),
        (50, "#e8eaf0", "Base  —  median",   "solid"),
        (95, "#27ae60", "Bull  —  95th pct", "dot"),
    ]
    for p, color, name, dash in line_styles:
        fig.add_trace(go.Scatter(
            x=labels, y=pcts[p], mode="lines", name=name,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
        ))

    # Today reference
    fig.add_hline(y=last, line_dash="dot",
                  line_color="rgba(255,255,255,0.12)", line_width=1)

    fig.update_layout(
        paper_bgcolor="#08090c",
        plot_bgcolor="#08090c",
        height=400,
        margin=dict(l=0, r=0, t=12, b=0),
        legend=dict(
            orientation="h", x=0, y=1.06,
            font=dict(color="#4a5066", size=11, family="DM Mono"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="#4a5066", size=10, family="DM Mono"),
            showline=False, zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="#4a5066", size=10, family="DM Mono"),
            tickprefix="$", tickformat=",.0f",
            showline=False, zeroline=False,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0f1117",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#e8eaf0", size=11, family="DM Mono"),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Sub-label ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="font-size:10px;color:#4a5066;letter-spacing:.06em;margin-top:-1rem;margin-bottom:2rem;">
      {n_sims:,} simulations · {years} year{'s' if years > 1 else ''} ·
      {vol_label} · model: Geometric Brownian Motion
    </div>
    """, unsafe_allow_html=True)

    # ── Probability row ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                color:#4a5066;margin-bottom:1rem;">Probability breakdown</div>
    """, unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    p1.metric("Probability of any gain",     f"{prob_gain:.1f}%")
    p2.metric("Probability of +20% or more", f"{prob_20up:.1f}%")
    p3.metric("Probability of losing 20%+",  f"{prob_loss20:.1f}%")

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)

    # ── Methodology ─────────────────────────────────────────────────────────
    with st.expander("Methodology"):
        st.markdown(f"""
**Model:** Geometric Brownian Motion (GBM)  
**Data source:** Yahoo Finance via yfinance · 5-year daily closing prices  
**Parameters for {asset['ticker']}:**

| Parameter | Value |
|---|---|
| Daily drift (μ) | {asset['mu']*100:.4f}% |
| Daily volatility (σ) | {asset['sigma']*100:.4f}% |
| Annualised return | {((1+asset['mu'])**252-1)*100:.2f}% |
| Annualised volatility | {asset['sigma']*(252**0.5)*100:.2f}% |
| Simulations run | {n_sims:,} |
| Time horizon | {years} year{'s' if years > 1 else ''} ({years*252} trading days) |
| Volatility multiplier | {vol_mult}× |

**Important:** GBM assumes log-normal returns and constant volatility — it does not capture fat tails,
mean reversion, or regime changes. Results are for educational purposes only and do not constitute
financial advice. Past volatility is not indicative of future results.
        """)

elif not asset:
    st.markdown("""
    <div style="padding: 5rem 0; text-align: center;">
      <div style="font-family:'Fraunces',serif; font-size:1.2rem; font-weight:200;
                  font-style:italic; color:#4a5066;">
        Enter a ticker above to begin
      </div>
    </div>
    """, unsafe_allow_html=True)
