"""
Monte Carlo Stock Price Simulator — Streamlit App
Deploy on Streamlit Cloud: https://streamlit.io/cloud
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main background */
.stApp { background-color: #0b0e14; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #12151e !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

[data-testid="stSidebar"] * { color: #e8eaf0 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #12151e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.25rem;
}

[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #e8eaf0 !important; font-family: 'Fraunces', serif !important; }

/* Buttons */
.stButton > button {
    background: #7ee8a2 !important;
    color: #0b0e14 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    height: 46px !important;
    transition: opacity 0.15s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

/* Text input */
.stTextInput > div > div > input {
    background: #12151e !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    height: 52px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #7ee8a2 !important;
    box-shadow: 0 0 0 3px rgba(126,232,162,0.12) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #7ee8a2 !important;
}

/* Info boxes */
.stInfo { background: rgba(56,189,248,0.08) !important; border-color: rgba(56,189,248,0.2) !important; }
.stSuccess { background: rgba(126,232,162,0.08) !important; border-color: rgba(126,232,162,0.2) !important; }
.stWarning { background: rgba(251,191,36,0.08) !important; border-color: rgba(251,191,36,0.2) !important; }
.stError { background: rgba(248,113,113,0.08) !important; border-color: rgba(248,113,113,0.2) !important; }

/* Card-like containers */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: #12151e;
    border-radius: 14px;
}

/* Headings */
h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; color: #e8eaf0 !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #12151e !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}

p, label, .stMarkdown { color: #e8eaf0; }
.stSlider label { color: #6b7280 !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str):
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        return None, f"No data found for '{ticker}'. Check the ticker symbol."

    closes = data["Close"].dropna().squeeze()
    returns = closes.pct_change().dropna()

    def scalar(x):
        return float(x.iloc[0]) if hasattr(x, "iloc") else float(x)

    mu    = scalar(returns.mean())
    sigma = scalar(returns.std())
    last  = scalar(closes.iloc[-1])

    info = yf.Ticker(ticker).info
    name = info.get("longName") or info.get("shortName") or ticker
    currency = info.get("currency", "USD")

    hist = closes.iloc[-252:].tolist()

    return {
        "ticker": ticker,
        "name": name,
        "currency": currency,
        "last_price": round(last, 2),
        "mu": mu,
        "sigma": sigma,
        "history": [round(p, 2) for p in hist],
    }, None


# ── Simulation ──────────────────────────────────────────────────────────────────
def run_simulation(last_price, mu, sigma, years, n_sims, vol_mult):
    steps     = int(years * 252)
    sigma_adj = sigma * vol_mult
    drift     = mu - 0.5 * sigma_adj ** 2
    rng       = np.random.default_rng()
    Z         = rng.standard_normal((n_sims, steps))
    log_ret   = drift + sigma_adj * Z
    paths     = last_price * np.exp(np.cumsum(log_ret, axis=1))
    paths     = np.hstack([np.full((n_sims, 1), last_price), paths])

    idx       = np.linspace(0, steps, min(80, steps + 1), dtype=int)
    sampled   = paths[:, idx]

    pcts = {p: np.percentile(sampled, p, axis=0) for p in [5, 25, 50, 75, 95]}
    final = paths[:, -1]

    labels = []
    for i in idx:
        mo = round(i / 252 * 12)
        if mo == 0:       labels.append("Now")
        elif mo < 12:     labels.append(f"{mo}mo")
        else:
            yr  = mo // 12
            rem = mo % 12
            labels.append(f"{yr}yr" + (f" {rem}mo" if rem else ""))

    return {
        "labels":    labels,
        "pcts":      pcts,
        "paths":     sampled[:200],
        "final":     final,
        "idx":       idx,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────────
def fmt(v, currency="USD"):
    sym = "$" if currency == "USD" else ""
    return f"{sym}{v:,.0f}"

def pct_change(new, old):
    c = (new - old) / old * 100
    return f"+{c:.1f}%" if c >= 0 else f"{c:.1f}%"


# ── UI ──────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="margin-bottom:2rem;">
  <div style="font-size:11px;letter-spacing:.12em;color:#7ee8a2;text-transform:uppercase;margin-bottom:.4rem;">Monte Carlo Simulator</div>
  <div style="font-family:'Fraunces',serif;font-size:clamp(1.8rem,4vw,2.8rem);font-weight:300;color:#e8eaf0;line-height:1.15;">
    Where could your stock <em style="color:#7ee8a2;">actually</em> end up?
  </div>
  <div style="font-size:13px;color:#6b7280;margin-top:.6rem;max-width:560px;line-height:1.7;">
    Enter any stock ticker and we'll run thousands of simulated futures using real market data — 
    giving you a realistic range of outcomes, not just a single guess.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    ticker_input = st.text_input(
        "Stock Ticker",
        value="SPY",
        max_chars=12,
        help="Enter any valid Yahoo Finance ticker. Examples: AAPL, TSLA, MSFT, BTC-USD, GLD, ^GSPC",
    ).upper().strip()

    st.markdown(
        "<div style='font-size:11px;color:#6b7280;margin-top:-8px;margin-bottom:12px'>"
        "Try: SPY · AAPL · TSLA · MSFT · NVDA · BTC-USD · GLD · AMZN"
        "</div>", unsafe_allow_html=True
    )

    fetch_clicked = st.button("📡 Load Data", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎛️ Simulation Parameters")

    years = st.slider(
        "Time Horizon",
        min_value=1, max_value=10, value=1, step=1,
        help="How many years into the future to simulate.",
        format="%d yr"
    )

    n_sims = st.slider(
        "Number of Simulations",
        min_value=100, max_value=3000, value=500, step=100,
        help="More simulations = more accurate probability estimates, but slower.",
    )

    vol_option = st.selectbox(
        "Volatility Scenario",
        options=["Normal (1×)", "High (1.5×)", "Very High (2×)", "Crisis (2.5×)"],
        help="Adjust how wild the market swings are. 'Crisis' simulates conditions like 2008 or 2020.",
    )

    vol_map = {"Normal (1×)": 1.0, "High (1.5×)": 1.5, "Very High (2×)": 2.0, "Crisis (2.5×)": 2.5}
    vol_mult = vol_map[vol_option]

    run_clicked = st.button("▶ Run Simulation", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:10px;color:#6b7280;line-height:1.7;'>"
        "⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. "
        "Past performance does not guarantee future results. "
        "Not financial advice."
        "</div>", unsafe_allow_html=True
    )


# ── State management ────────────────────────────────────────────────────────────
if "asset" not in st.session_state:
    st.session_state.asset   = None
if "results" not in st.session_state:
    st.session_state.results = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None


# ── Fetch data on button click or ticker change ─────────────────────────────────
if fetch_clicked or (ticker_input and ticker_input != st.session_state.last_ticker and st.session_state.asset is None):
    with st.spinner(f"Fetching 5 years of data for **{ticker_input}**…"):
        data, err = fetch_data(ticker_input)
    if err:
        st.error(f"❌ {err}")
    else:
        st.session_state.asset       = data
        st.session_state.last_ticker = ticker_input
        st.session_state.results     = None

# Auto-load SPY on first visit
if st.session_state.asset is None and not fetch_clicked:
    with st.spinner("Loading SPY data…"):
        data, err = fetch_data("SPY")
    if not err:
        st.session_state.asset       = data
        st.session_state.last_ticker = "SPY"


# ── Asset strip ─────────────────────────────────────────────────────────────────
asset = st.session_state.asset
if asset:
    c1, c2, c3, c4 = st.columns([1.2, 3, 1.5, 1.5])
    with c1:
        st.markdown(
            f"<div style='background:rgba(126,232,162,0.1);border:1px solid rgba(126,232,162,0.25);"
            f"border-radius:10px;padding:10px 16px;font-size:20px;font-weight:500;"
            f"color:#7ee8a2;letter-spacing:.06em;text-align:center;margin-top:4px;'>{asset['ticker']}</div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div style='padding-top:6px;'>"
            f"<div style='font-size:14px;color:#e8eaf0;font-weight:500;'>{asset['name']}</div>"
            f"<div style='font-size:11px;color:#6b7280;margin-top:2px;'>"
            f"Daily μ: {asset['mu']*100:.3f}% · Daily σ: {asset['sigma']*100:.3f}% · 5-year history</div>"
            f"</div>", unsafe_allow_html=True
        )
    with c3:
        st.metric("Current Price", f"${asset['last_price']:,.2f}")
    with c4:
        ann_return = ((1 + asset['mu']) ** 252 - 1) * 100
        st.metric("Ann. Return (hist.)", f"{ann_return:.1f}%")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── Run simulation ──────────────────────────────────────────────────────────────
if asset and (run_clicked or st.session_state.results is None):
    with st.spinner(f"Running {n_sims:,} simulations…"):
        st.session_state.results = run_simulation(
            asset["last_price"], asset["mu"], asset["sigma"],
            years, n_sims, vol_mult
        )

results = st.session_state.results

if asset and results:
    final    = results["final"]
    last     = asset["last_price"]
    currency = asset.get("currency", "USD")

    p5  = float(np.percentile(final, 5))
    p25 = float(np.percentile(final, 25))
    p50 = float(np.percentile(final, 50))
    p75 = float(np.percentile(final, 75))
    p95 = float(np.percentile(final, 95))

    prob_gain   = float(np.mean(final > last) * 100)
    prob_20up   = float(np.mean(final > last * 1.2) * 100)
    prob_loss20 = float(np.mean(final < last * 0.8) * 100)

    # ── Outcome metrics ─────────────────────────────────────────────────────────
    st.markdown("#### Projected outcomes at end of horizon")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🐻 Bear case (5th pct)",   fmt(p5, currency),  pct_change(p5, last))
    m2.metric("📉 Low (25th pct)",         fmt(p25, currency), pct_change(p25, last))
    m3.metric("📊 Base case (median)",     fmt(p50, currency), pct_change(p50, last))
    m4.metric("📈 High (75th pct)",        fmt(p75, currency), pct_change(p75, last))
    m5.metric("🚀 Bull case (95th pct)",   fmt(p95, currency), pct_change(p95, last))

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Chart ───────────────────────────────────────────────────────────────────
    labels  = results["labels"]
    pcts    = results["pcts"]
    paths   = results["paths"]

    fig = go.Figure()

    # Individual paths (thin, semi-transparent)
    for i, path in enumerate(paths[:150]):
        fig.add_trace(go.Scatter(
            x=labels, y=path,
            mode="lines",
            line=dict(color="rgba(56,189,248,0.05)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Shaded IQR band (25–75)
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=list(pcts[75]) + list(pcts[25])[::-1],
        fill="toself",
        fillcolor="rgba(126,232,162,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Middle 50% (IQR)",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Percentile lines
    for p, color, name, dash in [
        (5,  "#f87171", "Bear (5th pct)",   "dash"),
        (50, "#fbbf24", "Base (median)",     "solid"),
        (95, "#7ee8a2", "Bull (95th pct)",   "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=labels, y=pcts[p],
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: $%{{y:,.0f}}<extra></extra>",
        ))

    # Starting price reference line
    fig.add_hline(
        y=last, line_dash="dot",
        line_color="rgba(255,255,255,0.2)", line_width=1,
        annotation_text=f"Today: ${last:,.2f}",
        annotation_font_color="#6b7280",
        annotation_font_size=11,
    )

    fig.update_layout(
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text=f"<b>{asset['ticker']}</b> — {n_sims:,} simulations · {years} year{'s' if years>1 else ''} · {vol_option}",
            font=dict(color="#6b7280", size=13, family="DM Mono"),
            x=0,
        ),
        legend=dict(
            font=dict(color="#6b7280", size=11, family="DM Mono"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", x=0, y=1.06,
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="#6b7280", size=10, family="DM Mono"),
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="#6b7280", size=10, family="DM Mono"),
            tickprefix="$",
            tickformat=",.0f",
            showline=False,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Probability cards ────────────────────────────────────────────────────────
    st.markdown("#### Probability breakdown")
    p1, p2, p3 = st.columns(3)
    p1.metric("📈 Chance of any gain",       f"{prob_gain:.1f}%",   help="Probability the stock ends above its current price")
    p2.metric("🚀 Chance of +20% or more",   f"{prob_20up:.1f}%",   help="Probability the stock gains more than 20%")
    p3.metric("📉 Chance of losing 20%+",    f"{prob_loss20:.1f}%", help="Probability the stock loses more than 20%")

    # ── Explainer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How does this work? (click to learn more)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**📊 Real historical data**")
            st.markdown(
                "We fetch 5 years of daily closing prices to measure how the stock has actually "
                "behaved — its average daily return and how volatile it has been.",
                unsafe_allow_html=False
            )
        with col2:
            st.markdown("**🎲 Random sampling**")
            st.markdown(
                "Each simulation generates a sequence of random daily returns — statistically "
                "similar to the stock's history — to produce one possible future price path."
            )
        with col3:
            st.markdown("**🔁 Thousands of futures**")
            st.markdown(
                "By running hundreds of these paths, we see the full distribution of outcomes — "
                "not just one guess, but a realistic range from bear to bull."
            )
        with col4:
            st.markdown("**⚠️ Important caveat**")
            st.markdown(
                "The model assumes returns follow a log-normal distribution (Geometric Brownian Motion). "
                "Past volatility doesn't guarantee future results. Always consult a professional."
            )

        st.markdown(
            f"**Current parameters for {asset['ticker']}:** "
            f"Daily μ = {asset['mu']*100:.4f}% · "
            f"Daily σ = {asset['sigma']*100:.4f}% · "
            f"Annualised return = {((1+asset['mu'])**252-1)*100:.2f}% · "
            f"Annualised volatility = {asset['sigma']*np.sqrt(252)*100:.2f}%"
        )

else:
    if not asset:
        st.markdown("""
        <div style='text-align:center;padding:5rem 2rem;'>
          <div style='font-size:3.5rem;margin-bottom:1rem;'>📈</div>
          <div style='font-family:Fraunces,serif;font-size:1.4rem;font-weight:300;color:#6b7280;'>
            Enter a ticker in the sidebar to get started
          </div>
          <div style='font-size:12px;color:#6b7280;opacity:.6;margin-top:.5rem;'>
            Try SPY, AAPL, TSLA, MSFT, BTC-USD…
          </div>
        </div>
        """, unsafe_allow_html=True)
