# ─────────────────────────────────────────────────────────────
# APEX Trading Intelligence Platform
# Paste this entire file into your Streamlit app
# ─────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="APEX — Trading Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .bull  { color: #00c851; font-weight: bold; }
    .bear  { color: #ff4444; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 22px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────

def calculate_rsi(closes, period=14):
    closes = pd.Series(closes, dtype=float)
    delta = closes.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def find_swing_highs(highs, lookback=3):
    highs = list(highs)
    swings = []
    for i in range(lookback, len(highs) - lookback):
        if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, lookback + 1)):
            swings.append((i, highs[i]))
    return swings


def find_swing_lows(lows, lookback=3):
    lows = list(lows)
    swings = []
    for i in range(lookback, len(lows) - lookback):
        if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, lookback + 1)):
            swings.append((i, lows[i]))
    return swings


def detect_trend(swing_highs, swing_lows):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'neutral'
    hh = swing_highs[-1][1] > swing_highs[-2][1]
    hl = swing_lows[-1][1]  > swing_lows[-2][1]
    lh = swing_highs[-1][1] < swing_highs[-2][1]
    ll = swing_lows[-1][1]  < swing_lows[-2][1]
    if hh and hl:
        return 'uptrend'
    if lh and ll:
        return 'downtrend'
    return 'neutral'


def detect_choch(swing_highs, swing_lows, closes, trend):
    closes = list(closes)
    if trend == 'downtrend' and swing_highs:
        last_idx, last_price = swing_highs[-1]
        for i in range(last_idx + 1, len(closes)):
            if closes[i] > last_price:
                return {'type': 'BULL_CHOCH', 'index': i,
                        'price': closes[i], 'level': last_price}
    elif trend == 'uptrend' and swing_lows:
        last_idx, last_price = swing_lows[-1]
        for i in range(last_idx + 1, len(closes)):
            if closes[i] < last_price:
                return {'type': 'BEAR_CHOCH', 'index': i,
                        'price': closes[i], 'level': last_price}
    return None


def detect_divergence(closes, rsi_series, swing_highs, swing_lows, min_sep=5):
    divs = []
    rsi_list = list(rsi_series)
    n = len(rsi_list)

    # Regular Bullish: Price LL, RSI HL
    for i in range(1, len(swing_lows)):
        idx1, p1 = swing_lows[i - 1]
        idx2, p2 = swing_lows[i]
        if idx2 - idx1 < min_sep or idx1 >= n or idx2 >= n:
            continue
        r1, r2 = rsi_list[idx1], rsi_list[idx2]
        if pd.isna(r1) or pd.isna(r2):
            continue
        if p2 < p1 and r2 > r1:
            divs.append({'type': 'BULL_DIV',
                         'idx1': idx1, 'idx2': idx2,
                         'p1': p1, 'p2': p2, 'r1': r1, 'r2': r2})

    # Regular Bearish: Price HH, RSI LH
    for i in range(1, len(swing_highs)):
        idx1, p1 = swing_highs[i - 1]
        idx2, p2 = swing_highs[i]
        if idx2 - idx1 < min_sep or idx1 >= n or idx2 >= n:
            continue
        r1, r2 = rsi_list[idx1], rsi_list[idx2]
        if pd.isna(r1) or pd.isna(r2):
            continue
        if p2 > p1 and r2 < r1:
            divs.append({'type': 'BEAR_DIV',
                         'idx1': idx1, 'idx2': idx2,
                         'p1': p1, 'p2': p2, 'r1': r1, 'r2': r2})
    return divs


def detect_squeeze(closes, highs, lows, bb_period=20, kc_mult=1.5):
    if len(closes) < bb_period + 1:
        return {'squeeze_on': False, 'momentum': 0, 'momentum_dir': 'neutral'}
    c = np.array(closes[-bb_period:], dtype=float)
    h = np.array(highs[-(bb_period + 1):], dtype=float)
    lo = np.array(lows[-(bb_period + 1):], dtype=float)
    cl_prev = np.array(closes[-(bb_period + 1):-1], dtype=float)

    mean    = c.mean()
    std     = c.std()
    upper_bb = mean + 2 * std
    lower_bb = mean - 2 * std

    tr = np.maximum(h[1:] - lo[1:],
         np.maximum(np.abs(h[1:] - cl_prev),
                    np.abs(lo[1:] - cl_prev)))
    atr = tr.mean()

    squeeze_on = (upper_bb < mean + kc_mult * atr) and \
                 (lower_bb > mean - kc_mult * atr)
    momentum   = float(closes[-1]) - mean

    return {
        'squeeze_on': squeeze_on,
        'momentum': round(momentum, 4),
        'momentum_dir': 'bullish' if momentum > 0 else 'bearish'
    }


def calculate_adx(highs, lows, closes, period=14):
    try:
        h = pd.Series(highs, dtype=float)
        l = pd.Series(lows, dtype=float)
        c = pd.Series(closes, dtype=float)

        plus_dm  = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm]   = 0
        minus_dm[minus_dm < plus_dm]  = 0     # only one can win per bar

        tr  = pd.concat([h - l,
                         (h - c.shift()).abs(),
                         (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period).mean()

        pdi = 100 * plus_dm.ewm(span=period).mean() / (atr + 1e-10)
        mdi = 100 * minus_dm.ewm(span=period).mean() / (atr + 1e-10)
        dx  = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
        adx = dx.ewm(span=period).mean()
        return float(adx.iloc[-1]) if not adx.empty else 20.0
    except Exception:
        return 20.0


def score_setup(trend, choch, divergences, squeeze, adx_val, vol_ratio):
    score   = 0
    signals = []

    if trend == 'uptrend':
        score += 20;  signals.append('✅ Uptrend Structure (HH/HL)')
    elif trend == 'downtrend':
        score += 15;  signals.append('✅ Downtrend Structure (LH/LL)')

    if choch:
        score += 25
        signals.append('🔼 Bull CHoCH' if choch['type'] == 'BULL_CHOCH' else '🔽 Bear CHoCH')

    if divergences:
        score += 20
        d = divergences[-1]
        signals.append('📈 Bullish Divergence' if d['type'] == 'BULL_DIV'
                        else '📉 Bearish Divergence')

    if squeeze['squeeze_on']:
        score += 15
        signals.append(f'🟡 Squeeze Firing ({squeeze["momentum_dir"]})')

    if adx_val > 25:
        score += 10;  signals.append(f'💪 Strong Trend (ADX {adx_val:.0f})')
    elif adx_val > 20:
        score += 5

    if vol_ratio > 1.5:
        score += 5;   signals.append(f'📊 Volume Spike ({vol_ratio:.1f}x avg)')

    if score >= 75:   grade = 'A'
    elif score >= 55: grade = 'B'
    elif score >= 40: grade = 'C'
    else:             grade = 'D'

    return score, grade, signals


def generate_setup(ticker, df, direction, setup_type, confidence, grade, signals):
    price = float(df['Close'].iloc[-1])
    atr   = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])

    sh = find_swing_highs(df['High'].values.flatten())
    sl = find_swing_lows(df['Low'].values.flatten())

    if direction == 'long':
        entry_low  = round(price * 0.998, 2)
        entry_high = round(price * 1.005, 2)
        stop_loss  = round((sl[-1][1] if sl else price * 0.95) - atr * 0.3, 2)
        risk       = entry_high - stop_loss
        target_1   = round(entry_high + risk * 2.0, 2)
        target_2   = round(entry_high + risk * 3.5, 2)
    else:
        entry_high = round(price * 1.002, 2)
        entry_low  = round(price * 0.995, 2)
        stop_loss  = round((sh[-1][1] if sh else price * 1.05) + atr * 0.3, 2)
        risk       = stop_loss - entry_low
        target_1   = round(entry_low - risk * 2.0, 2)
        target_2   = round(entry_low - risk * 3.5, 2)

    rr     = round(abs(target_1 - entry_high) / (abs(entry_high - stop_loss) + 1e-10), 1)
    strike = round(target_1 / 5) * 5
    opt    = 'Calls' if direction == 'long' else 'Puts'
    expiry = (datetime.now() + timedelta(days=45)).strftime('%B %Y')

    return {
        'ticker':        ticker,
        'setup_type':    setup_type,
        'direction':     direction,
        'confidence':    confidence,
        'grade':         grade,
        'current_price': price,
        'entry_low':     entry_low,
        'entry_high':    entry_high,
        'stop_loss':     stop_loss,
        'target_1':      target_1,
        'target_2':      target_2,
        'risk_reward':   rr,
        'signals':       signals,
        'options':       f'{expiry} ${strike} {opt}',
        'atr':           round(atr, 2),
    }


def analyze_ticker(ticker, period='6mo', interval='1d'):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None

        # Flatten MultiIndex columns (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if len(df) < 30:
            return None

        closes  = df['Close'].values.flatten().astype(float)
        highs   = df['High'].values.flatten().astype(float)
        lows    = df['Low'].values.flatten().astype(float)
        volumes = df['Volume'].values.flatten().astype(float)

        rsi        = calculate_rsi(closes)
        sh         = find_swing_highs(highs)
        sl         = find_swing_lows(lows)
        trend      = detect_trend(sh, sl)
        choch      = detect_choch(sh, sl, closes, trend)
        divs       = detect_divergence(closes, rsi, sh, sl)
        squeeze    = detect_squeeze(closes, highs, lows)
        adx        = calculate_adx(highs, lows, closes)
        avg_vol    = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        vol_ratio  = float(volumes[-1]) / (float(avg_vol) + 1e-10)

        score, grade, sigs = score_setup(trend, choch, divs, squeeze, adx, vol_ratio)
        if grade == 'D':
            return None

        # Pick direction and setup label
        if choch and choch['type'] == 'BULL_CHOCH':
            direction, setup_type = 'long',  'Bull CHoCH Reversal'
        elif choch and choch['type'] == 'BEAR_CHOCH':
            direction, setup_type = 'short', 'Bear CHoCH Reversal'
        elif divs and divs[-1]['type'] == 'BULL_DIV':
            direction, setup_type = 'long',  'Bullish Divergence'
        elif divs and divs[-1]['type'] == 'BEAR_DIV':
            direction, setup_type = 'short', 'Bearish Divergence'
        elif squeeze['squeeze_on'] and trend == 'uptrend':
            direction, setup_type = 'long',  'Bullish Squeeze'
        elif squeeze['squeeze_on'] and trend == 'downtrend':
            direction, setup_type = 'short', 'Bearish Squeeze'
        elif trend == 'uptrend':
            direction, setup_type = 'long',  'Momentum Continuation'
        else:
            direction, setup_type = 'short', 'Bearish Momentum'

        setup = generate_setup(ticker, df, direction, setup_type, score, grade, sigs)
        setup.update({'df': df, 'rsi': rsi, 'swing_highs': sh,
                      'swing_lows': sl, 'divergences': divs,
                      'choch': choch, 'trend': trend, 'squeeze': squeeze})
        return setup

    except Exception:
        return None


# ─────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────

def build_chart(setup):
    df  = setup['df']
    rsi = list(setup['rsi'])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.70, 0.30],
        subplot_titles=(f"{setup['ticker']} — {setup['setup_type']}", "RSI (14)")
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'].values.flatten(),
        high=df['High'].values.flatten(),
        low=df['Low'].values.flatten(),
        close=df['Close'].values.flatten(),
        name='Price',
        increasing_line_color='#00c851',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='#00c851',
        decreasing_fillcolor='#ff4444',
    ), row=1, col=1)

    # ── Entry zone ──
    fig.add_hrect(
        y0=setup['entry_low'], y1=setup['entry_high'],
        fillcolor='rgba(0, 200, 81, 0.12)', line_width=0,
        annotation_text='Entry', annotation_position='right',
        annotation_font_color='#00c851', row=1, col=1
    )

    # ── Stop loss ──
    fig.add_hline(
        y=setup['stop_loss'], line_dash='dash', line_color='#ff4444', line_width=1.5,
        annotation_text=f"Stop ${setup['stop_loss']}",
        annotation_font_color='#ff4444',
        annotation_position='right', row=1, col=1
    )

    # ── Targets ──
    fig.add_hline(
        y=setup['target_1'], line_dash='dot', line_color='#00ff88', line_width=1.5,
        annotation_text=f"T1 ${setup['target_1']}",
        annotation_font_color='#00ff88',
        annotation_position='right', row=1, col=1
    )
    fig.add_hline(
        y=setup['target_2'], line_dash='dot', line_color='#00c851', line_width=1,
        annotation_text=f"T2 ${setup['target_2']}",
        annotation_font_color='#00c851',
        annotation_position='right', row=1, col=1
    )

    # ── CHoCH marker ──
    if setup['choch']:
        c = setup['choch']
        idx = c['index']
        if idx < len(df):
            is_bull = c['type'] == 'BULL_CHOCH'
            fig.add_trace(go.Scatter(
                x=[df.index[idx]],
                y=[c['price']],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-up' if is_bull else 'triangle-down',
                    size=16,
                    color='#00c851' if is_bull else '#ff4444',
                    line=dict(color='white', width=1)
                ),
                text=['CHoCH ↑' if is_bull else 'CHoCH ↓'],
                textposition='top center' if is_bull else 'bottom center',
                textfont=dict(size=11, color='white'),
                name='CHoCH', showlegend=True
            ), row=1, col=1)

    # ── Swing high / low dots ──
    for idx, price in setup['swing_highs'][-5:]:
        if idx < len(df):
            fig.add_trace(go.Scatter(
                x=[df.index[idx]], y=[price],
                mode='markers',
                marker=dict(symbol='circle', size=6, color='#ff9900'),
                name='Swing High', showlegend=False
            ), row=1, col=1)

    for idx, price in setup['swing_lows'][-5:]:
        if idx < len(df):
            fig.add_trace(go.Scatter(
                x=[df.index[idx]], y=[price],
                mode='markers',
                marker=dict(symbol='circle', size=6, color='#00aaff'),
                name='Swing Low', showlegend=False
            ), row=1, col=1)

    # ── RSI line ──
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        line=dict(color='#7b68ee', width=1.5),
        name='RSI', hovertemplate='RSI: %{y:.1f}'
    ), row=2, col=1)

    fig.add_hrect(y0=70, y1=100,
                  fillcolor='rgba(255,68,68,0.07)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30,
                  fillcolor='rgba(0,200,81,0.07)', line_width=0, row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='rgba(255,68,68,0.4)',
                  line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='rgba(0,200,81,0.4)',
                  line_width=1, row=2, col=1)

    # ── Divergence lines on RSI pane ──
    for div in setup['divergences'][-3:]:
        i1, i2 = div['idx1'], div['idx2']
        if i1 < len(df) and i2 < len(df):
            color = '#00c851' if div['type'] == 'BULL_DIV' else '#ff4444'
            fig.add_trace(go.Scatter(
                x=[df.index[i1], df.index[i2]],
                y=[div['r1'], div['r2']],
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=5, color=color),
                name='Divergence', showlegend=False
            ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#151820',
        height=640,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=90, t=40, b=0),
        font=dict(color='#cccccc', size=12),
        xaxis2=dict(showgrid=True, gridcolor='#1e2230'),
        yaxis=dict(showgrid=True, gridcolor='#1e2230'),
        yaxis2=dict(showgrid=True, gridcolor='#1e2230', range=[0, 100]),
    )
    return fig


# ─────────────────────────────────────────
# AI ADVISOR
# ─────────────────────────────────────────

def ask_advisor(question, context, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_prompt = """You are APEX, an elite AI market strategist and trading analyst.

You think and communicate like a senior portfolio manager at a top hedge fund.
Be direct, precise, and confident. Think in terms of risk/reward and probability.
Never give vague answers — give specific levels, specific strategies, specific reasoning.
Only reference data provided to you in the context below.
Keep responses concise but complete. Use bullet points when listing multiple setups.
If no scan data is available, say so and give general market guidance instead."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": f"Market Context:\n{context}\n\nUser Question: {question}"}
            ],
            max_tokens=600,
            temperature=0.4
        )
        return response.choices[0].message.content

    except Exception as e:
        return (f"⚠️ AI Advisor error: {e}\n\n"
                "Make sure your OpenAI API key is correct and has credits.")


def build_context(setups):
    if not setups:
        return "No scan results available. Run the scanner first."
    ctx = f"Date: {datetime.now().strftime('%B %d, %Y')}\n\nTop Setups:\n"
    for i, s in enumerate(setups[:5], 1):
        ctx += (f"\n#{i} {s['ticker']} | {s['setup_type']} | {s['direction'].upper()}"
                f" | Grade {s['grade']} | Conf {s['confidence']}/100"
                f" | Price ${s['current_price']}"
                f" | Entry ${s['entry_low']}–${s['entry_high']}"
                f" | Stop ${s['stop_loss']} | T1 ${s['target_1']} | T2 ${s['target_2']}"
                f" | R:R {s['risk_reward']}:1 | Options: {s['options']}"
                f" | Signals: {', '.join(s['signals'])}\n")
    return ctx


# ─────────────────────────────────────────
# DEFAULT WATCHLIST
# ─────────────────────────────────────────

DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA',
    'SPY',  'QQQ',  'META', 'AMZN', 'GOOGL',
    'JPM',  'GS',   'XOM',  'CCJ',  'GLD',
    'MSTR', 'COIN', 'PLTR', 'SOFI', 'SQ'
]


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────

def main():

    # ── SIDEBAR ──────────────────────────
    with st.sidebar:
        st.markdown("## 📈 APEX Intelligence")
        st.caption("AI-Powered Trading Signals")
        st.divider()

        page = st.radio("Navigate",
                        ["🔍 Scanner", "📊 Chart Analysis",
                         "🤖 AI Advisor", "⚙️ Settings"],
                        label_visibility="collapsed")
        st.divider()

        st.markdown("#### OpenAI API Key")
        openai_key = st.text_input("Key", type="password",
                                   placeholder="sk-...",
                                   label_visibility="collapsed",
                                   help="Needed for the AI Advisor tab only")
        st.caption("Get a key at platform.openai.com")
        st.divider()

        st.markdown("#### Watchlist")
        raw = st.text_area("Tickers (one per line)",
                           value='\n'.join(DEFAULT_TICKERS),
                           height=220,
                           label_visibility="collapsed")
        tickers = [t.strip().upper()
                   for t in raw.replace(',', '\n').split('\n') if t.strip()]
        st.divider()

        st.markdown("#### Scan Settings")
        period   = st.selectbox("Lookback Period", ['3mo', '6mo', '1y'], index=1)
        interval = st.selectbox("Bar Interval",    ['1d',  '1wk'],       index=0)

    # ── SCANNER ──────────────────────────
    if '🔍 Scanner' in page:
        st.title("🔍 Market Scanner")
        st.caption("Detects CHoCH, divergence, squeezes, and trend setups across your watchlist.")

        c1, c2, c3 = st.columns([2, 1, 1])
        run          = c1.button("▶  Run Scan", type="primary", use_container_width=True)
        grade_filter = c2.selectbox("Min Grade", ['All', 'A only', 'A & B'])
        dir_filter   = c3.selectbox("Direction",  ['All', 'Long Only', 'Short Only'])

        if run:
            results  = []
            bar      = st.progress(0, text="Starting scan…")
            for i, t in enumerate(tickers):
                bar.progress((i + 1) / len(tickers), text=f"Scanning {t}…")
                r = analyze_ticker(t, period=period, interval=interval)
                if r:
                    results.append(r)
            bar.empty()

            if grade_filter == 'A only':
                results = [r for r in results if r['grade'] == 'A']
            elif grade_filter == 'A & B':
                results = [r for r in results if r['grade'] in ('A', 'B')]
            if dir_filter == 'Long Only':
                results = [r for r in results if r['direction'] == 'long']
            elif dir_filter == 'Short Only':
                results = [r for r in results if r['direction'] == 'short']

            results.sort(key=lambda x: x['confidence'], reverse=True)
            st.session_state['scan_results'] = results
            st.session_state['scan_context'] = build_context(results)

        if 'scan_results' in st.session_state:
            results = st.session_state['scan_results']
            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Setups",  len(results))
            m2.metric("Grade A",       sum(1 for r in results if r['grade'] == 'A'))
            m3.metric("Long Setups",   sum(1 for r in results if r['direction'] == 'long'))
            m4.metric("Short Setups",  sum(1 for r in results if r['direction'] == 'short'))
            st.divider()

            if not results:
                st.warning("No qualifying setups found. Expand your watchlist or lower the grade filter.")
            else:
                for rank, s in enumerate(results, 1):
                    gem   = {'A': '🟢', 'B': '🟡', 'C': '🔴'}.get(s['grade'], '⚪')
                    arrow = '🔼' if s['direction'] == 'long' else '🔽'
                    label = (f"#{rank}  {s['ticker']}  {gem} Grade {s['grade']}  "
                             f"{arrow} {s['setup_type']}  |  "
                             f"Conf {s['confidence']}/100  |  R:R {s['risk_reward']}:1")

                    with st.expander(label):
                        col1, col2, col3 = st.columns(3)

                        col1.markdown("**Entry Zone**")
                        col1.markdown(f"`${s['entry_low']} – ${s['entry_high']}`")
                        col1.markdown("**Stop Loss**")
                        col1.markdown(f"🔴 `${s['stop_loss']}`")

                        col2.markdown("**Target 1**")
                        col2.markdown(f"🟢 `${s['target_1']}`")
                        col2.markdown("**Target 2**")
                        col2.markdown(f"🟢 `${s['target_2']}`")

                        col3.markdown("**Options Idea**")
                        col3.markdown(f"📋 {s['options']}")
                        col3.markdown("**Risk / Reward**")
                        col3.markdown(f"⚖️ {s['risk_reward']} : 1")

                        st.markdown("**Signals Detected:**  " +
                                    "  ·  ".join(s['signals']))

                        if st.button(f"Open Chart → {s['ticker']}",
                                     key=f"btn_{s['ticker']}_{rank}"):
                            st.session_state['chart_ticker'] = s['ticker']
                            st.session_state['chart_setup']  = s
                            st.rerun()
        else:
            st.info("Click **Run Scan** above to analyze your watchlist.")

    # ── CHART ANALYSIS ───────────────────
    elif '📊 Chart Analysis' in page:
        st.title("📊 Chart Analysis")

        col1, col2, col3 = st.columns([3, 1, 1])
        ticker_in = col1.text_input(
            "Ticker", value=st.session_state.get('chart_ticker', 'AAPL'),
            label_visibility="collapsed", placeholder="Enter ticker e.g. AAPL"
        ).upper().strip()
        cp = col2.selectbox("Period",   ['3mo', '6mo', '1y'], index=1,
                            label_visibility="collapsed")
        ci = col3.selectbox("Interval", ['1d', '1wk'],        index=0,
                            label_visibility="collapsed")

        go_btn = st.button("Analyze", type="primary")

        if go_btn:
            st.session_state.pop('chart_setup', None)
            with st.spinner(f"Analyzing {ticker_in}…"):
                setup = analyze_ticker(ticker_in, period=cp, interval=ci)
            st.session_state['chart_setup']  = setup
            st.session_state['chart_ticker'] = ticker_in

        setup = st.session_state.get('chart_setup')

        if setup:
            gem = {'A': '🏆', 'B': '🥈', 'C': '🥉'}.get(setup['grade'], '')
            d   = '🟢 LONG' if setup['direction'] == 'long' else '🔴 SHORT'

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Current Price", f"${setup['current_price']}")
            m2.metric("Grade",         f"{gem} {setup['grade']}")
            m3.metric("Confidence",    f"{setup['confidence']}/100")
            m4.metric("R : R",         f"{setup['risk_reward']} : 1")
            m5.metric("Direction",     d)

            st.plotly_chart(build_chart(setup), use_container_width=True)

            st.divider()
            left, right = st.columns(2)

            with left:
                st.markdown("### 📋 Trade Plan")
                st.table(pd.DataFrame({
                    'Level': ['Entry Zone', 'Stop Loss', 'Target 1', 'Target 2', 'Options Idea'],
                    'Value': [
                        f"${setup['entry_low']} – ${setup['entry_high']}",
                        f"${setup['stop_loss']}",
                        f"${setup['target_1']}",
                        f"${setup['target_2']}",
                        setup['options']
                    ]
                }))

            with right:
                st.markdown("### 🔎 Signals")
                for sig in setup['signals']:
                    st.markdown(f"- {sig}")
                st.markdown(f"**Trend:** {setup['trend'].capitalize()}")
                st.markdown(f"**Squeeze:** {'🟡 ON — coiling' if setup['squeeze']['squeeze_on'] else 'Off'}")

        elif st.session_state.get('chart_ticker'):
            st.error(f"Could not analyse **{ticker_in}**. "
                     "Check the symbol and try a longer period.")
        else:
            st.info("Enter a ticker above and click **Analyze**.")

    # ── AI ADVISOR ───────────────────────
    elif '🤖 AI Advisor' in page:
        st.title("🤖 APEX AI Advisor")
        st.caption("Powered by GPT-4o — asks are answered using your latest scan data.")

        if not openai_key:
            st.warning("⚠️ Paste your OpenAI API key in the sidebar to activate the AI Advisor.")
            st.markdown("""
**Quick setup:**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **Create new secret key**
3. Copy it and paste it into the sidebar field

Cost at personal usage: typically **$2–10/month**.
""")
            return

        st.markdown("**Quick questions — click one or type your own:**")
        qc1, qc2, qc3 = st.columns(3)
        preset = None
        if qc1.button("🏆 Best setups right now?",    use_container_width=True):
            preset = "What are the best setups right now? Give me entry, stop, and target for each."
        if qc2.button("⚖️ Best risk/reward trade?",   use_container_width=True):
            preset = "Which setup has the best risk/reward ratio and why?"
        if qc3.button("📋 Explain the top setup",     use_container_width=True):
            preset = "Walk me through the top setup in detail — structure, signals, and what I should do."

        question = st.text_input("Ask APEX anything…",
                                 value=preset or '',
                                 placeholder="e.g. Is NVDA bullish right now?")

        if st.button("Ask APEX →", type="primary") and question:
            ctx = st.session_state.get('scan_context',
                  "No scan data yet. Run the scanner for context-aware answers.")
            with st.spinner("APEX is thinking…"):
                answer = ask_advisor(question, ctx, openai_key)
            st.divider()
            st.markdown("### APEX says:")
            st.markdown(answer)

    # ── SETTINGS ─────────────────────────
    elif '⚙️ Settings' in page:
        st.title("⚙️ Settings")
        st.info("Default parameters are optimised for daily swing trading. "
                "Custom parameter saving is coming in a future update.")

        st.markdown("### Signal Engine")
        c1, c2 = st.columns(2)
        c1.slider("RSI Period",                    7,  21,  14)
        c1.slider("Swing Point Lookback (bars)",   2,   5,   3)
        c1.slider("Min Divergence Gap (bars)",     3,  10,   5)
        c2.slider("ADX Trend Strength Threshold", 15,  35,  25)
        c2.slider("Volume Spike Multiplier",      1.0, 3.0, 1.5)
        c2.slider("Squeeze KC Multiplier",        1.0, 2.0, 1.5)

        st.markdown("### Scoring & Grades")
        st.slider("Grade A — minimum score", 60, 90, 75)
        st.slider("Grade B — minimum score", 40, 70, 55)

        st.markdown("### About")
        st.markdown("""
| Component | Detail |
|---|---|
| Data source | Yahoo Finance (yfinance) — free, no API key |
| AI model | OpenAI GPT-4o |
| Signals | CHoCH, BOS, RSI Divergence, Squeeze, ADX, Volume |
| Timeframes | Daily and Weekly (intraday coming soon) |
| Universe | Your watchlist (edit in sidebar) |
""")


if __name__ == '__main__':
    main()
