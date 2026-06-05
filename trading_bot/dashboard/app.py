"""Streamlit dashboard for live monitoring of the paper trading bot.

Launch from the project root:  streamlit run dashboard/app.py

Shows open positions with live P&L, today's trade log (with AI reasoning),
the per-ticker sentiment breakdown, the cumulative equity curve, win rate by
strategy, and the most recent AI analysis output.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Allow `streamlit run dashboard/app.py` to import the project packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from config import MARKETDATA_BASE_URL, settings  # noqa: E402
from database.db import Database  # noqa: E402

st.set_page_config(page_title="AI Options Trading Bot", layout="wide")


@st.cache_resource
def get_db() -> Database:
    return Database()


def fetch_option_quotes(symbols: list[str]) -> dict[str, float]:
    """Synchronously re-quote open option symbols via marketdata.app (best effort)."""
    if not symbols or not settings.marketdata_api_key:
        return {}
    headers = {
        "Authorization": f"Bearer {settings.marketdata_api_key}",
        "Accept": "application/json",
    }
    out: dict[str, float] = {}
    for sym in symbols:
        try:
            resp = httpx.get(
                f"{MARKETDATA_BASE_URL}/v1/options/quotes/{sym}/",
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001
            continue
        if data.get("s") != "ok":
            continue

        def first(key: str):
            v = data.get(key)
            return v[0] if isinstance(v, list) and v else None

        mid, bid, ask, last = first("mid"), first("bid"), first("ask"), first("last")
        if mid is not None:
            out[sym] = round(float(mid), 4)
        elif bid is not None and ask is not None and ask > 0:
            out[sym] = round((bid + ask) / 2, 4)
        elif last is not None:
            out[sym] = float(last)
    return out


def render_header(db: Database) -> None:
    equity_curve = db.get_equity_curve()
    realized = db.realized_pnl_total()
    equity = equity_curve[-1]["equity"] if equity_curve else settings.starting_capital
    cash = equity_curve[-1]["cash"] if equity_curve else settings.starting_capital
    total_return = (equity - settings.starting_capital) / settings.starting_capital * 100

    st.title("🤖 AI Options Paper Trading Bot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${equity:,.0f}", f"{total_return:+.2f}%")
    c2.metric("Cash", f"${cash:,.0f}")
    c3.metric("Realized P&L", f"${realized:,.0f}")
    c4.metric("Open Positions", db.count_open_positions())


def render_open_positions(db: Database) -> None:
    st.subheader("📌 Open Positions (live P&L)")
    trades = db.get_open_trades()
    if not trades:
        st.info("No open positions.")
        return

    quotes = fetch_option_quotes([t.option_symbol for t in trades])
    rows = []
    for t in trades:
        current = quotes.get(t.option_symbol, t.entry_price)
        unreal = t.unrealized_pnl(current)
        unreal_pct = (current - t.entry_price) / t.entry_price * 100 if t.entry_price else 0
        rows.append(
            {
                "Ticker": t.ticker,
                "Strategy": t.strategy,
                "Dir": t.direction.value,
                "Contracts": t.contracts,
                "Entry $": round(t.entry_price, 2),
                "Current $": round(current, 2),
                "Stop $": round(t.stop_price, 2) if t.stop_price else None,
                "Expiry": t.expiration,
                "Unreal P&L $": round(unreal, 2),
                "Unreal %": round(unreal_pct, 1),
                "Conf %": round(t.confidence, 0),
                "Sentiment": round(t.sentiment_score, 2),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_trade_log(db: Database) -> None:
    st.subheader("📒 Trade Log (with AI reasoning)")
    trades = db.get_trades(limit=50)
    if not trades:
        st.info("No trades recorded yet.")
        return
    rows = []
    for t in trades:
        rows.append(
            {
                "Opened": t.opened_at.strftime("%Y-%m-%d %H:%M") if t.opened_at else "",
                "Ticker": t.ticker,
                "Strategy": t.strategy,
                "Dir": t.direction.value,
                "Status": t.status.value,
                "Entry $": round(t.entry_price, 2),
                "Exit $": round(t.exit_price, 2) if t.exit_price is not None else None,
                "P&L $": round(t.pnl, 2) if t.pnl is not None else None,
                "Conf %": round(t.confidence, 0),
                "Close reason": t.close_reason or "",
                "Reasoning": t.reasoning,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_sentiment(db: Database) -> None:
    st.subheader("🧭 Sentiment Breakdown per Ticker")
    rows = []
    for ticker in settings.watchlist:
        s = db.get_latest_sentiment(ticker)
        if not s:
            continue
        rows.append(
            {
                "Ticker": ticker,
                "Unified": round(s["unified_score"], 2),
                "Bias": s["bias"],
                "News": round(s["news_score"] or 0, 2),
                "IV rank": round(s["iv_rank_score"] or 0, 2),
                "Put/Call": round(s["put_call_score"] or 0, 2),
                "VIX": round(s["vix_score"] or 0, 2),
                "Flow": round(s["options_flow_score"] or 0, 2),
                "VIX level": s["vix_level"],
                "Fear&Greed": s["fear_greed"],
            }
        )
    if not rows:
        st.info("No sentiment scores computed yet.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = px.bar(
        df, x="Ticker", y="Unified", color="Unified",
        color_continuous_scale="RdYlGn", range_color=[-1, 1],
        title="Unified sentiment by ticker",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_equity_curve(db: Database) -> None:
    st.subheader("📈 Equity Curve")
    curve = db.get_equity_curve()
    if not curve:
        st.info("No equity history yet — run a cycle to populate.")
        return
    df = pd.DataFrame(curve)
    df["created_at"] = pd.to_datetime(df["created_at"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["created_at"], y=df["equity"],
                             mode="lines+markers", name="Equity"))
    fig.add_hline(
        y=settings.starting_capital, line_dash="dash",
        annotation_text="Starting capital",
    )
    fig.update_layout(yaxis_title="Equity ($)", xaxis_title="Time")
    st.plotly_chart(fig, use_container_width=True)


def render_strategy_stats(db: Database) -> None:
    st.subheader("🏆 Win Rate by Strategy")
    stats = db.strategy_stats()
    if not stats:
        st.info("No closed trades yet.")
        return
    df = pd.DataFrame(stats)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="strategy", y="win_rate", title="Win rate %",
                     color="win_rate", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(df, x="strategy", y="total_pnl", title="Total P&L $",
                      color="total_pnl", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig2, use_container_width=True)


def render_last_analysis(db: Database) -> None:
    st.subheader("🧠 Last AI Analysis")
    signal = db.get_latest_signal()
    if not signal:
        st.info("No AI analysis recorded yet.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Ticker", signal["ticker"])
    c2.metric("Direction", signal["direction"])
    c3.metric("Confidence", f"{signal['confidence']:.0f}%")
    st.write(f"**Strategy:** {signal['strategy']}")
    st.write(f"**Executed:** {'✅ yes' if signal['executed'] else '⏸️ no'}")
    st.write(f"**Reasoning:** {signal['reasoning']}")
    st.caption(f"Recorded at {signal['created_at']}")


def main() -> None:
    db = get_db()

    with st.sidebar:
        st.header("Controls")
        st.write(f"Watchlist: {', '.join(settings.watchlist)}")
        st.write(f"Starting capital: ${settings.starting_capital:,.0f}")
        st.write(f"Min confidence: {settings.min_confidence}%")
        st.write(f"Max positions: {settings.max_open_positions}")
        if st.button("🔄 Refresh"):
            st.rerun()
        st.caption(f"Loaded {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

    render_header(db)
    render_open_positions(db)
    render_equity_curve(db)
    col_a, col_b = st.columns(2)
    with col_a:
        render_strategy_stats(db)
    with col_b:
        render_last_analysis(db)
    render_sentiment(db)
    render_trade_log(db)


# Streamlit runs this file as __main__; the guard also keeps imports side-effect free.
if __name__ == "__main__":
    main()
