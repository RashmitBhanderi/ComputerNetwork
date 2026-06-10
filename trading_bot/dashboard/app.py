"""Streamlit command-center dashboard for the AI options paper trading bot.

Launch from the project root:  streamlit run dashboard/app.py

Surfaces the full decision pipeline: account equity, live market context
(VIX + Fear & Greed), the per-ticker sentiment breakdown, every AI decision
with its confidence vs the trade threshold, open positions with live P&L, the
equity curve, win rate by strategy, and the full trade log.
"""
from __future__ import annotations

import json
import os
import sys
import time
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

st.set_page_config(
    page_title="AI Options Trading Bot",
    page_icon="🤖",
    layout="wide",
)

# ─── light styling ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      div[data-testid="stMetric"] {
        background: #11161d; border: 1px solid #232b36; border-radius: 12px;
        padding: 14px 16px;
      }
      .badge {display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:0.8rem; font-weight:600;}
      .bull {background:#0f3d2e; color:#3ddc84;}
      .bear {background:#3d1414; color:#ff6b6b;}
      .neut {background:#2a2f38; color:#c0c6cf;}
    </style>
    """,
    unsafe_allow_html=True,
)

_COMPONENTS = {
    "news_score": "News",
    "iv_rank_score": "IV rank",
    "put_call_score": "Put/Call",
    "vix_score": "VIX",
    "options_flow_score": "Flow",
}


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


def _bias_badge(bias: str) -> str:
    cls = {"bullish": "bull", "bearish": "bear"}.get(bias, "neut")
    return f'<span class="badge {cls}">{bias}</span>'


# ─── data assembly ───────────────────────────────────────────────────────────
def latest_market_context(db: Database) -> dict:
    """Newest VIX / Fear & Greed across the watchlist's sentiment rows."""
    newest = None
    for t in settings.watchlist:
        s = db.get_latest_sentiment(t)
        if s and (newest is None or (s["created_at"] or "") > (newest["created_at"] or "")):
            newest = s
    if not newest:
        return {"vix": None, "fear_greed": None, "as_of": None}
    return {
        "vix": newest.get("vix_level"),
        "fear_greed": newest.get("fear_greed"),
        "as_of": newest.get("created_at"),
    }


def watchlist_rows(db: Database) -> list[dict]:
    rows = []
    for ticker in settings.watchlist:
        sent = db.get_latest_sentiment(ticker) or {}
        sig = db.get_latest_signal(ticker) or {}
        levels = {}
        if sig.get("raw_json"):
            try:
                levels = json.loads(sig["raw_json"])
            except (json.JSONDecodeError, TypeError):
                levels = {}
        rows.append({"ticker": ticker, "sentiment": sent, "signal": sig, "levels": levels})
    return rows


# ─── sections ────────────────────────────────────────────────────────────────
def render_header(db: Database) -> None:
    curve = db.get_equity_curve()
    realized = db.realized_pnl_total()
    equity = curve[-1]["equity"] if curve else settings.starting_capital
    cash = curve[-1]["cash"] if curve else settings.starting_capital
    ret = (equity - settings.starting_capital) / settings.starting_capital * 100
    ctx = latest_market_context(db)

    st.title("🤖 AI Options Paper Trading Bot")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Equity", f"${equity:,.0f}", f"{ret:+.2f}%")
    c2.metric("Cash", f"${cash:,.0f}")
    c3.metric("Realized P&L", f"${realized:,.0f}")
    c4.metric("Open Positions", f"{db.count_open_positions()} / {settings.max_open_positions}")
    vix = ctx["vix"]
    c5.metric(
        "VIX", f"{vix:.2f}" if vix is not None else "—",
        "⚠️ halt" if (vix is not None and vix > settings.vix_halt_threshold) else "ok",
        delta_color="inverse",
    )
    fg = ctx["fear_greed"]
    fg_label = "—"
    if fg is not None:
        mood = ("Extreme Fear" if fg < 25 else "Fear" if fg < 45
                else "Neutral" if fg < 55 else "Greed" if fg < 75 else "Extreme Greed")
        fg_label = f"{fg:.0f}"
    c6.metric("Fear & Greed", fg_label, mood if fg is not None else None,
              delta_color="off")
    if ctx["as_of"]:
        st.caption(f"Market context as of {ctx['as_of']}")


def render_watchlist(db: Database) -> None:
    st.subheader("🎯 Watchlist — sentiment & AI decisions")
    rows = watchlist_rows(db)
    has_any = any(r["sentiment"] or r["signal"] for r in rows)
    if not has_any:
        st.info("No analysis yet. Run `python main.py --once` to populate the watchlist.")
        return

    # Summary table
    table = []
    traded = below = 0
    for r in rows:
        s, sig = r["sentiment"], r["signal"]
        conf = sig.get("confidence")
        executed = bool(sig.get("executed"))
        if executed:
            traded += 1
        elif conf is not None and conf < settings.min_confidence:
            below += 1
        status = "✅ Traded" if executed else ("⏸ Skipped" if sig else "—")
        table.append({
            "Ticker": r["ticker"],
            "Bias": s.get("bias", "—"),
            "Unified": s.get("unified_score"),
            "AI Direction": sig.get("direction", "—"),
            "Strategy": sig.get("strategy", "—"),
            "Confidence": conf if conf is not None else 0,
            "Status": status,
            "Headlines": s.get("headline_count"),
        })
    df = pd.DataFrame(table)

    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Unified": st.column_config.NumberColumn("Unified", format="%.2f",
                                                     help="-1 bearish .. +1 bullish"),
            "Confidence": st.column_config.ProgressColumn(
                "AI Confidence", min_value=0, max_value=100, format="%d%%"),
            "Headlines": st.column_config.NumberColumn("Headlines", format="%d"),
        },
    )

    banner = (
        f"**{traded}** traded · **{below}** below the {settings.min_confidence}% "
        f"confidence threshold · {len(rows)} tickers analysed."
    )
    if traded == 0 and below > 0:
        st.warning(banner + "  The AI is standing aside — lower `MIN_CONFIDENCE` "
                   "in `.env` to be more aggressive.")
    else:
        st.success(banner)

    # Sentiment component heatmap
    heat = []
    for r in rows:
        s = r["sentiment"]
        if not s:
            continue
        row = {"Ticker": r["ticker"]}
        for key, label in _COMPONENTS.items():
            row[label] = s.get(key)
        row["Unified"] = s.get("unified_score")
        heat.append(row)
    if heat:
        hdf = pd.DataFrame(heat).set_index("Ticker")
        fig = px.imshow(
            hdf, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
            text_auto=".2f", aspect="auto",
            title="Sentiment breakdown (-1 bearish → +1 bullish)",
        )
        fig.update_layout(height=60 + 40 * len(hdf), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Per-ticker detail
    st.markdown("##### 🔍 Per-ticker detail")
    for r in rows:
        s, sig, lv = r["sentiment"], r["signal"], r["levels"]
        if not (s or sig):
            continue
        bias = s.get("bias", "—")
        conf = sig.get("confidence")
        header = (f"{r['ticker']}  —  {bias}"
                  + (f"  ·  AI: {sig.get('direction','?')} "
                     f"{sig.get('strategy','')} @ {conf:.0f}%" if sig else ""))
        with st.expander(header):
            left, right = st.columns([1, 1])
            with left:
                st.markdown(_bias_badge(bias), unsafe_allow_html=True)
                if s:
                    comp = pd.DataFrame(
                        {"component": list(_COMPONENTS.values()),
                         "score": [s.get(k) or 0 for k in _COMPONENTS]}
                    )
                    bar = px.bar(comp, x="score", y="component", orientation="h",
                                 range_x=[-1, 1], color="score",
                                 color_continuous_scale="RdYlGn",
                                 range_color=[-1, 1])
                    bar.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                                      showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(bar, use_container_width=True)
            with right:
                if sig:
                    st.metric("AI confidence", f"{conf:.0f}%" if conf is not None else "—")
                    st.write(f"**Direction:** {sig.get('direction','—')}")
                    st.write(f"**Strategy:** {sig.get('strategy','—')}")
                    levels_txt = []
                    for label, key in (("Entry", "entry_level"), ("Target", "exit_level"),
                                       ("Stop", "stop_level")):
                        if lv.get(key) is not None:
                            levels_txt.append(f"{label} {lv[key]}")
                    if levels_txt:
                        st.write("**Levels:** " + " · ".join(levels_txt))
                    if lv.get("target_contract"):
                        st.caption(f"Contract: {lv['target_contract']}")
                    st.write(f"**Reasoning:** {sig.get('reasoning','—')}")
                    st.caption(f"{'✅ Executed' if sig.get('executed') else '⏸ Not executed'}"
                               f" · {sig.get('created_at','')}")
                else:
                    st.info("No AI decision recorded yet for this ticker.")


def render_positions(db: Database) -> None:
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
        rows.append({
            "Ticker": t.ticker, "Strategy": t.strategy, "Dir": t.direction.value,
            "Contracts": t.contracts, "Entry $": round(t.entry_price, 2),
            "Current $": round(current, 2),
            "Stop $": round(t.stop_price, 2) if t.stop_price else None,
            "Expiry": t.expiration, "Unreal P&L $": round(unreal, 2),
            "Unreal %": round(unreal_pct, 1), "Conf %": round(t.confidence, 0),
        })
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={
            "Unreal P&L $": st.column_config.NumberColumn(format="$%.2f"),
            "Unreal %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


def render_equity(db: Database) -> None:
    st.subheader("📈 Equity Curve")
    curve = db.get_equity_curve()
    if not curve:
        st.info("No equity history yet.")
        return
    df = pd.DataFrame(curve)
    df["created_at"] = pd.to_datetime(df["created_at"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["created_at"], y=df["equity"],
                             mode="lines+markers", name="Equity",
                             line=dict(color="#3ddc84")))
    fig.add_hline(y=settings.starting_capital, line_dash="dash",
                  annotation_text="Start")
    fig.update_layout(height=320, yaxis_title="Equity ($)", xaxis_title=None,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_strategy_stats(db: Database) -> None:
    st.subheader("🏆 Performance by Strategy")
    stats = db.strategy_stats()
    if not stats:
        st.info("No closed trades yet.")
        return
    df = pd.DataFrame(stats)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="strategy", y="win_rate", title="Win rate %",
                     color="win_rate", color_continuous_scale="Blues")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(df, x="strategy", y="total_pnl", title="Total P&L $",
                      color="total_pnl", color_continuous_scale="RdYlGn")
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)


def render_trade_log(db: Database) -> None:
    st.subheader("📒 Trade Log")
    trades = db.get_trades(limit=50)
    if not trades:
        st.info("No trades recorded yet.")
        return
    rows = []
    for t in trades:
        rows.append({
            "Opened": t.opened_at.strftime("%Y-%m-%d %H:%M") if t.opened_at else "",
            "Ticker": t.ticker, "Strategy": t.strategy, "Dir": t.direction.value,
            "Status": t.status.value, "Entry $": round(t.entry_price, 2),
            "Exit $": round(t.exit_price, 2) if t.exit_price is not None else None,
            "P&L $": round(t.pnl, 2) if t.pnl is not None else None,
            "Conf %": round(t.confidence, 0), "Close": t.close_reason or "",
            "Reasoning": t.reasoning,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    db = get_db()

    with st.sidebar:
        st.header("⚙️ Controls")
        st.write(f"**Watchlist:** {', '.join(settings.watchlist)}")
        st.write(f"**Capital:** ${settings.starting_capital:,.0f}")
        st.write(f"**Min confidence:** {settings.min_confidence}%")
        st.write(f"**Max positions:** {settings.max_open_positions}")
        st.write(f"**Stop / target:** -{settings.stop_loss_pct:.0f}% / +60%")
        st.divider()
        auto = st.number_input("Auto-refresh (seconds, 0 = off)",
                               min_value=0, max_value=600, value=0, step=15)
        if st.button("🔄 Refresh now", use_container_width=True):
            st.rerun()
        st.caption(f"Loaded {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

    render_header(db)
    st.divider()
    render_watchlist(db)
    st.divider()
    render_positions(db)
    c1, c2 = st.columns([3, 2])
    with c1:
        render_equity(db)
    with c2:
        render_strategy_stats(db)
    render_trade_log(db)

    if auto and auto > 0:
        time.sleep(int(auto))
        st.rerun()


# Streamlit runs this file as __main__; the guard keeps imports side-effect free.
if __name__ == "__main__":
    main()
