"""Streamlit dashboard: live 0DTE signals to copy-trade into moomoo + backtest.

Launch from the repo root:  streamlit run zero_dte/dashboard.py

Tab 1 — Live signals: evaluates the watchlist at/after 10:30 ET, shows the
side (call/put), ATM strike, estimated premium, and a ready-to-copy moomoo
order ticket sized by the risk manager.
Tab 2 — Backtest: renders results/backtest_*.{csv,json} produced by
``python -m zero_dte.backtest``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from zero_dte import data as yd  # noqa: E402
from zero_dte.config import DAILY_0DTE, FRIDAY_0DTE, RESULTS_DIR, config  # noqa: E402
from zero_dte.pricing import atm_strike, bs_price, iv_from_vix, strike_increment  # noqa: E402
from zero_dte.risk import RiskManager  # noqa: E402
from zero_dte.signals import evaluate_session  # noqa: E402

st.set_page_config(page_title="0DTE Signals", page_icon="⚡", layout="wide")
st.title("⚡ 0DTE Options Signals — moomoo copy-trade")
st.caption(
    "Long calls/puts only · SPY & QQQ daily, single names Fridays · "
    f"capital ${config.starting_capital:,.0f} · premiums are Black-Scholes "
    "estimates — confirm the live quote in moomoo before sending."
)


@st.cache_data(ttl=300)
def load_bars(symbol: str) -> pd.DataFrame:
    return yd.hourly_bars(symbol, "60d")


@st.cache_data(ttl=300)
def market_now() -> tuple[float | None, date]:
    return yd.latest_vix(), yd.now_eastern().date()


def order_ticket(symbol: str, side: str, strike: float, contracts: int,
                 est_premium: float, session: date) -> str:
    return (
        f"moomoo ORDER TICKET ({session:%b %d %Y})\n"
        f"  Symbol:    {symbol} {session:%m/%d} {strike:g} {side.upper()} (0DTE)\n"
        f"  Action:    BUY TO OPEN\n"
        f"  Quantity:  {contracts} contract(s)\n"
        f"  Type:      LIMIT @ ~${est_premium:.2f} (verify live mid first)\n"
        f"  Stop:      sell if premium <= ${est_premium * (1 - config.stop_loss_pct/100):.2f} (-{config.stop_loss_pct:.0f}%)\n"
        f"  Target:    sell at >= ${est_premium * (1 + config.take_profit_pct/100):.2f} (+{config.take_profit_pct:.0f}%)\n"
        f"  Time exit: flatten by 15:45 ET — never hold to expiry"
    )


tab_live, tab_bt = st.tabs(["🔴 Live signals", "📊 Backtest 2025-2026"])

with tab_live:
    vix, today = market_now()
    now = yd.now_eastern()
    after_decision = now.hour + now.minute / 60.0 >= config.entry_hour
    weekday = today.weekday() < 5

    c1, c2, c3 = st.columns(3)
    c1.metric("VIX", f"{vix:.2f}" if vix else "—",
              "⚠️ above halt" if (vix and vix > config.vix_max) else "ok",
              delta_color="inverse")
    c2.metric("Session", f"{today:%a %b %d}")
    c3.metric("Decision time (10:30 ET)", "reached ✅" if after_decision else "not yet ⏳")

    tradeable = list(DAILY_0DTE) + (FRIDAY_0DTE if today.weekday() == 4 else [])
    skipped = [s for s in FRIDAY_0DTE if s not in tradeable]
    if skipped:
        st.caption(f"{', '.join(skipped)} only have Friday expiries — back on Friday.")

    if not weekday:
        st.info("Market closed (weekend).")
    elif vix and vix > config.vix_max:
        st.error(f"VIX {vix:.1f} > {config.vix_max:.0f} — risk halt, no trades today.")
    elif not after_decision:
        st.info("Signals generate after the opening range completes at 10:30 ET.")
    else:
        rm = RiskManager()
        rm.start_session(today)
        fired = 0
        for symbol in tradeable:
            try:
                bars = load_bars(symbol)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"{symbol}: data fetch failed ({exc})")
                continue
            sig = evaluate_session(symbol, bars, today)
            if sig is None:
                st.write(f"**{symbol}** — no confluence, standing aside.")
                continue
            iv = iv_from_vix(vix or 20.0)
            strike = atm_strike(sig.entry_underlying,
                                strike_increment(symbol, sig.entry_underlying))
            spot = yd.latest_price(symbol) or sig.entry_underlying
            hours_left = max(config.close_hour - (now.hour + now.minute / 60.0), 0.25)
            prem = bs_price(spot, strike, hours_left, iv, sig.side == "call")
            decision = rm.check_entry(today, prem, vix)
            fired += 1
            icon = "🟢 CALL" if sig.side == "call" else "🔴 PUT"
            with st.expander(f"{icon} — {symbol} (score {sig.score}/3)", expanded=True):
                st.write(" · ".join(sig.reasons))
                if decision.approved:
                    st.code(order_ticket(symbol, sig.side, strike,
                                         decision.contracts, prem, today))
                    rm.record_entry()
                else:
                    st.warning(f"Signal valid but risk-blocked: {decision.reason}")
        if fired == 0:
            st.success("No signals across the watchlist — cash is a position.")

with tab_bt:
    summary_path = RESULTS_DIR / "backtest_summary.json"
    if not summary_path.exists():
        st.info("No backtest results yet. Run:  `python -m zero_dte.backtest`")
    else:
        summary = json.loads(summary_path.read_text())
        o = summary["overall"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Return", f"{summary['total_return_pct']:+.1f}%",
                  f"${summary['end_capital'] - summary['start_capital']:+,.0f}")
        c2.metric("Trades", o["trades"])
        c3.metric("Win rate", f"{o['win_rate']:.0f}%")
        c4.metric("Profit factor", o["profit_factor"])
        c5.metric("Max drawdown", f"{summary['max_drawdown_pct']:.1f}%")
        st.caption(summary["note"])

        eq_path = RESULTS_DIR / "backtest_equity.csv"
        if eq_path.exists():
            eq = pd.read_csv(eq_path)
            fig = go.Figure(go.Scatter(x=eq.session, y=eq.equity, mode="lines",
                                       line=dict(color="#3ddc84")))
            fig.add_hline(y=summary["start_capital"], line_dash="dash")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="Equity ($)")
            st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**By year**")
            st.dataframe(pd.DataFrame(summary["by_year"]).T,
                         use_container_width=True)
            st.markdown("**By exit reason**")
            st.dataframe(pd.DataFrame(summary["by_exit"]).T,
                         use_container_width=True)
        with cols[1]:
            st.markdown("**By symbol**")
            st.dataframe(pd.DataFrame(summary["by_symbol"]).T,
                         use_container_width=True)

        trades_path = RESULTS_DIR / "backtest_trades.csv"
        if trades_path.exists():
            st.markdown("**Trade log**")
            st.dataframe(pd.read_csv(trades_path).sort_values("session", ascending=False),
                         use_container_width=True, hide_index=True, height=400)
