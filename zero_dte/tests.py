"""Self-contained test suite (no network):  python -m zero_dte.tests"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from zero_dte.config import config
from zero_dte.pricing import atm_strike, bs_price, iv_from_vix, strike_increment
from zero_dte.risk import RiskManager
from zero_dte.signals import evaluate_session

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  {status}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def test_pricing() -> None:
    print("pricing:")
    atm_call = bs_price(500, 500, 5.5, 0.20, True)
    atm_put = bs_price(500, 500, 5.5, 0.20, False)
    check("ATM 0DTE premium in plausible range", 0.5 < atm_call < 10, f"{atm_call:.2f}")
    check("ATM call ≈ put (small rate drift)", abs(atm_call - atm_put) < 0.5)
    check("premium decays with time", bs_price(500, 500, 1.0, 0.20, True) < atm_call)
    check("premium rises with IV", bs_price(500, 500, 5.5, 0.40, True) > atm_call)
    itm = bs_price(510, 500, 0.05, 0.20, True)
    check("near-expiry ITM ≈ intrinsic", abs(itm - 10) < 0.6, f"{itm:.2f}")
    check("never below intrinsic", bs_price(510, 500, 5.5, 0.01, True) >= 10)
    check("iv_from_vix maps 20 → 0.20", abs(iv_from_vix(20) - 0.20 * config.iv_multiplier) < 1e-9)
    check("atm_strike rounds to increment", atm_strike(523.4, 1.0) == 523)
    check("strike increment SPY=1", strike_increment("SPY", 600) == 1.0)


def test_risk() -> None:
    print("risk:")
    rm = RiskManager(capital=1500)
    d = date(2026, 1, 5)
    ok = rm.check_entry(d, premium=2.0, vix=20)        # $200/contract, cap $300
    check("sizes 1 contract on $1500 @ $2.00", ok.approved and ok.contracts == 1)
    too_rich = rm.check_entry(d, premium=4.0, vix=20)  # $400 > 20% cap
    check("blocks premium above per-trade cap", not too_rich.approved)
    halted = rm.check_entry(d, premium=2.0, vix=40)
    check("blocks when VIX > max", not halted.approved)
    rm.record_entry(); rm.record_entry()
    third = rm.check_entry(d, premium=2.0, vix=20)
    check("blocks 3rd trade of the day", not third.approved)
    rm2 = RiskManager(capital=1500)
    rm2.start_session(d)
    rm2.record_pnl(-1500 * config.daily_loss_limit_pct / 100 - 1)
    stopped = rm2.check_entry(d, premium=1.0, vix=20)
    check("daily loss limit halts trading", not stopped.approved)
    rm3 = RiskManager(capital=1500)
    rm3.start_session(d); rm3.record_pnl(-500)
    rm3.start_session(date(2026, 1, 6))
    fresh = rm3.check_entry(date(2026, 1, 6), premium=1.0, vix=20)
    check("new session resets daily state", fresh.approved)


def _synthetic_bars(trend_up: bool) -> pd.DataFrame:
    """30 trading days of trending-but-oscillating hourly bars + final-day breakout.

    The wiggle matters: a monotone series pegs RSI at 100/0 and the engine's
    chase-veto would (correctly) suppress every signal.
    """
    import math
    import pytz
    tz = pytz.timezone("US/Eastern")
    rows, idx = [], []
    px = 500.0
    drift = 0.6 if trend_up else -0.6
    i = 0
    for d in pd.bdate_range("2026-04-01", periods=30):
        for h, m in [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)]:
            i += 1
            px += drift / 7
            wig = 0.45 * math.sin(i * 1.3)
            close = px + wig
            rows.append({"open": close - 0.05, "high": close + 0.4,
                         "low": close - 0.4, "close": close, "volume": 1e6})
            idx.append(tz.localize(pd.Timestamp(d.date()).replace(hour=h, minute=m)))
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    # Final session: 10:30 open breaks the opening range in the trend direction.
    last = df.index.date == df.index.date[-1]
    or_loc = df.index[last][0]
    dec_loc = df.index[last][1]
    if trend_up:
        df.loc[dec_loc, "open"] = float(df.loc[or_loc, "high"]) + 1.0
    else:
        df.loc[dec_loc, "open"] = float(df.loc[or_loc, "low"]) - 1.0
    return df


def test_signals() -> None:
    print("signals:")
    up = _synthetic_bars(trend_up=True)
    session = up.index.date[-1]
    sig = evaluate_session("SPY", up, session)
    check("uptrend + breakout fires a CALL", sig is not None and sig.side == "call",
          str(sig))
    down = _synthetic_bars(trend_up=False)
    sig_d = evaluate_session("SPY", down, down.index.date[-1])
    check("downtrend + breakdown fires a PUT", sig_d is not None and sig_d.side == "put",
          str(sig_d))
    check("deterministic (same input, same output)",
          evaluate_session("SPY", up, session) == sig)
    tiny = up.iloc[:5]
    check("insufficient history → no signal",
          evaluate_session("SPY", tiny, tiny.index.date[-1]) is None)


def main() -> None:
    test_pricing()
    test_risk()
    test_signals()
    if FAILURES:
        print(f"\n❌ {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("\n✅ all tests passed")


if __name__ == "__main__":
    main()
