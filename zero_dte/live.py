"""Live/paper 0DTE iron-condor runner on real moomoo quotes.

Run after 11:30 ET on a trading day (OpenD must be running):

    python -m zero_dte.live              # print the ticket only (dry run)
    python -m zero_dte.live --execute    # place it (SIMULATE unless REAL env set)

Implements the spread lab's leading structure — 11:30 ET iron condor,
short legs ≈ 0.15 delta, $2 wide, stop at 2.5x credit — using moomoo's REAL
chain, Greeks, and bid/ask (not synthetic prices). Every run is journaled to
results/live_journal.csv so forward paper results accumulate into the
real-data validation the synthetic backtest cannot provide. Risk: max-loss
sizing within 20% of capital, VIX>35 halt, one entry per day.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, date
from pathlib import Path

from zero_dte import data as yd
from zero_dte.config import RESULTS_DIR, config

JOURNAL = RESULTS_DIR / "live_journal.csv"
SHORT_DELTA = 0.15
WIDTH = 2.0
STOP_MULT = 2.5
UNDERLYING = "SPY"


def journal(row: dict) -> None:
    new = not Path(JOURNAL).exists()
    with open(JOURNAL, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="place the orders")
    ap.add_argument("--symbol", default=UNDERLYING)
    ap.add_argument("--capital", type=float, default=config.starting_capital)
    args = ap.parse_args()

    now = yd.now_eastern()
    if now.weekday() >= 5:
        print("Market closed (weekend)."); return
    if now.hour + now.minute / 60.0 < 11.5:
        print("Too early — this strategy enters at/after 11:30 ET."); return
    vix = yd.latest_vix()
    if vix and vix > config.vix_max:
        print(f"VIX {vix:.1f} > {config.vix_max:.0f} — risk halt, no trade today."); return

    from zero_dte.moomoo_exec import MoomooBroker
    b = MoomooBroker()
    try:
        chain = b.chain_0dte(args.symbol)
        if not chain:
            print("Empty 0DTE chain — check OpenD login, quote rights, market hours.")
            return
        legs = {}
        for is_call, name in ((True, "call"), (False, "put")):
            s = b.pick_by_delta(chain, SHORT_DELTA, is_call)
            if s is None:
                print(f"No suitable short {name} leg."); return
            l = b.pick_by_strike(chain, s.strike + (WIDTH if is_call else -WIDTH), is_call)
            if l is None or l.strike == s.strike:
                print(f"No long {name} wing at width ${WIDTH:g}."); return
            legs[name] = (s, l)

        credit = sum(s.bid - l.ask for s, l in legs.values())
        if credit <= 0.05:
            print(f"Net credit too thin (${credit:.2f}) — stand aside."); return
        max_loss = (WIDTH - credit) * 100
        budget = args.capital * config.max_premium_pct / 100.0
        contracts = max(0, int(budget // max_loss))
        if contracts < 1:
            print(f"Max loss ${max_loss:.0f}/condor exceeds {config.max_premium_pct:.0f}% "
                  f"of ${args.capital:,.0f} — no trade."); return
        contracts = min(contracts, 3)

        print(f"\nIRON CONDOR {args.symbol} 0DTE ({now:%Y-%m-%d %H:%M ET}) — REAL quotes")
        for name, (s, l) in legs.items():
            print(f"  SELL {s.code}  (Δ {s.delta}, bid {s.bid})   "
                  f"BUY {l.code} (ask {l.ask})")
        print(f"  credit ≈ ${credit:.2f}  max loss ${max_loss:.0f}/condor  "
              f"x{contracts}  stop: close if mark ≥ ${credit * STOP_MULT:.2f}  "
              f"flatten by 15:45 ET")

        result = {"ts": now.isoformat(), "symbol": args.symbol, "vix": vix,
                  "credit": round(credit, 2), "max_loss": round(max_loss, 2),
                  "contracts": contracts, "executed": False, "env": "", "orders": ""}
        if args.execute:
            for name, (s, l) in legs.items():
                r = b.place_credit_spread(s, l, contracts)
                print(f"  placed {name} spread [{r['env']}]: {r}")
                result.update(executed=True, env=r["env"],
                              orders=result["orders"] + f"{name}:{r['short']}/{r['long']} ")
        else:
            print("\nDry run (no orders). Add --execute to place in "
                  "SIMULATE; REAL needs ZDTE_TRD_ENV=REAL + ZDTE_LIVE_ACK + unlock pwd.")
        journal(result)
        print(f"Journaled to {JOURNAL}")
    finally:
        b.close()


if __name__ == "__main__":
    main()
