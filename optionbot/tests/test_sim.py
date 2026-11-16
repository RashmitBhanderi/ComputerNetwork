"""Simulated trading days against a fake broker: python -m pytest tests"""
import random
from datetime import datetime, timedelta

import pandas as pd
import yaml
from pathlib import Path

from bot.engine import ET, Engine
from bot.broker import BrokerError

CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())


class Clock:
    def __init__(self, t): self.t = t
    def now(self): return self.t
    def sleep(self, s): self.t += timedelta(seconds=s)


class FakeBroker:
    """SPY trends by `drift` per 5-min bar; option mid follows the underlying with delta 0.45."""
    def __init__(self, clock, drift, reject_orders=False):
        self.clock, self.drift, self.reject = clock, drift, reject_orders
        self.orders, self.entry_spot = [], {}
    def close(self): pass
    def account(self): return {"assets": 1500.0, "cash": 1500.0, "power": 1500.0}
    def spot(self, sym):
        n = self.clock.now()
        mins = max(0, (n - n.replace(hour=9, minute=30, second=0)).total_seconds() / 60)
        return 600 + self.drift * mins / 5 if sym in ("SPY", "QQQ", "IWM") else 100.0
    def underlying(self, syms): return {s: {"last": self.spot(s), "prev_close": 600.0, "open": 600.0} for s in syms}
    def bars(self, sym, n=120):
        now = self.clock.now()
        rows, t = [], now.replace(hour=9, minute=35, second=0, microsecond=0)
        prev = t - timedelta(days=1)
        for i in range(60):
            rows.append({"time_key": (prev + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                         "open": 600, "high": 600.2, "low": 599.8, "close": 600, "volume": 1000})
        i = 0
        while t <= now:
            c = (600 + self.drift * (i + 1)) if sym in ("SPY", "QQQ", "IWM") else 100.0
            rows.append({"time_key": t.strftime("%Y-%m-%d %H:%M:%S"), "open": c - self.drift, "high": c + 0.05,
                         "low": c - self.drift - 0.05, "close": c, "volume": 1000})
            t += timedelta(minutes=5); i += 1
        return pd.DataFrame(rows)
    def option_candidates(self, sym, direction, spot):
        base = {"symbol": sym, "direction": direction, "expiry": "x", "dte": 3, "theta": -0.1, "iv": 20,
                "oi": 5000, "volume": 900, "multiplier": 100}
        return [dict(base, code=f"US.{sym}{direction[0]}1", strike=spot, bid=0.78, ask=0.82, delta=0.45),
                dict(base, code=f"US.{sym}{direction[0]}2", strike=spot, bid=3.0, ask=3.1, delta=0.5),   # too dear
                dict(base, code=f"US.{sym}{direction[0]}3", strike=spot, bid=0.5, ask=0.8, delta=0.4)]   # wide
    def quote(self, code):
        sym = code[3:6]
        sign = 1 if code[6] == "C" else -1
        mid = max(0.05, 0.80 + sign * 0.45 * (self.spot(sym) - self.entry_spot.get(code, self.spot(sym))))
        return {"bid": round(mid - 0.02, 2), "ask": round(mid + 0.02, 2), "last": mid}
    def execute(self, code, qty, price, side, wait):
        if self.reject: raise BrokerError("trade locked")
        self.orders.append((side, code, qty, price))
        if side == "BUY": self.entry_spot[code] = self.spot(code[3:6])
        return {"status": "FILLED_ALL", "filled": qty, "avg": price}
    def position_qty(self, code): return 1


class FakeAI:
    def __init__(self, take=True): self.take, self.calls = take, 0
    def morning_brief(self, m, c): return "mixed: test"
    def daily_review(self, day, hist, c): return "REVIEW-OK"
    def review(self, cands, ctx, c):
        self.calls += 1
        return {x["symbol"]: {"take": self.take, "confidence": 0.8, "reason": "t"} for x in cands}


def run(tmp_path, drift, ai=None, cfg=None, **kw):
    clock = Clock(datetime(2026, 11, 17, 9, 0, tzinfo=ET))
    b = FakeBroker(clock, drift, **kw)
    cfg = dict(CFG, mode="live", **(cfg or {}))
    e = Engine(cfg, tmp_path, lambda: b, ai=ai or FakeAI(), now=clock.now, sleep=clock.sleep)
    e.run_day()
    return e, b


def test_uptrend_buys_calls_and_takes_profit(tmp_path):
    e, b = run(tmp_path, 0.15)
    rows = e.day["closed"]
    assert rows and all(r["direction"] == "CALL" for r in rows)
    assert any(r["reason"] == "target" for r in rows)
    assert all(o[2] == 1 and o[3] <= 1.5 for o in b.orders if o[0] == "BUY")   # 1 contract, <= $100
    assert not e.pos and e.day["reported"]
    assert e.day["trades"] <= CFG["max_trades_per_day"]
    assert (tmp_path / "trades.csv").exists()
    assert "REVIEW-OK" in (tmp_path / "reports" / "2026-11-17.md").read_text()


def test_downtrend_buys_puts(tmp_path):
    e, _ = run(tmp_path, -0.15)
    assert e.day["closed"] and all(r["direction"] == "PUT" for r in e.day["closed"])


def test_flat_market_no_trades(tmp_path):
    e, b = run(tmp_path, 0.0)
    assert e.day["trades"] == 0 and not b.orders


def test_ai_veto_blocks_everything(tmp_path):
    ai = FakeAI(take=False)
    e, b = run(tmp_path, 0.15, ai=ai)
    assert not b.orders and 0 < ai.calls <= CFG["ai_max_calls_per_day"]


def test_reversal_stops_out_and_respects_loss_limit(tmp_path):
    clock = Clock(datetime(2026, 11, 17, 9, 0, tzinfo=ET))
    b = FakeBroker(clock, 0.15)
    real_spot = b.spot
    def spot(sym):  # rally until 10:30 then collapse
        n = clock.now()
        if n.hour * 60 + n.minute > 630:
            b.drift = -0.6
        return real_spot(sym) if n.hour * 60 + n.minute <= 630 else 600 + 0.15 * 12 - 0.6 * ((n.hour * 60 + n.minute) - 630) / 5
    b.spot = spot
    e = Engine(dict(CFG, mode="live", take_profit_pct=500), tmp_path, lambda: b, ai=FakeAI(), now=clock.now, sleep=clock.sleep)
    e.run_day()
    assert not e.pos
    limit = e.day["start_equity"] * CFG["max_daily_loss_pct"] / 100
    assert e.day["realized"] > -limit - 60          # one position's worth of slippage past the limit at most
    assert e.day["spent"] <= e.day["start_cash"] + 1e-6


def test_cash_budget_never_exceeded(tmp_path):
    e, _ = run(tmp_path, 0.15, cfg={"max_premium_pct": 40, "max_open_positions": 3})
    assert e.day["spent"] <= e.day["start_cash"] + 1e-6


def test_order_errors_halt(tmp_path):
    e, _ = run(tmp_path, 0.15, reject_orders=True)
    assert "order errors" in e.day["halted"] and e.day["trades"] == 0


def test_holiday_and_weekend(tmp_path):
    for d, ok in [((2026, 11, 26), False), ((2026, 11, 21), False), ((2026, 11, 17), True)]:
        clock = Clock(datetime(*d, 10, 0, tzinfo=ET))
        assert Engine(CFG, tmp_path, None, now=clock.now).is_trading_day() is ok
