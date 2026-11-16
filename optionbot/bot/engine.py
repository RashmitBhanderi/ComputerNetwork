"""Daily loop: prepare -> scan for entries -> manage exits -> flatten -> report."""
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from . import ai as ai_mod
from . import signals
from .broker import BrokerError
from .risk import Risk

ET = ZoneInfo("America/New_York")
log = logging.getLogger("engine")


class Engine:
    def __init__(self, cfg, data_dir, broker_factory, ai=ai_mod, now=None, sleep=time.sleep):
        self.c, self.dir, self.broker_factory, self.ai = cfg, Path(data_dir), broker_factory, ai
        self.now = now or (lambda: datetime.now(ET))
        self.sleep = sleep
        self.risk = Risk(cfg)
        self.dry = cfg["mode"] == "dry_run"
        self.b = None
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.s = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.s.setdefault("total_pnl", 0.0)
        self.s.setdefault("positions", {})
        self.s.setdefault("day", {})
        self.order_errors = 0

    # ---- helpers ----
    def save(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=1))
        tmp.replace(self.state_file)

    @property
    def day(self):
        return self.s["day"]

    @property
    def pos(self):
        return self.s["positions"]

    def at(self, hhmm: str, shift_early=True) -> datetime:
        n = self.now()
        h, m = map(int, hhmm.split(":"))
        t = n.replace(hour=h, minute=m, second=0, microsecond=0)
        if shift_early and n.strftime("%Y-%m-%d") in self.c.get("early_closes", []):
            t -= timedelta(hours=3)
        return t

    def is_trading_day(self) -> bool:
        n = self.now()
        return n.weekday() < 5 and n.strftime("%Y-%m-%d") not in self.c.get("holidays", [])

    def halt(self, why: str):
        if not self.day.get("halted"):
            log.warning("HALT for the day: %s", why)
            self.day["halted"] = why
            self.save()

    def _exec(self, code, qty, price, side, wait):
        if self.dry:
            return {"status": "FILLED_ALL", "filled": qty, "avg": price}
        return self.b.execute(code, qty, price, side, wait)

    # ---- day lifecycle ----
    def prepare(self):
        today = self.now().strftime("%Y-%m-%d")
        if self.day.get("date") == today:
            log.info("resuming %s: %d trades, realised %.2f", today, self.day["trades"], self.day["realized"])
            return
        acct = self.b.account()
        equity = self.risk.equity(self.s, None if self.dry else acct["assets"])
        self.s["day"] = {"date": today, "start_equity": equity,
                         "start_cash": equity if self.dry else min(equity, acct["cash"]),
                         "realized": 0.0, "spent": 0.0, "proceeds": 0.0, "trades": 0, "losers": [],
                         "halted": "", "ai_calls": 0, "brief": "", "rejected": {}, "closed": []}
        log.info("new day %s | mode=%s | bot equity %.2f | cash usable %.2f | account assets %.2f",
                 today, self.c["mode"], equity, self.day["start_cash"], acct["assets"])
        if self.pos:
            log.warning("carrying %d position(s) from a previous session; they will be managed/flattened", len(self.pos))
        if equity < 100:
            self.halt("bot equity below $100")
        if self.c.get("ai_enabled"):
            try:
                self.day["brief"] = self.ai.morning_brief(self.b.underlying(self.c["symbols"]), self.c)
                self.day["ai_calls"] += 1
                log.info("morning brief: %s", self.day["brief"] or "(none)")
            except Exception as e:
                log.error("morning brief failed: %s", e)
        self.save()

    def open_pnl(self, quotes: dict) -> float:
        total = 0.0
        for code, p in self.pos.items():
            q = quotes.get(code)
            if q and q["bid"] > 0:
                total += ((q["bid"] + q["ask"]) / 2 - p["entry"]) * p["qty"] * p["mult"]
        return total

    def _quotes(self) -> dict:
        return {code: self.b.quote(code) for code in self.pos}

    # ---- exits ----
    def manage(self, force: str | None = None):
        if not self.pos:
            return
        quotes = self._quotes()
        if not force and self.risk.loss_limit_hit(self.day, self.open_pnl(quotes)):
            self.halt("daily loss limit")
            force = "daily loss limit"
        for code in list(self.pos):
            p, q = self.pos[code], quotes[code]
            if q["bid"] <= 0:
                log.warning("%s has no bid; cannot exit yet", code)
                continue
            mid = (q["bid"] + q["ask"]) / 2
            reason = force or p.get("exiting") or self.risk.exit_reason(p, mid)
            if reason:
                self.close_position(code, q, reason)
        self.save()

    def close_position(self, code, q, reason):
        p = self.pos[code]
        p["exiting"] = reason
        p["exit_tries"] = p.get("exit_tries", 0) + 1
        price = q["bid"] if p["exit_tries"] == 1 else max(0.01, round(q["bid"] * 0.9, 2))
        try:
            o = self._exec(code, p["qty"], price, "SELL", 8)
        except BrokerError as e:
            log.error("SELL %s failed: %s", code, e)
            if not self.dry and self.b.position_qty(code) == 0:
                log.warning("%s no longer in the account; treating as closed at bid", code)
                o = {"filled": p["qty"], "avg": q["bid"]}
            else:
                return
        if o["filled"] <= 0:
            return
        fee = self.c["fee_per_contract"] * o["filled"] * 2
        pnl = (o["avg"] - p["entry"]) * o["filled"] * p["mult"] - fee
        self.day["realized"] += pnl
        self.day["proceeds"] += o["avg"] * o["filled"] * p["mult"]
        self.s["total_pnl"] += pnl
        row = {"date": self.day["date"], "opened": p["opened"], "closed": self.now().strftime("%H:%M:%S"),
               "symbol": p["symbol"], "code": code, "direction": p["direction"], "qty": o["filled"],
               "entry": p["entry"], "exit": round(o["avg"], 2), "pnl": round(pnl, 2), "reason": reason,
               "ai": p.get("ai", ""), "mode": self.c["mode"]}
        self.day["closed"].append(row)
        self._journal(row)
        log.info("CLOSED %s x%d @ %.2f (%s) pnl %.2f | day %.2f", code, o["filled"], o["avg"], reason, pnl,
                 self.day["realized"])
        if pnl < 0 and p["symbol"] not in self.day["losers"]:
            self.day["losers"].append(p["symbol"])
        p["qty"] -= o["filled"]
        if p["qty"] <= 0:
            del self.pos[code]

    def _journal(self, row):
        f = self.dir / "trades.csv"
        new = not f.exists()
        with f.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # ---- entries ----
    def pick_contract(self, symbol, direction, spot, max_price):
        c, best = self.c, None
        for o in self.b.option_candidates(symbol, direction, spot):
            if o["bid"] <= 0 or o["ask"] <= 0 or o["ask"] > max_price:
                continue
            mid = (o["bid"] + o["ask"]) / 2
            if (o["ask"] - o["bid"]) / mid * 100 > c["max_spread_pct"]:
                continue
            if not c["delta_min"] <= abs(o["delta"]) <= c["delta_max"]:
                continue
            if o["oi"] < c["min_open_interest"] or o["volume"] < c["min_volume"]:
                continue
            o["spread_pct"] = round((o["ask"] - o["bid"]) / mid * 100, 1)
            if best is None or abs(abs(o["delta"]) - c["delta_target"]) < abs(abs(best["delta"]) - c["delta_target"]):
                best = o
        return best

    def scan(self):
        c, now = self.c, self.now()
        if self.day.get("halted"):
            return
        quotes = self._quotes()
        open_pnl = self.open_pnl(quotes)
        ok, why = self.risk.can_enter(self.day, self.pos, "", open_pnl)
        if not ok:
            log.info("no entries: %s", why)
            if why == "daily loss limit":
                self.halt(why)
            return
        equity = self.risk.equity(self.s, None)
        max_price = self.risk.max_contract_price(self.day, equity)
        if max_price < 0.10:
            log.info("no entries: budget for today is used up")
            return
        today = now.strftime("%Y-%m-%d")
        und = self.b.underlying(c["symbols"])
        sigs = {}
        for sym in c["symbols"]:
            try:
                sg = signals.evaluate(self.b.bars(sym), today, und.get(sym, {}).get("prev_close"))
            except BrokerError as e:
                log.warning("%s bars: %s", sym, e)
                continue
            if sg:
                sigs[sym] = sg
        ranked = sorted((s for s in sigs if sigs[s]["direction"] and sigs[s]["score"] >= c["min_score"]),
                        key=lambda s: -sigs[s]["score"])
        slots = min(c["max_open_positions"] - len(self.pos), c["max_trades_per_day"] - self.day["trades"])
        cands = []
        for sym in ranked:
            if len(cands) >= slots + 1:
                break
            if not self.risk.can_enter(self.day, self.pos, sym, open_pnl)[0]:
                continue
            rej = self.day["rejected"].get(sym)
            if rej and now - datetime.fromisoformat(rej) < timedelta(minutes=c["ai_recheck_minutes"]):
                continue
            try:
                opt = self.pick_contract(sym, sigs[sym]["direction"], sigs[sym]["price"], max_price)
            except BrokerError as e:
                log.warning("%s chain: %s", sym, e)
                continue
            if opt:
                cands.append({"symbol": sym, "signal": sigs[sym], "contract": opt})
        log.info("scan: %d signals, %d tradable candidates %s", len(ranked), len(cands), [x["symbol"] for x in cands])
        if not cands:
            return
        verdicts = {x["symbol"]: {"take": True, "confidence": 1.0, "reason": "AI disabled"} for x in cands}
        if c.get("ai_enabled"):
            if self.day["ai_calls"] >= c["ai_max_calls_per_day"]:
                log.info("AI call budget used; no more entries today")
                return
            ctx = {"time_et": now.strftime("%H:%M"), "morning_brief": self.day["brief"],
                   "SPY": sigs.get("SPY"), "QQQ": sigs.get("QQQ"),
                   "today": {"trades": self.day["trades"], "realized_pnl": round(self.day["realized"], 2),
                             "open_positions": [p["symbol"] + " " + p["direction"] for p in self.pos.values()]},
                   "rules": {"stop_pct": -c["stop_loss_pct"], "target_pct": c["take_profit_pct"],
                             "flat_by": c["flatten_time"]}}
            self.day["ai_calls"] += 1
            verdicts = self.ai.review(cands, ctx, c)
        for x in cands:
            v = verdicts.get(x["symbol"], {"take": False, "confidence": 0, "reason": "no AI answer"})
            log.info("AI %s %s %s: %s (%.2f) %s", x["symbol"], x["signal"]["direction"], x["contract"]["code"],
                     "TAKE" if v["take"] else "skip", v["confidence"], v["reason"])
            if not v["take"]:
                self.day["rejected"][x["symbol"]] = now.isoformat()
                continue
            if self.risk.can_enter(self.day, self.pos, x["symbol"], open_pnl)[0]:
                self.enter(x, v, equity)
            if x["contract"]["code"] not in self.pos:  # approved but not filled: cool off, don't re-ask the AI at once
                self.day["rejected"][x["symbol"]] = now.isoformat()
        self.save()

    def enter(self, cand, verdict, equity):
        opt, code = cand["contract"], cand["contract"]["code"]
        q = self.b.quote(code)
        if q["bid"] <= 0 or q["ask"] <= 0:
            return
        mid = (q["bid"] + q["ask"]) / 2
        if (q["ask"] - q["bid"]) / mid * 100 > self.c["max_spread_pct"]:
            log.info("%s spread widened; skipped", code)
            return
        price = min(q["ask"], round(mid + 0.01, 2))
        qty = self.risk.size(self.day, equity, price, opt["multiplier"])
        if qty < 1:
            log.info("%s no longer fits the budget; skipped", code)
            return
        try:
            o = self._exec(code, qty, price, "BUY", 20)
            self.order_errors = 0
        except BrokerError as e:
            self.order_errors += 1
            log.error("BUY %s failed: %s", code, e)
            if self.order_errors >= 3:
                self.halt("3 order errors in a row (is trading unlocked in OpenD?)")
            return
        if o["filled"] <= 0:
            log.info("BUY %s x%d @ %.2f not filled; cancelled", code, qty, price)
            return
        stop, target = self.risk.levels(o["avg"])
        self.pos[code] = {"symbol": cand["symbol"], "direction": opt["direction"], "qty": o["filled"],
                          "mult": opt["multiplier"], "entry": o["avg"], "stop": stop, "target": target,
                          "opened": self.now().strftime("%H:%M:%S"), "ai": verdict["reason"]}
        self.day["trades"] += 1
        self.day["spent"] += o["avg"] * o["filled"] * opt["multiplier"]
        log.info("OPENED %s x%d @ %.2f stop %.2f target %.2f (trade %d/%d)", code, o["filled"], o["avg"], stop,
                 target, self.day["trades"], self.c["max_trades_per_day"])
        self.save()

    # ---- reporting ----
    def report(self):
        d = self.day
        if not d or d.get("reported"):
            return
        rows = d.get("closed", [])
        wins = sum(1 for r in rows if r["pnl"] > 0)
        lines = [f"# {d['date']} ({self.c['mode']})", "",
                 f"- Trades closed: {len(rows)} ({wins} winners)",
                 f"- Day P&L: {d['realized']:.2f}",
                 f"- Bot equity: {self.risk.equity(self.s, None):.2f} (started the day at {d['start_equity']:.2f})",
                 f"- AI calls: {d['ai_calls']}", f"- Halted: {d['halted'] or 'no'}",
                 f"- Morning brief: {d['brief'] or '-'}", "",
                 "| symbol | contract | qty | entry | exit | pnl | reason |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['symbol']} | {r['code']} | {r['qty']} | {r['entry']} | {r['exit']} | {r['pnl']} | {r['reason']} |"
                  for r in rows]
        if self.c.get("ai_enabled") and self.c.get("daily_review"):
            try:
                review = self.ai.daily_review({k: v for k, v in d.items() if k != "rejected"}, self._history(d["date"]), self.c)
                lines += ["", "## Claude's review", "", review or "(review unavailable)"]
            except Exception as e:
                log.error("daily review failed: %s", e)
        self._notify(f"{d['date']}: {len(rows)} trades, P&L {d['realized']:+.2f}, equity {self.risk.equity(self.s, None):.2f}")
        (self.dir / "reports").mkdir(exist_ok=True)
        (self.dir / "reports" / f"{d['date']}.md").write_text("\n".join(lines) + "\n")
        d["reported"] = True
        self.save()
        log.info("day done: %d trades, P&L %.2f", len(rows), d["realized"])

    def _history(self, today, n=40):
        f = self.dir / "trades.csv"
        if not f.exists():
            return []
        with f.open() as fh:
            rows = [r for r in csv.DictReader(fh) if r["date"] != today]
        return [{k: r[k] for k in ("date", "opened", "symbol", "direction", "entry", "exit", "pnl", "reason")} for r in rows[-n:]]

    def _notify(self, text):
        if sys.platform == "darwin":
            try:
                subprocess.run(["osascript", "-e", f'display notification "{text}" with title "optionbot"'], timeout=10)
            except Exception:
                pass

    # ---- loops ----
    def connect(self, until: datetime) -> bool:
        while self.now() < until:
            try:
                self.b = self.broker_factory()
                return True
            except Exception as e:
                log.error("cannot connect to OpenD (%s); retrying in 60s", e)
                self.sleep(60)
        return False

    def run_day(self):
        close = self.at("16:00")
        first, last, flat = self.at(self.c["first_entry"], False), self.at(self.c["last_entry"]), self.at(self.c["flatten_time"])
        if not self.connect(flat):
            log.error("never reached OpenD today")
            return
        try:
            self.prepare()
            next_scan = first
            while self.now() < close:
                n = self.now()
                try:
                    if (self.dir / "STOP").exists():
                        self.halt("STOP file present")
                        self.manage(force="manual stop")
                    elif n >= flat:
                        self.manage(force="end of day")
                        if not self.pos:
                            break
                    elif n >= self.at("09:30", False):
                        self.manage()
                        if first <= n < last and n >= next_scan:
                            step = self.c["scan_minutes"]
                            next_scan = n.replace(second=10, microsecond=0) + timedelta(minutes=step - n.minute % step)
                            self.scan()
                except BrokerError as e:
                    log.error("broker error: %s", e)
                except Exception:
                    log.exception("unexpected error in loop")
                self.sleep(self.c["manage_seconds"])
            if self.pos:
                log.error("POSITIONS STILL OPEN AT CLOSE: %s - check the moomoo app", list(self.pos))
            self.report()
        finally:
            try:
                self.b.close()
            except Exception:
                pass
            self.b = None

    def run_forever(self):
        while True:
            n = self.now()
            done = self.day.get("date") == n.strftime("%Y-%m-%d") and self.day.get("reported")
            if self.is_trading_day() and not done and self.at(self.c["prep_time"]) <= n < self.at("16:00"):
                self.run_day()
            self.sleep(30)
