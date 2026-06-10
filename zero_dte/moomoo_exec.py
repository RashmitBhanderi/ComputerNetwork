"""moomoo OpenAPI broker layer: real chains/Greeks/quotes + guarded execution.

Prerequisites (on YOUR machine — cannot run in a cloud sandbox):
  1. moomoo account with US options trading enabled.
  2. OpenD gateway running locally (download from moomoo OpenAPI site, log in).
  3. pip install moomoo-api
  4. US options LV1 quote rights on your moomoo account (for live quotes).

Safety model:
  * Default trade environment is SIMULATE (moomoo paper). REAL requires BOTH
    env vars:  ZDTE_TRD_ENV=REAL  and  ZDTE_LIVE_ACK=I_UNDERSTAND_THE_RISKS
    plus MOOMOO_UNLOCK_PWD (moomoo trade-unlock password).
  * Orders are LIMIT only, one spread (2 legs) at a time, hard cap 3 contracts.

NOTE: legs are placed as two separate orders (OpenAPI has no native US-option
combo ticket), so brief one-legged exposure exists between fills. The short
leg is only sent after the long (protective) leg fills.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

HOST = os.getenv("MOOMOO_HOST", "127.0.0.1")
PORT = int(os.getenv("MOOMOO_PORT", "11111"))
MAX_CONTRACTS = 3


def _env_is_real() -> bool:
    return (os.getenv("ZDTE_TRD_ENV", "SIMULATE").upper() == "REAL"
            and os.getenv("ZDTE_LIVE_ACK", "") == "I_UNDERSTAND_THE_RISKS")


@dataclass
class OptionQuote:
    code: str
    strike: float
    is_call: bool
    bid: float
    ask: float
    delta: Optional[float]
    iv: Optional[float]

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2, 2)


class MoomooBroker:
    """Thin wrapper over moomoo OpenAPI quote + trade contexts."""

    def __init__(self) -> None:
        try:
            import moomoo  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "moomoo-api not installed. Run: pip install moomoo-api "
                "(and start the OpenD gateway)."
            ) from exc
        self.mm = __import__("moomoo")
        self.quote = self.mm.OpenQuoteContext(host=HOST, port=PORT)
        self.trade = self.mm.OpenSecTradeContext(
            filter_trdmarket=self.mm.TrdMarket.US, host=HOST, port=PORT,
            security_firm=self.mm.SecurityFirm.FUTUINC,
        )
        self.trd_env = (self.mm.TrdEnv.REAL if _env_is_real()
                        else self.mm.TrdEnv.SIMULATE)
        if self.trd_env == self.mm.TrdEnv.REAL:
            pwd = os.getenv("MOOMOO_UNLOCK_PWD", "")
            ret, msg = self.trade.unlock_trade(pwd)
            if ret != self.mm.RET_OK:
                raise RuntimeError(f"REAL unlock failed: {msg}")

    def close(self) -> None:
        self.quote.close()
        self.trade.close()

    # ─── data ────────────────────────────────────────────────────────────────
    def chain_0dte(self, underlying: str) -> list[OptionQuote]:
        """Today's-expiry chain with live bid/ask + Greeks."""
        today = date.today().isoformat()
        ret, df = self.quote.get_option_chain(
            code=f"US.{underlying}", start=today, end=today)
        if ret != self.mm.RET_OK or df is None or df.empty:
            return []
        codes = list(df["code"])
        self.quote.subscribe(codes, [self.mm.SubType.QUOTE], subscribe_push=False)
        time.sleep(1.0)  # let snapshots populate
        ret2, snap = self.quote.get_market_snapshot(codes)
        if ret2 != self.mm.RET_OK:
            return []
        out: list[OptionQuote] = []
        for _, r in snap.iterrows():
            out.append(OptionQuote(
                code=r["code"],
                strike=float(r.get("option_strike_price") or 0),
                is_call=str(r.get("option_type", "")).upper().startswith("CALL"),
                bid=float(r.get("bid_price") or 0),
                ask=float(r.get("ask_price") or 0),
                delta=r.get("option_delta"),
                iv=r.get("option_implied_volatility"),
            ))
        return out

    def pick_by_delta(self, chain: list[OptionQuote], target: float,
                      is_call: bool) -> Optional[OptionQuote]:
        cands = [q for q in chain if q.is_call == is_call and q.delta is not None
                 and q.bid > 0 and abs(q.delta) <= target]
        if not cands:
            return None
        return max(cands, key=lambda q: abs(q.delta))   # closest under target

    def pick_by_strike(self, chain: list[OptionQuote], strike: float,
                       is_call: bool) -> Optional[OptionQuote]:
        cands = [q for q in chain if q.is_call == is_call]
        return min(cands, key=lambda q: abs(q.strike - strike)) if cands else None

    # ─── execution ───────────────────────────────────────────────────────────
    def _order(self, code: str, qty: int, price: float, side_buy: bool) -> Any:
        ret, df = self.trade.place_order(
            price=round(price, 2), qty=qty, code=code,
            trd_side=self.mm.TrdSide.BUY if side_buy else self.mm.TrdSide.SELL,
            order_type=self.mm.OrderType.NORMAL,   # limit
            trd_env=self.trd_env,
        )
        if ret != self.mm.RET_OK:
            raise RuntimeError(f"order failed ({code}): {df}")
        return df

    def place_credit_spread(self, short: OptionQuote, long: OptionQuote,
                            contracts: int) -> dict:
        """Buy the protective leg first, then sell the short leg."""
        n = min(contracts, MAX_CONTRACTS)
        env = "REAL" if self.trd_env == self.mm.TrdEnv.REAL else "SIMULATE"
        long_fill = self._order(long.code, n, long.ask, side_buy=True)
        short_fill = self._order(short.code, n, short.bid, side_buy=False)
        return {"env": env, "contracts": n,
                "long": str(long_fill.get("order_id", long_fill)),
                "short": str(short_fill.get("order_id", short_fill)),
                "est_credit": round(short.bid - long.ask, 2)}

    def positions(self):
        ret, df = self.trade.position_list_query(trd_env=self.trd_env)
        return df if ret == self.mm.RET_OK else None


def check_connection() -> None:
    """CLI:  python -m zero_dte.moomoo_exec  — verify OpenD + account."""
    b = MoomooBroker()
    env = "REAL ⚠️" if b.trd_env == b.mm.TrdEnv.REAL else "SIMULATE (paper)"
    print(f"Connected to OpenD at {HOST}:{PORT} | trade env: {env}")
    chain = b.chain_0dte("SPY")
    print(f"SPY 0DTE chain: {len(chain)} contracts with quotes"
          if chain else "SPY 0DTE chain empty — check quote rights / market hours")
    b.close()


if __name__ == "__main__":
    check_connection()
