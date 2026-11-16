"""moomoo OpenD wrapper. The engine only talks to this interface, so tests can swap in a fake."""
import logging
import os
import time

log = logging.getLogger("broker")

DONE = {"FILLED_ALL", "CANCELLED_ALL", "CANCELLED_PART", "FAILED", "SUBMIT_FAILED", "DISABLED", "DELETED"}


class BrokerError(Exception):
    pass


class MoomooBroker:
    def __init__(self, cfg: dict):
        import moomoo as mm
        self.mm, self.c = mm, cfg
        self.env = mm.TrdEnv.REAL if cfg["mode"] == "live" else mm.TrdEnv.SIMULATE
        self.q = mm.OpenQuoteContext(host=cfg["opend_host"], port=cfg["opend_port"])
        self.t = mm.OpenSecTradeContext(filter_trdmarket=mm.TrdMarket.US, host=cfg["opend_host"],
                                        port=cfg["opend_port"], security_firm=cfg["security_firm"])
        self._subscribed = set()
        try:
            self.acc_id = self._pick_account()
        except Exception:
            self.close()  # the SDK's network threads keep the process alive unless both contexts are closed
            raise
        pwd = os.environ.get("MOOMOO_TRADE_PWD")
        if cfg["mode"] == "live" and pwd:  # otherwise unlock trading in the OpenD window
            ret, data = self.t.unlock_trade(password=pwd)
            if ret != mm.RET_OK:
                log.warning("unlock_trade failed (%s); unlock in the OpenD window instead", data)

    def close(self):
        self.q.close()
        self.t.close()

    def _ok(self, ret, data, what):
        if ret != self.mm.RET_OK:
            raise BrokerError(f"{what}: {data}")
        return data

    def _pick_account(self) -> int:
        df = self._ok(*self.t.get_acc_list(), "get_acc_list")
        df = df[df["trd_env"] == self.env]
        if self.c.get("account_id"):
            df = df[df["acc_id"] == int(self.c["account_id"])]
        if df.empty:
            raise BrokerError(f"no {self.env} account found in OpenD for security_firm={self.c['security_firm']} "
                              "(check security_firm / account_id in config.yaml)")
        return int(df["acc_id"].iloc[0])

    def accounts(self):
        return self._ok(*self.t.get_acc_list(), "get_acc_list")

    def account(self) -> dict:
        df = self._ok(*self.t.accinfo_query(trd_env=self.env, acc_id=self.acc_id, refresh_cache=True,
                                            currency=self.mm.Currency.USD), "accinfo_query")
        r = df.iloc[0]
        cash = float(r["cash"])
        # Multi-currency accounts (e.g. moomoo Canada) report "cash" as every currency converted to USD.
        # US options settle in USD, so only the USD balance is spendable.
        try:
            cash = min(cash, float(r["us_cash"]))
        except (KeyError, TypeError, ValueError):
            pass  # single-currency account: field missing or "N/A"
        return {"assets": float(r["total_assets"]), "cash": cash, "power": float(r["power"])}

    # ---- market data ----
    def bars(self, symbol: str, n: int = 120):
        code = "US." + symbol
        if code not in self._subscribed:
            self._ok(*self.q.subscribe([code], [self.mm.SubType.K_5M], subscribe_push=False), "subscribe " + code)
            self._subscribed.add(code)
            time.sleep(0.3)
        return self._ok(*self.q.get_cur_kline(code, n, ktype=self.mm.KLType.K_5M), "kline " + code)

    def snapshot(self, codes: list[str]) -> dict:
        out = {}
        for i in range(0, len(codes), 300):
            df = self._ok(*self.q.get_market_snapshot(codes[i:i + 300]), "snapshot")
            for _, r in df.iterrows():
                out[r["code"]] = r
        return out

    def underlying(self, symbols: list[str]) -> dict:
        snap = self.snapshot(["US." + s for s in symbols])
        return {c[3:]: {"last": float(r["last_price"]), "prev_close": float(r["prev_close_price"]),
                        "open": float(r["open_price"])} for c, r in snap.items()}

    def quote(self, code: str) -> dict:
        r = self.snapshot([code])[code]
        return {"bid": float(r["bid_price"]), "ask": float(r["ask_price"]), "last": float(r["last_price"])}

    def option_candidates(self, symbol: str, direction: str, spot: float) -> list[dict]:
        """Contracts of the nearest expiry inside the DTE window, with quotes and greeks."""
        mm, c, code = self.mm, self.c, "US." + symbol
        exp = self._ok(*self.q.get_option_expiration_date(code), "expirations " + symbol)
        exp = exp[(exp["option_expiry_date_distance"] >= c["min_dte"]) &
                  (exp["option_expiry_date_distance"] <= c["max_dte"])]
        if exp.empty:
            return []
        row = exp.sort_values("option_expiry_date_distance").iloc[0]
        date, dte = row["strike_time"], int(row["option_expiry_date_distance"])
        otype = mm.OptionType.CALL if direction == "CALL" else mm.OptionType.PUT
        time.sleep(3.1)  # get_option_chain is limited to 10 calls / 30 s
        chain = self._ok(*self.q.get_option_chain(code, start=date, end=date, option_type=otype), "chain " + symbol)
        chain = chain[(chain["strike_price"] > spot * 0.93) & (chain["strike_price"] < spot * 1.07)]
        if chain.empty:
            return []
        snap = self.snapshot(list(chain["code"]))
        out = []
        for oc, r in snap.items():
            out.append({"code": oc, "symbol": symbol, "direction": direction, "expiry": date, "dte": dte,
                        "strike": float(r["option_strike_price"]), "bid": float(r["bid_price"]),
                        "ask": float(r["ask_price"]), "delta": float(r["option_delta"]),
                        "theta": float(r["option_theta"]), "iv": float(r["option_implied_volatility"]),
                        "oi": int(r["option_open_interest"]), "volume": int(r["volume"]),
                        "multiplier": int(r["option_contract_size"] or 100)})
        return out

    # ---- trading ----
    def place(self, code: str, qty: int, price: float, side: str) -> str:
        mm = self.mm
        df = self._ok(*self.t.place_order(price=price, qty=qty, code=code,
                                          trd_side=mm.TrdSide.BUY if side == "BUY" else mm.TrdSide.SELL,
                                          order_type=mm.OrderType.NORMAL, trd_env=self.env, acc_id=self.acc_id,
                                          time_in_force=mm.TimeInForce.DAY, remark="optionbot"), "place_order")
        return str(df["order_id"].iloc[0])

    def order(self, order_id: str) -> dict:
        df = self._ok(*self.t.order_list_query(order_id=order_id, trd_env=self.env, acc_id=self.acc_id,
                                               refresh_cache=True), "order_list_query")
        if df.empty:
            return {"status": "N/A", "filled": 0, "avg": 0.0}
        r = df.iloc[0]
        return {"status": str(r["order_status"]), "filled": int(r["dealt_qty"]), "avg": float(r["dealt_avg_price"])}

    def cancel(self, order_id: str):
        ret, data = self.t.modify_order(self.mm.ModifyOrderOp.CANCEL, order_id, 0, 0,
                                        trd_env=self.env, acc_id=self.acc_id)
        if ret != self.mm.RET_OK:
            log.warning("cancel %s: %s", order_id, data)

    def execute(self, code: str, qty: int, price: float, side: str, wait: float) -> dict:
        """Limit order; wait up to `wait` seconds, then cancel the remainder. Returns filled qty and avg price."""
        oid = self.place(code, qty, price, side)
        end = time.time() + wait
        o = {"status": "N/A", "filled": 0, "avg": 0.0}
        while time.time() < end:
            time.sleep(1.0)
            o = self.order(oid)
            if o["status"] in DONE:
                return o
        self.cancel(oid)
        for _ in range(10):  # wait for the cancel to settle so the fill count is final
            time.sleep(1.0)
            o = self.order(oid)
            if o["status"] in DONE:
                break
        return o

    def position_qty(self, code: str) -> int:
        df = self._ok(*self.t.position_list_query(code=code, trd_env=self.env, acc_id=self.acc_id,
                                                  refresh_cache=True), "position_list_query")
        return int(df["qty"].sum()) if not df.empty else 0
