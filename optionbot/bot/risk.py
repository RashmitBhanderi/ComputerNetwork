"""Hard risk rules. Everything here is deterministic; the AI never touches it."""


class Risk:
    def __init__(self, cfg: dict):
        self.c = cfg

    def equity(self, state: dict, account_assets: float | None) -> float:
        eq = self.c["capital"] + state.get("total_pnl", 0.0)
        if account_assets is not None:
            eq = min(eq, account_assets)
        return max(eq, 0.0)

    def day_pnl(self, day: dict, open_pnl: float = 0.0) -> float:
        return day.get("realized", 0.0) + open_pnl

    def loss_limit_hit(self, day: dict, open_pnl: float) -> bool:
        limit = day["start_equity"] * self.c["max_daily_loss_pct"] / 100
        return self.day_pnl(day, open_pnl) <= -limit

    def budget_left(self, day: dict) -> float:
        """Cash the bot may still spend today."""
        spent = day.get("spent", 0.0)
        if self.c.get("reuse_proceeds_same_day"):
            spent -= day.get("proceeds", 0.0)
        return day["start_cash"] - spent

    def can_enter(self, day: dict, positions: dict, symbol: str, open_pnl: float) -> tuple[bool, str]:
        if day.get("halted"):
            return False, "halted: " + day["halted"]
        if self.loss_limit_hit(day, open_pnl):
            return False, "daily loss limit"
        if day.get("trades", 0) >= self.c["max_trades_per_day"]:
            return False, "max trades reached"
        if len(positions) >= self.c["max_open_positions"]:
            return False, "max open positions"
        if any(p["symbol"] == symbol for p in positions.values()):
            return False, "already in " + symbol
        if symbol in day.get("losers", []):
            return False, symbol + " already lost today"
        return True, ""

    def size(self, day: dict, equity: float, ask: float, multiplier: int = 100) -> int:
        """Number of contracts, 0 if the contract doesn't fit."""
        cost = ask * multiplier
        if cost <= 0:
            return 0
        cap = min(equity * self.c["max_premium_pct"] / 100, self.budget_left(day))
        return int(cap // cost)

    def max_contract_price(self, day: dict, equity: float, multiplier: int = 100) -> float:
        return min(equity * self.c["max_premium_pct"] / 100, self.budget_left(day)) / multiplier

    def exit_reason(self, pos: dict, mid: float) -> str | None:
        """Also ratchets the stop to breakeven. Mutates pos."""
        entry = pos["entry"]
        chg = (mid / entry - 1) * 100
        pos["peak_pct"] = max(pos.get("peak_pct", 0.0), chg)
        if pos["peak_pct"] >= self.c["breakeven_trigger_pct"]:
            pos["stop"] = max(pos["stop"], entry)
        if mid >= pos["target"]:
            return "target"
        if mid <= pos["stop"]:
            return "breakeven stop" if pos["stop"] >= entry else "stop loss"
        return None

    def levels(self, entry: float) -> tuple[float, float]:
        return (round(entry * (1 - self.c["stop_loss_pct"] / 100), 2),
                round(entry * (1 + self.c["take_profit_pct"] / 100), 2))
