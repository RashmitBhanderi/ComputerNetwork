"""Translate an AI signal into a concrete, priceable options order.

The analyst returns a direction, a strategy name, and a confidence. This module
applies the entry rule (confidence >= MIN_CONFIDENCE and a directional/vol bias)
and then picks the contract(s) to trade from the focused options snapshot,
producing an `OrderPlan` with an entry premium.

For multi-leg strategies the net debit is approximated from the long and short
legs available in the ATM window; for single-leg strategies we use the ATM
contract's mid price. Greeks on the plan come from the primary (long) leg.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import settings
from logging_config import get_logger
from models import Direction, OptionContract, OptionsSnapshot, TradeSignal

log = get_logger(__name__)


@dataclass
class OrderPlan:
    ticker: str
    strategy: str
    direction: Direction
    option_symbol: str
    option_type: str
    strike: float
    expiration: str
    entry_premium: float  # per share (x100 per contract)
    delta: Optional[float]
    theta: Optional[float]
    iv: Optional[float]
    is_spread: bool


class StrategyEngine:
    def should_enter(self, signal: TradeSignal) -> tuple[bool, str]:
        """Entry gate based on AI confidence and a non-neutral stance."""
        if signal.confidence < settings.min_confidence:
            return False, (
                f"confidence {signal.confidence:.0f}% < {settings.min_confidence}%"
            )
        if signal.direction == Direction.NEUTRAL and not self._is_neutral_strategy(
            signal.strategy
        ):
            return False, "neutral direction with directional strategy"
        return True, "entry criteria met"

    @staticmethod
    def _is_neutral_strategy(strategy: str) -> bool:
        s = strategy.lower()
        return any(k in s for k in ("condor", "butterfly", "straddle", "strangle"))

    @staticmethod
    def _is_spread(strategy: str) -> bool:
        s = strategy.lower()
        return any(k in s for k in ("spread", "condor", "butterfly", "calendar"))

    def build_order(
        self, signal: TradeSignal, options: OptionsSnapshot
    ) -> Optional[OrderPlan]:
        """Select contracts and price the trade from the snapshot."""
        if not options.contracts or options.atm_strike is None:
            log.info("No chain to build order for %s", signal.ticker)
            return None

        # Long leg type follows the directional bias.
        long_type = "call" if signal.direction != Direction.BEARISH else "put"
        # Neutral / vol strategies default to calls for the primary leg.
        if signal.direction == Direction.NEUTRAL:
            long_type = "call"

        long_leg = self._pick_contract(options, options.atm_strike, long_type)
        if long_leg is None or long_leg.mid is None:
            log.info("No priceable ATM %s for %s", long_type, signal.ticker)
            return None

        is_spread = self._is_spread(signal.strategy)
        entry_premium = long_leg.mid

        if is_spread:
            short_leg = self._pick_short_leg(options, options.atm_strike, long_type)
            if short_leg is not None and short_leg.mid is not None:
                # Net debit for a vertical debit spread = long mid - short mid.
                net = round(long_leg.mid - short_leg.mid, 4)
                entry_premium = net if net > 0 else long_leg.mid

        if entry_premium <= 0:
            return None

        return OrderPlan(
            ticker=signal.ticker,
            strategy=signal.strategy,
            direction=signal.direction,
            option_symbol=long_leg.symbol,
            option_type=long_leg.option_type,
            strike=long_leg.strike,
            expiration=long_leg.expiration,
            entry_premium=round(entry_premium, 4),
            delta=long_leg.delta,
            theta=long_leg.theta,
            iv=long_leg.iv,
            is_spread=is_spread,
        )

    @staticmethod
    def _pick_contract(
        options: OptionsSnapshot, strike: float, option_type: str
    ) -> Optional[OptionContract]:
        matches = [
            c for c in options.contracts
            if c.option_type == option_type and c.strike == strike
        ]
        if matches:
            return matches[0]
        # Fall back to the nearest strike of that type.
        typed = [c for c in options.contracts if c.option_type == option_type]
        if not typed:
            return None
        return min(typed, key=lambda c: abs(c.strike - strike))

    def _pick_short_leg(
        self, options: OptionsSnapshot, atm_strike: float, long_type: str
    ) -> Optional[OptionContract]:
        """Pick the next strike out as the short leg of a vertical spread."""
        typed = sorted(
            (c for c in options.contracts if c.option_type == long_type),
            key=lambda c: c.strike,
        )
        strikes = [c.strike for c in typed]
        if atm_strike not in strikes:
            return None
        idx = strikes.index(atm_strike)
        # Bull call spread: short the higher strike. Bear put spread: short lower.
        target_idx = idx + 1 if long_type == "call" else idx - 1
        if 0 <= target_idx < len(typed):
            return typed[target_idx]
        return None
