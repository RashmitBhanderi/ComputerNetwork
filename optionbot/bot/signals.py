"""Momentum signal from 5-minute bars. Pure functions, no broker access."""
import pandas as pd


def rsi(closes: pd.Series, n: int = 14) -> float:
    d = closes.diff().dropna()
    if len(d) < n:
        return 50.0
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    if dn == 0:
        return 100.0
    return float(100 - 100 / (1 + up / dn))


def evaluate(bars: pd.DataFrame, today: str, prev_close: float | None = None) -> dict | None:
    """bars: columns time_key, open, high, low, close, volume (oldest first).
    Returns a dict with direction 'CALL' / 'PUT' / None and the facts behind it."""
    if bars is None or len(bars) < 25:
        return None
    day = bars[bars["time_key"].astype(str).str.startswith(today)]
    if len(day) < 4:
        return None
    price = float(day["close"].iloc[-1])
    tp = (day["high"] + day["low"] + day["close"]) / 3
    vol = day["volume"].clip(lower=0)
    vwap = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else float(tp.mean())
    ema9 = float(bars["close"].ewm(span=9, adjust=False).mean().iloc[-1])
    ema20 = float(bars["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    or_high = float(day["high"].iloc[:3].max())
    or_low = float(day["low"].iloc[:3].min())
    r = rsi(bars["close"])
    mom = price - float(day["close"].iloc[-4])
    avg_vol = float(vol.iloc[:-1].mean()) if len(vol) > 1 else 0.0
    rel_vol = float(vol.iloc[-2] / avg_vol) if avg_vol > 0 and len(vol) > 2 else 1.0

    bull = [price > vwap, ema9 > ema20, price > or_high, mom > 0, 50 < r < 75]
    bear = [price < vwap, ema9 < ema20, price < or_low, mom < 0, 25 < r < 50]
    bs, rs = sum(bull), sum(bear)
    direction, score = (None, max(bs, rs))
    if bs > rs:
        direction, score = "CALL", bs
    elif rs > bs:
        direction, score = "PUT", rs
    return {
        "direction": direction, "score": int(score), "price": round(price, 2),
        "vwap": round(vwap, 2), "ema9": round(ema9, 2), "ema20": round(ema20, 2),
        "opening_range": [round(or_low, 2), round(or_high, 2)], "rsi": round(r, 1),
        "momentum_15m": round(mom, 2), "rel_volume": round(rel_vol, 2),
        "day_change_pct": round((price / prev_close - 1) * 100, 2) if prev_close else None,
        "day_range": [round(float(day["low"].min()), 2), round(float(day["high"].max()), 2)],
    }
