# ⚡ 0DTE Options Signal Tool (moomoo copy-trade)

Long **calls and puts only** — no spreads, no selling — sized for a small
**$1,000–$2,000** account, traded manually on **moomoo** via copy-paste order
tickets from the dashboard.

> ## ⚠️ Read the backtest verdict first
>
> This strategy was backtested over **2025 + 2026 YTD** on real hourly
> SPY/QQQ/AAPL/TSLA/NVDA/AMZN data (synthetic Black-Scholes 0DTE premiums
> seeded from VIX, slippage + commissions included). **It loses money in every
> tested configuration:**
>
> | Variant | Trades | Win % | Return | Profit factor |
> |---|---|---|---|---|
> | Baseline (2/3 votes, −30% stop, +60% target) | 75 | 28% | **−73%** | 0.55 |
> | Wide stop −50% / target +100% | 67 | 31% | −75% | 0.65 |
> | No stop, target +100%, time exit | 52 | 44% | −81% | 0.68 |
> | Strict 3/3 confluence | 1 | — | −5% | — |
>
> The lesson is the classic one: **buying ATM 0DTE options on simple momentum
> loses to theta decay and friction.** Use this tool for paper practice and
> signal study — do **not** trade it live with real money as-is.

## 🔬 Walk-forward strategy research (the "find a profitable strategy" hunt)

`python -m zero_dte.research` runs a systematic search designed to resist
self-deception: **4 entry archetypes** (opening-range breakout, 14:30 afternoon
momentum, 13:30 fade, gap-and-go) × **4 exit schemes** × **VIX/trend filters**
= 96 configs, ranked on TRAIN (→ Jun 2025) and judged only on TEST
(Jul 2025 → Jun 2026), plus a **walk-forward ML gate** (numpy logistic
regression over gap, opening-range width, momentum, VIX level/change —
retrained monthly, never sees its own evaluation month).

**Result: no configuration survives out-of-sample.** Best in-sample PF 0.52;
best OOS PF 0.43. The ML gate's learned policy is to take **zero trades** — no
market-condition combination exceeded 55% win probability. Gated account:
**0.0%** vs **−74.2%** trading the best base config. *The AI's optimal action
in this strategy class is standing aside* — that is the capital-preserving
edge it found, and it's wired into the live dashboard.

### Why, structurally

An ATM 0DTE option held 14:30 → 15:45 loses ~60% of its value to theta alone
if the underlying doesn't move (premium ∝ √time). Add a ~4% spread + commissions
and the underlying must move further than hourly-resolution signals can predict.

### What a real edge would require (in honesty order)

1. **Different trade structure** — defined-risk premium *selling* (credit
   spreads, iron condors) is where most documented 0DTE edges live; you
   excluded spreads, so this tool doesn't do them.
2. **Minute-level entries** with tight invalidation (requires paid intraday
   options data) instead of hourly bars.
3. **Convexity selection** (cheap OTM on confirmed trend days) — needs real
   historical options quotes to validate, not Black-Scholes synthesis.

## 🧠 The AI layer

* **ML gate** (`research.py`) — backtestable, walk-forward; currently gates
  everything out (capital preservation).
* **Claude live advisor** (`ai_advisor.py`) — at runtime, each rule-engine
  ticket is sent to Claude (uses your `ANTHROPIC_API_KEY`) with the research
  verdict embedded in its prompt; it returns go/no-go + max-1-contract sizing
  + reasoning, shown on every dashboard ticket. Default answer is NO-GO; no
  key → conservative offline fallback (always NO-GO).

## What's in the box

| File | Purpose |
|---|---|
| `config.py` | Capital, risk knobs, watchlist, session times |
| `data.py` | Free Yahoo data: hourly bars (~730d), daily bars, VIX |
| `signals.py` | 10:30 ET decision: opening-range breakout + EMA trend + prior-day momentum, RSI chase-veto |
| `pricing.py` | Black-Scholes synthetic 0DTE premiums (clearly labelled estimates) |
| `risk.py` | Per-trade premium cap (20% of capital), daily loss halt (−15%), max 2 trades/day, VIX>35 halt |
| `backtest.py` | 2025–2026 backtest → `results/*.csv/json` |
| `dashboard.py` | Streamlit: live signals + moomoo order tickets + backtest viewer |
| `tests.py` | 19 offline unit tests |

**Watchlist:** SPY & QQQ have 0DTE every trading day; AAPL/TSLA/NVDA/AMZN only
expire Fridays, so they're only tradeable on Fridays (handled automatically).

## Run it

```bash
pip install -r trading_bot/requirements.txt   # same deps (pandas, streamlit, plotly, pytz)

python -m zero_dte.tests                      # unit tests (no network)
python -m zero_dte.backtest                   # rerun the 2025-2026 backtest
streamlit run zero_dte/dashboard.py           # dashboard
```

## The moomoo copy-trade workflow

1. Open the dashboard after **10:30 ET** on a trading day.
2. If a signal fires and passes risk checks, you get a ready-made ticket:

```
moomoo ORDER TICKET (Jun 09 2026)
  Symbol:    SPY 06/09 600 CALL (0DTE)
  Action:    BUY TO OPEN
  Quantity:  1 contract(s)
  Type:      LIMIT @ ~$2.45 (verify live mid first)
  Stop:      sell if premium <= $1.72 (-30%)
  Target:    sell at >= $3.92 (+60%)
  Time exit: flatten by 15:45 ET — never hold to expiry
```

3. Enter it in moomoo **paper trading** manually. The premium shown is a
   Black-Scholes estimate — always confirm against moomoo's live quote.
4. Manage the exit per the ticket (stop / target / 15:45 flat).

Direct moomoo API automation is possible later via moomoo OpenAPI (requires
running their local **OpenD** gateway + the `moomoo-api` Python package), but
manual copy-trade is the safest starting point for a small account.

## Honest limitations

* **Synthetic premiums.** No free historical 0DTE quotes exist; the backtest
  prices options with Black-Scholes from VIX. Real spreads on 0DTE are wider
  and IV is dynamic intraday — real results would likely be *worse*.
* **Hourly bars.** Stops/targets are evaluated against hourly extremes
  (stop checked before target — conservative). Minute-level data would change
  individual fills but not the strategy verdict.
* **It's a paper tool.** The backtest says the edge isn't there. Improving it
  means better entries (minute data, vol regime filters) or different trade
  structures — not bigger size.
