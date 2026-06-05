# 🤖 AI Options Paper Trading Bot

An end-to-end, **paper-trading** options bot that fuses real-time market data,
news sentiment, and Claude-powered analysis into disciplined, risk-managed
trade decisions — and **learns from its own track record** by injecting past
trade outcomes back into every new prompt.

> ⚠️ **Educational / research use only.** This bot trades *paper* money against
> Alpaca's paper API. It is not financial advice and is not wired to a live
> brokerage. Options trading carries substantial risk.

---

## ✨ What it does

1. **Data pipeline** — pulls prices (Alpaca → marketdata.app fallback), the
   options chain + Greeks (marketdata.app), the latest 20 headlines per ticker
   (NewsAPI + optional Benzinga), plus VIX, CNN Fear & Greed, and the put/call
   ratio.
2. **Sentiment engine** — scores headlines −1.0…+1.0 with Claude, then blends
   `news 30% + IV-rank 25% + put/call 20% + VIX 15% + options-flow 10%` into a
   single score and a bullish / bearish / neutral bias.
3. **AI analysis** — sends the full market context (price action, ATM chain,
   sentiment breakdown, **last 10 trades with outcomes**, and per-strategy win
   rates) to Claude, which returns a direction, strategy, confidence, reasoning,
   and entry/exit levels. Trades only fire at **confidence ≥ 65%**.
4. **Paper trading engine** — $100k virtual capital, position + Greeks tracking,
   auto-close on a −40% stop, +60% target, or expiry, with every trade logged.
5. **Learning loop** — closed-trade outcomes are written to SQLite and replayed
   into future prompts so the model refines its reasoning over time.
6. **Risk management** — ≤5% portfolio risk per trade, ≤3 open positions, an
   earnings blackout window, and a VIX > 35 halt (except volatility strategies).
7. **Scheduler** — APScheduler cron jobs (US/Eastern) drive the daily lifecycle.
8. **Dashboard** — a Streamlit app for live monitoring.

---

## 🏗️ Architecture

```
                          ┌────────────────────────────────────────────┐
                          │                main.py                      │
                          │   APScheduler (US/Eastern cron jobs)        │
                          │  08:30 premarket · 09:45 scan · 12:30 mid   │
                          │  15:45 eod · 17:00 report                   │
                          └───────────────┬────────────────────────────┘
                                          │ orchestrates
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
┌───────────────┐              ┌─────────────────────┐            ┌──────────────────┐
│  data/        │              │  sentiment/         │            │  ai/             │
│  market_data  │  prices ───▶ │  news_sentiment ◀── Claude       │  prompt_builder  │
│  options_data │  chain  ───▶ │  market_sentiment   │            │  memory ◀── DB   │
│  news_data    │  news   ───▶ │  sentiment_engine ──┼─ score ──▶ │  analyst  ──▶ Claude
└───────┬───────┘              └─────────────────────┘            └────────┬─────────┘
        │ Alpaca / marketdata.app / NewsAPI                                │ TradeSignal
        │                                                                  ▼
        │                                                       ┌────────────────────┐
        │                                                       │  trading/          │
        │                                                       │  strategy  (entry) │
        │                                                       │  risk_manager      │
        │                                                       │  paper_trader      │
        │                                                       └─────────┬──────────┘
        │                                                                 │ trades/outcomes
        ▼                                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          database/db.py  (SQLite: trades, signals,                 │
│                          sentiment, equity_curve)                                  │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                         │ reads
                                         ▼
                              ┌────────────────────────┐
                              │  dashboard/app.py       │
                              │  (Streamlit live view)  │
                              └────────────────────────┘
```

The **learning loop** is the dashed feedback path: `paper_trader` writes closed
trade outcomes to SQLite → `ai/memory` loads the last 10 → `prompt_builder`
injects them → `analyst` sends them to Claude on the next decision.

---

## 📂 Project layout

```
trading_bot/
├── main.py                  # Entry point + APScheduler
├── config.py                # Keys, settings, constants (from .env)
├── claude_client.py         # Shared Anthropic async client
├── models.py                # Pydantic models (signals, trades, sentiment)
├── logging_config.py        # Console + rotating file logging
├── data/
│   ├── http_client.py       # Async httpx + tenacity retry helper
│   ├── market_data.py       # Price/quote/VIX (Alpaca + marketdata.app)
│   ├── options_data.py      # Chain, Greeks, put/call ratio, IV rank
│   └── news_data.py         # Headlines (NewsAPI + Benzinga)
├── sentiment/
│   ├── news_sentiment.py    # Headline scoring via Claude
│   ├── market_sentiment.py  # VIX, Fear/Greed, options-flow normalisers
│   └── sentiment_engine.py  # Weighted unified score + bias
├── ai/
│   ├── memory.py            # Loads past trades for in-context learning
│   ├── prompt_builder.py    # Builds the analysis prompt
│   └── analyst.py           # Sends to Claude, parses TradeSignal
├── trading/
│   ├── strategy.py          # Entry rules + contract selection
│   ├── risk_manager.py      # Sizing, exposure, VIX/earnings guards
│   └── paper_trader.py      # Simulated execution, P&L, stops
├── database/
│   └── db.py                # SQLite handler
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── logs/                    # Rotating log files
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup

### 1. Install (Python 3.11+)

```bash
cd trading_bot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure keys

```bash
cp .env.example .env
# then edit .env and paste in your keys
```

### 3. Run the bot

```bash
python main.py            # start the scheduler (runs jobs at their cron times)
python main.py --once     # run one full cycle immediately (great for testing)
```

### 4. Launch the dashboard (separate terminal)

```bash
streamlit run dashboard/app.py
```

The bot runs end-to-end even with missing keys — any unavailable data source or
the Claude calls fall back to neutral values and the affected tickers are simply
skipped, so you can wire up keys incrementally.

---

## 🔑 Getting each API key

| Service | Used for | Where to get it |
|---|---|---|
| **Alpaca** (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) | Prices + paper account | [alpaca.markets](https://alpaca.markets) → sign up → **Paper Trading** dashboard → *Generate New Key*. Keep `ALPACA_BASE_URL=https://paper-api.alpaca.markets`. |
| **marketdata.app** (`MARKETDATA_API_KEY`) | Options chain, Greeks, IV, VIX | [dashboard.marketdata.app](https://dashboard.marketdata.app) → register → request a token (emailed to you). A data vendor (not a brokerage), so it's available **outside the US, including Canada**. Free access is a **30-day trial**. |
| **NewsAPI** (`NEWS_API_KEY`) | Headlines | [newsapi.org/register](https://newsapi.org/register) → free developer key. |
| **Benzinga** (`BENZINGA_API_KEY`) | *Optional* secondary news | [benzinga.com/apis](https://www.benzinga.com/apis). Leave blank to disable. |
| **Anthropic** (`ANTHROPIC_API_KEY`) | Claude sentiment + analysis | [console.anthropic.com](https://console.anthropic.com) → *API Keys*. |

---

## ⚙️ Configuration knobs

All optional; defaults live in `config.py` and can be overridden in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `WATCHLIST` | `SPY,QQQ,AAPL,TSLA,NVDA,AMZN` | Tickers to scan |
| `STARTING_CAPITAL` | `100000` | Virtual capital |
| `MIN_CONFIDENCE` | `65` | Min AI confidence to trade (%) |
| `MAX_OPEN_POSITIONS` | `3` | Concurrent position cap |
| `MAX_RISK_PER_TRADE_PCT` | `5` | Max portfolio risk per trade (%) |
| `STOP_LOSS_PCT` | `40` | Stop loss on premium (%) |
| `VIX_HALT_THRESHOLD` | `35` | Halt non-volatility trades above this VIX |

Options focus is **0–21 DTE**, **ATM ± 2 strikes** (see `min_dte`, `max_dte`,
`strikes_around_atm` in `config.py`).

---

## 🧠 Models used

* **Sentiment scoring** — `claude-sonnet-4-6` (high volume, cost-efficient).
* **Trade analysis** — `claude-opus-4-8` with adaptive thinking (the
  correctness-sensitive step).

> The original spec referenced `claude-sonnet-4-20250514`; that snapshot is
> deprecated, so the bot uses the current Sonnet/Opus IDs. Override the
> constants in `config.py` if you need a specific model.

---

## 📝 Notes & caveats

* **IV rank** is approximated from ATM implied volatility (the chain endpoint
  doesn't expose a trailing-52-week IV series); it is clearly labelled as an
  estimate.
* **Spreads** are priced from the long/short legs in the ATM window; the engine
  tracks the primary leg for Greeks and mark-to-market.
* The bot never connects to a live brokerage — `paper_trader.py` simulates all
  fills locally and persists them to SQLite.
