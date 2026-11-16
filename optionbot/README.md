# optionbot

Intraday long-options bot for moomoo (via OpenD), with Claude reviewing every
candidate trade through the Claude Code CLI on your Mac.

## What it does each weekday (US/Eastern)

| Time | Action |
|---|---|
| 09:00 | Connects to OpenD, reads the account, asks Claude for a short morning brief |
| 09:45–15:00 | Every 5 min: scores each symbol on 5 momentum checks, picks a liquid contract, sends the survivors to Claude, buys the approved ones |
| all day | Every 20 s: checks open positions against stop (-30%), target (+50%), breakeven stop and the daily loss limit |
| 15:45 | Sells everything. Nothing is held overnight |
| 16:00 | Writes `data/<mode>/reports/<date>.md` |

All limits live in `config.yaml`. The AI can only veto trades; it cannot change size, stops or limits.

## Setup (once)

1. Open **moomoo OpenD**, log in, and leave it running on port 11111.
   For live trading, click **Unlock Trade** in the OpenD window each time it starts.
2. Install the Claude Code CLI and log in with your Claude subscription:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

```bash
claude
```

3. Install the bot:

```bash
cd optionbot && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

4. Test every connection (places no orders):

```bash
.venv/bin/python -m bot check
```

5. Rehearse during market hours with no orders sent:

```bash
.venv/bin/python -m bot run --mode dry_run
```

6. Start it for real as a background service (starts at login, restarts on crash, keeps the Mac awake):

```bash
bash scripts/install_mac.sh
```

## Day to day

```bash
tail -f data/service.log
```

```bash
.venv/bin/python -m bot status
```

Emergency stop, sells everything and halts for the day:

```bash
touch data/live/STOP
```

Remove that file to allow trading again the next day. To remove the service:

```bash
bash scripts/install_mac.sh uninstall
```

## Things to know

- The Mac must be on, logged in, and online, with OpenD running and unlocked. Closing the lid stops the bot.
- If the bot or Mac dies with positions open, nothing is watching them. Check the moomoo app.
- Stops are checked every 20 seconds and sent as limit orders at the bid; a fast move can exit worse than -30%.
- `capital` in `config.yaml` caps what the bot sizes from, even if the account holds more.
- With `reuse_proceeds_same_day: false` the bot spends each dollar once per day, which is what a cash account requires.
- `holidays` and `early_closes` in `config.yaml` are listed through 2027 and need extending after that.

## Tests

```bash
.venv/bin/pip install pytest && .venv/bin/python -m pytest tests
```
