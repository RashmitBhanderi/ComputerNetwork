"""Usage: python -m bot {check|run|status|flatten} [--config config.yaml] [--mode live|paper|dry_run]"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .engine import ET, Engine

ROOT = Path(__file__).resolve().parent.parent


def setup_logging(data: Path):
    (data / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fmt.converter = lambda *a: datetime.now(ET).timetuple()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(data / "logs" / f"{datetime.now(ET):%Y-%m}.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


def check(cfg):
    """Read-only test of every moving part. Places no orders."""
    from . import ai, signals
    from .broker import MoomooBroker
    print(f"mode: {cfg['mode']}")
    b = MoomooBroker(cfg)
    try:
        print("OpenD connection: OK")
        print(b.accounts()[["acc_id", "trd_env", "acc_type", "security_firm"]].to_string(index=False))
        print(f"using account {b.acc_id}: {b.account()}")
        und = b.underlying(["SPY"])["SPY"]
        print(f"SPY quote: {und}")
        bars = b.bars("SPY")
        print(f"SPY 5-min bars: {len(bars)} (last {bars['time_key'].iloc[-1]})")
        print("SPY signal:", signals.evaluate(bars, datetime.now(ET).strftime("%Y-%m-%d"), und["prev_close"]))
        opts = b.option_candidates("SPY", "CALL", und["last"])
        print(f"SPY call contracts in DTE window: {len(opts)}")
        if opts:
            print("  sample:", min(opts, key=lambda o: abs(abs(o["delta"]) - 0.45)))
    finally:
        b.close()  # otherwise the SDK's threads keep the process alive after an error
    if cfg.get("ai_enabled"):
        r = ai._call('Reply with exactly: {"ok": true}', cfg)
        print("Claude CLI:", "OK" if r and "ok" in r else "FAILED (run `claude` once and log in)")
    print("check finished; no orders were placed.")


def main():
    ap = argparse.ArgumentParser(prog="bot")
    ap.add_argument("cmd", choices=["check", "run", "status", "flatten"])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--mode", choices=["live", "paper", "dry_run"])
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.mode:
        cfg["mode"] = a.mode
    data = ROOT / "data" / cfg["mode"]
    setup_logging(data)

    if a.cmd == "check":
        return check(cfg)
    from .broker import MoomooBroker
    eng = Engine(cfg, data, lambda: MoomooBroker(cfg))
    if a.cmd == "status":
        print(json.dumps({"total_pnl": eng.s["total_pnl"], "day": {k: v for k, v in eng.day.items() if k != "closed"},
                          "positions": eng.pos}, indent=1))
    elif a.cmd == "flatten":
        eng.b = eng.broker_factory()
        eng.halt("manual flatten")
        eng.manage(force="manual flatten")
        print("remaining positions:", list(eng.pos))
        eng.b.close()
    else:
        logging.getLogger("main").info("starting in %s mode", cfg["mode"].upper())
        eng.run_forever()


if __name__ == "__main__":
    main()
