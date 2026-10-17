"""Runs the paper sleeves once an hour after live_bot.py: fetches Kraken 1h data for the 10 coins ONCE, then hands it
to each sleeve. The fetch covers everything since the earliest go-live (+60 days warmup), because each sleeve rebuilds
its trades from that history every run. Each sleeve is isolated: one failing never stops the other."""
import json, time, traceback
import pandas as pd
from live_bot import fetch_hourly
import donchian_bot, dip_bot

SLEEVES = [donchian_bot, dip_bot]
MIN_DAYS, WARMUP_DAYS = 150, 60


def history_days():
    lives = []
    for s in SLEEVES:
        if s.STATE.exists():
            g = json.loads(s.STATE.read_text()).get("go_live")
            if g:
                lives.append(pd.Timestamp(g))
    if not lives:
        return MIN_DAYS
    return max(MIN_DAYS, (pd.Timestamp.now() - min(lives)).days + WARMUP_DAYS)


def fetch_all(days):
    h1, missing = {}, []
    for c, p in donchian_bot.PRODUCTS.items():
        try:
            h1[c] = fetch_hourly(p, days)
        except Exception as e:
            missing.append(f"{c} ({type(e).__name__})")
        time.sleep(0.3)
    return h1, missing


if __name__ == "__main__":
    h1, missing = fetch_all(history_days())
    if not h1:
        raise SystemExit("no Kraken data")
    failed = []
    for s in SLEEVES:
        try:
            s.run(h1, missing)
        except Exception:
            traceback.print_exc()
            failed.append(s.__name__)
    if failed:
        raise SystemExit(f"sleeve(s) failed: {', '.join(failed)}")
