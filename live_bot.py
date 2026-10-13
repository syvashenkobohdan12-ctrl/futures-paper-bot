"""PAPER trading bot: daily 72d momentum + 4h MACD (ADX>40 entry filter) on BTC/ETH/SOL.
$10k paper account, 2% risk per trade, 5% stop, 3R take-profit, max hold 7 days (re-entry if signal still holds), breakeven stop after +1.5R, total notional capped at 10x equity.
Run once per hour (a few minutes past the hour). Each run: fetch Kraken Futures public 1h perpetual candles and
hourly funding (no key, US-reachable), compute signals exactly like the backtest, then feed every newly closed hour
through risk.step(). Funding is charged hourly like Kraken does (relativeFundingRate x notional; longs pay when > 0).
No real orders: fills are simulated locally. State persists in paper_state.json; events go to paper_events.csv.
shortcut: hours missed while the PC was off are replayed at their historical prices; fine for paper, not for real money."""
import json, time, datetime as dt
from pathlib import Path
import pandas as pd, requests
import risk, dashboard

HERE = Path(__file__).parent
STATE, EVENTS, LOG = HERE / "paper_state.json", HERE / "paper_events.csv", HERE / "paper_bot.log"
EQUITY = HERE / "paper_equity.csv"
START_CASH = 100_000
KRAKEN = "https://futures.kraken.com"
PRODUCTS = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD", "SOL": "PF_SOLUSD"}   # Kraken linear USD perpetuals
HISTORY_DAYS = 220            # warmup: EMA/ADX/vol windows need ~3x their length to match the backtest
JOIN_OPEN_TRENDS = False      # False = like the backtest: only enter on a fresh signal flip
CFG = dict(risk=0.02, stop_pct=0.05, max_lev=10, max_hold=168, tp_r=3, reenter=("time",), be_r=1.5)   # swing: max 7 days, 3R (15%) target, re-enter after a 7-day exit if the signal still holds, stop to breakeven after +1.5R (7.5%)


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fetch_hourly(product, days=HISTORY_DAYS):
    """Closed 1h perp candles + hourly funding, naive-UTC index, oldest first."""
    now = pd.Timestamp.now(tz="UTC").floor("h")              # current hour is still forming: exclude it
    start, rows = now - pd.Timedelta(days=days), []
    while start < now:
        end = min(now, start + pd.Timedelta(hours=1000))
        r = requests.get(f"{KRAKEN}/api/charts/v1/trade/{product}/1h",
                         params={"from": int(start.timestamp()), "to": int(end.timestamp())}, timeout=30)
        r.raise_for_status()
        rows += r.json()["candles"]
        start = end
        time.sleep(0.2)                                      # public rate limit
    df = pd.DataFrame(rows).drop_duplicates("time")
    df.index = pd.to_datetime(df.pop("time"), unit="ms")
    df = df.sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[df.index < now.tz_localize(None)]
    df = df.asfreq("h").ffill()                              # hours with no trades
    r = requests.get(f"{KRAKEN}/derivatives/api/v4/historicalfundingrates", params={"symbol": product}, timeout=30)
    r.raise_for_status()
    f = pd.DataFrame(r.json()["rates"])
    # a rate stamped T settles the hour that ended at T, i.e. our bar labelled T-1h
    f = pd.Series(f.relativeFundingRate.astype(float).values,
                  index=pd.to_datetime(f.timestamp).dt.tz_convert(None) - pd.Timedelta(hours=1))
    df["funding"] = f[~f.index.duplicated()].reindex(df.index).fillna(0.0)
    return df


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return dict(risk.new_state(START_CASH), last_ts=None, start_cash=START_CASH)


def run():
    h1 = {c: fetch_hourly(p) for c, p in PRODUCTS.items()}
    idx = h1["BTC"].index.intersection(h1["ETH"].index).intersection(h1["SOL"].index)
    h1 = {c: d.loc[idx] for c, d in h1.items()}
    sig = risk.signals(h1)
    state = load_state()
    if state.get("liquidated"):
        log(f"account was liquidated at {state['liquidated']}; delete {STATE.name} to restart")
        return
    if state["last_ts"] is None:                              # first run: start from the last closed hour
        state["last_ts"] = str(idx[-2])
        if not JOIN_OPEN_TRENDS:
            state["prev"] = {k: int(v.iloc[-2]) for k, v in sig.items()}
        log(f"new paper account ${state['cash']:,.0f}; current signals: "
            + ", ".join(f"{k}={int(v.iloc[-2]):+d}" for k, v in sig.items()))
    new = idx[idx > pd.Timestamp(state["last_ts"])]
    events, curve = [], []
    for ts in new:
        bars = {c: dict(o=d.at[ts, "open"], h=d.at[ts, "high"], l=d.at[ts, "low"], c=d.at[ts, "close"], f=d.at[ts, "funding"])
                for c, d in h1.items()}
        events += risk.step(state, ts, bars, {k: int(v.at[ts]) for k, v in sig.items()}, **CFG)
        state["last_ts"] = str(ts)
        closes = {c: b["c"] for c, b in bars.items()}
        curve.append(dict(ts=ts, equity=risk.equity(state, closes), cash=state["cash"],
                          exposure=sum(t["u"] * closes[k.split("|")[0]] for k, t in state["open"].items())))
        if state["liquidated"]:
            break
    for e in events:
        log(f"{e['event'].upper():6s} {e.get('leg', '')} side={e.get('side', '')} px={e.get('px', 0):,.2f}"
            + (f" stop={e['stop']:,.2f}" if "stop" in e else "") + (f" pnl=${e['pnl']:,.2f} ({e['why']})" if "pnl" in e else ""))
    if events:
        pd.DataFrame(events).to_csv(EVENTS, mode="a", header=not EVENTS.exists(), index=False)
    px = {c: d["close"].iloc[-1] for c, d in h1.items()}
    eq = risk.equity(state, px)
    notional = sum(t["u"] * px[k.split("|")[0]] for k, t in state["open"].items())
    log(f"processed {len(new)} new hour(s) to {state['last_ts']} | equity ${eq:,.2f} | exposure {notional / eq:.2f}x | "
        + ("open: " + ", ".join(f"{k} {'L' if t['d'] > 0 else 'S'}@{t['e']:,.2f}" for k, t in state["open"].items())
           if state["open"] else "flat"))
    if curve:
        pd.DataFrame(curve).to_csv(EQUITY, mode="a", header=not EQUITY.exists(), index=False)
    STATE.write_text(json.dumps(state, indent=1))
    dashboard.render(state, h1, sig, CFG)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # keep the scheduled task alive; next hour retries from saved state
        log(f"ERROR {type(e).__name__}: {e}")
        raise
