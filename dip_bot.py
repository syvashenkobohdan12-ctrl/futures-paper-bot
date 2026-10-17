"""PAPER sleeve: dip bounce on 10 Kraken perps. Separate $10k paper account; no real orders.
  BUY  when a coin's daily close is more than 8% below the previous close (buy at that close)
  SELL at the next day's close
Long only, 10% of the account per trade, at most 3 trades a day (the 3 biggest drops) so a market-wide crash
can't put most of the account into one bet. Stateless like donchian_bot: rebuilt from Kraken history each run.
Backtest (dip.py, 2020-26, Binance daily, fees + funding): 58% of trades win, +1.96% per trade; with the 3-a-day
cap +16%/yr, worst day -8%, worst drawdown -15%. Writes dip_state.json, dip_trades.csv and dip.html."""
import json, datetime as dt
from pathlib import Path
import pandas as pd
import donchian_bot

HERE = Path(__file__).parent
STATE, TRADES, PAGE = HERE / "dip_state.json", HERE / "dip_trades.csv", HERE / "dip.html"
START_CASH, SLICE, DROP, MAX_PER_DAY, COST = 10_000, 0.10, 0.08, 3, 0.0007


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}  dip  {msg}")


def signals(days, go):
    """Every (day, coin, drop) from go-live on where the close fell more than DROP; capped to the biggest MAX_PER_DAY."""
    rows = []
    for c, d in days.items():
        r = d.close.pct_change()
        for ts in d.index[(r < -DROP) & (d.index >= go)]:
            rows.append(dict(day=ts, coin=c, drop=float(r[ts]), entry=float(d.close[ts])))
    s = pd.DataFrame(rows, columns=["day", "coin", "drop", "entry"])
    return s.sort_values(["day", "drop"]).groupby("day").head(MAX_PER_DAY).sort_values("day")


def run(h1=None, missing=None):
    if h1 is None:                                   # standalone: fetch like the breakout sleeve does
        h1, missing = {}, []
        for c, p in donchian_bot.PRODUCTS.items():
            try:
                h1[c] = donchian_bot.fetch_hourly(p, 150)
            except Exception as e:
                missing.append(f"{c} ({type(e).__name__})")
    missing = missing or []
    days = {c: donchian_bot.daily_bars(d) for c, d in h1.items()}
    last_day = max(d.index[-1] for d in days.values())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "go_live" not in state:
        state["go_live"] = str((last_day + pd.Timedelta(days=1)).date())
        log(f"new paper sleeve ${START_CASH:,}; go-live {state['go_live']}")
    sig = signals(days, pd.Timestamp(state["go_live"]))
    ledger = []
    for t in sig.to_dict("records"):
        d = days[t["coin"]]
        nxt = d.index[d.index > t["day"]]
        booked = START_CASH + sum(x["pnl"] for x in ledger if x["status"] == "closed" and x["exit_day"] <= t["day"])
        t["usd"] = SLICE * booked
        if len(nxt):                                 # sold at the next day's close
            x = nxt[0]
            ret = d.close[x] / t["entry"] - 1 - 2 * COST - d.funding[x]
            t.update(exit_day=x, exit=float(d.close[x]), status="closed")
        else:                                        # bought at the last close, sells at the next one
            ret = -COST
            t.update(exit_day=None, exit=None, status="open")
        t.update(ret=ret, pnl=t["usd"] * ret)
        ledger.append(t)
    led = pd.DataFrame(ledger)
    if len(led):
        led.to_csv(TRADES, index=False)
    eq = START_CASH + (led.pnl.sum() if len(led) else 0.0)
    state.update(last_day=str(last_day.date()), equity=round(eq, 2), open=int((led.status == "open").sum()) if len(led) else 0,
                 trades=int(len(led)), missing=missing)
    state.setdefault("equity_by_day", {})[str(last_day.date())] = round(eq, 2)
    STATE.write_text(json.dumps(state, indent=1))
    render(state, led, days)
    log(f"day {last_day.date()} | equity ${eq:,.2f} | trades {len(led)} | open {state['open']}")


def render(state, led, days):
    m, cls = donchian_bot._money, donchian_bot._cls
    now = []
    for c, d in days.items():
        r = d.close.iloc[-1] / d.close.iloc[-2] - 1
        trig = d.close.iloc[-1] * (1 - DROP)
        now.append(f"<tr><td>{c}</td><td>{d.close.iloc[-1]:,.4g}</td><td class='{cls(r)}'>{r:+.1%}</td><td>{trig:,.4g}</td></tr>")
    rows = "".join(f"<tr><td>{x.coin}</td><td>{x.day:%Y-%m-%d}</td><td class='neg'>{x.drop:+.1%}</td><td>{x.entry:,.4g}</td>"
                   f"<td>{'' if x.exit is None or x.exit != x.exit else f'{x.exit:,.4g}'}</td><td>{x.status}</td>"
                   f"<td class='{cls(x.ret)}'>{x.ret:+.1%}</td><td class='{cls(x.pnl)}'>{m(x.pnl)}</td></tr>"
                   for x in led.iloc[::-1].itertuples()) if len(led) else ""
    rows = rows or "<tr><td colspan='8' class='muted'>No trades yet: waits for a daily close more than 8% down</td></tr>"
    closed = led[led.status == "closed"] if len(led) else led
    eq = state["equity"]
    kpis = [("Equity", m(eq), ""), ("P&L", f"{m(eq - START_CASH)} ({eq / START_CASH - 1:+.2%})", cls(eq - START_CASH)),
            ("Trades", f"{len(led)}" + (f" · {(closed.ret > 0).mean():.0%} won" if len(closed) else ""), ""),
            ("Live since", state["go_live"], "")]
    kpi_html = "".join(f"<div class='kpi'><div class='lab'>{a}</div><div class='val {c}'>{b}</div></div>" for a, b, c in kpis)
    style = donchian_bot.PAGE.read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0] if donchian_bot.PAGE.exists() else ""
    PAGE.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="900"><title>Dip paper sleeve</title><style>{style}</style></head><body><main>
<h1>Dip sleeve <span class="muted" style="font-size:13px;font-weight:400">· PAPER · 10 Kraken perps · <a href="index.html">main bot</a> · <a href="donchian.html">breakout sleeve</a></span></h1>
<div class="muted">Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC · last closed day {state['last_day']}
{(' · <b class="neg">no data: ' + ', '.join(state['missing']) + '</b>') if state.get('missing') else ''}</div>
<div class="rules"><b>Buy</b> when a coin closes the day more than 8% down. <b>Sell</b> at the next day's close. 10% of the account per trade, at most 3 a day.</div>
<div class="kpis">{kpi_html}</div>
<h2>Each coin now</h2><div class="card"><table><tr><th>Coin</th><th>Last close</th><th>Last day</th><th>Buys if a close is below</th></tr>{''.join(now)}</table></div>
<h2>Trades</h2><div class="card"><table><tr><th>Coin</th><th>Day</th><th>Drop</th><th>Buy</th><th>Sell</th><th>Status</th><th>Return</th><th>P&amp;L</th></tr>{rows}</table></div>
<p class="muted">Backtest 2020–2026 (Binance daily, fees and funding): 58% of trades win, +1.96% per trade; with the 3-a-day cap +16%/yr, worst drawdown −15%. Most of the profit came in 2020–21.</p>
</main></body></html>""", encoding="utf-8")


if __name__ == "__main__":
    run()
