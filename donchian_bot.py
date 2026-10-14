"""PAPER sleeve: the simple breakout (simple.py) on 10 Kraken perps. Separate $10k paper account; no real orders.
  BUY  when the daily close is the highest close of the last 20 days
  SELL when the daily close is the lowest close of the last 10 days
Long only, each buy gets 10% of the account. Stateless: each run fetches Kraken 1h candles, builds closed UTC days,
applies simple.position() and rebuilds the trades since go-live (signals from before go-live are ignored).
Writes donchian_state.json, donchian_trades.csv and donchian.html."""
import json, datetime as dt, time
from pathlib import Path
import pandas as pd
import simple
from live_bot import fetch_hourly

HERE = Path(__file__).parent
STATE, TRADES, PAGE = HERE / "donchian_state.json", HERE / "donchian_trades.csv", HERE / "donchian.html"
PRODUCTS = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD", "SOL": "PF_SOLUSD", "XRP": "PF_XRPUSD", "BNB": "PF_BNBUSD",
            "DOGE": "PF_DOGEUSD", "ADA": "PF_ADAUSD", "AVAX": "PF_AVAXUSD", "LINK": "PF_LINKUSD", "SUI": "PF_SUIUSD"}
START_CASH, SLICE, HISTORY_DAYS = 10_000, 0.10, 120
AGG = dict(open="first", high="max", low="min", close="last", volume="sum", funding="sum")


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}  breakout  {msg}")


def daily_bars(h1):
    d = h1.resample("1D").agg(AGG).dropna()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return d[d.index < today]                       # closed days only


def trades_since(d, go):
    """Buy/sell at the signal day's close. A position already held at go-live is ignored until it resets."""
    pos = simple.position(d.close)
    out, held, armed = [], 0, False
    for ts, p, c, f in zip(d.index, pos, d.close, d.funding):
        if ts < go:
            continue
        if not armed:                                # wait until we're flat once, so we never join mid-trend
            armed = p == 0
            continue
        if p and not held:
            out.append(dict(entry_day=ts, entry=c, funding=0.0))
        elif held:
            out[-1]["funding"] += f
        if not p and held:
            out[-1].update(exit_day=ts, exit=c)
        held = p
    return out


def run():
    h1, missing = {}, []
    for c, p in PRODUCTS.items():
        try:
            h1[c] = fetch_hourly(p, HISTORY_DAYS)
        except Exception as e:
            missing.append(f"{c} ({type(e).__name__})")
        time.sleep(0.3)
    if not h1:
        raise RuntimeError("no Kraken data")
    days = {c: daily_bars(d) for c, d in h1.items()}
    last_day = max(d.index[-1] for d in days.values())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("rules") != "simple":               # (re)start the paper account on these rules
        state = dict(go_live=str((last_day + pd.Timedelta(days=1)).date()), rules="simple")
        log(f"new paper sleeve ${START_CASH:,}; go-live {state['go_live']}")
    go = pd.Timestamp(state["go_live"])
    rows = [dict(t, coin=c) for c, d in days.items() for t in trades_since(d, go)]
    rows.sort(key=lambda t: t["entry_day"])
    ledger = []
    for t in rows:                                   # each buy gets 10% of the cash booked by trades closed before it
        booked = START_CASH + sum(x["pnl"] for x in ledger if x["status"] == "closed" and x["exit_day"] < t["entry_day"])
        t["usd"] = SLICE * booked
        px = t.get("exit", days[t["coin"]].close.iloc[-1])
        ret = px / t["entry"] - 1 - t["funding"] - simple.COST * (2 if "exit" in t else 1)
        t.update(status="closed" if "exit" in t else "open", ret=ret, pnl=t["usd"] * ret, last=px)
        ledger.append(t)
    cash = START_CASH + sum(x["pnl"] for x in ledger if x["status"] == "closed")
    led = pd.DataFrame(ledger)
    if len(led):
        led.to_csv(TRADES, index=False)
    eq = cash + (led.pnl[led.status == "open"].sum() if len(led) else 0.0)
    state.update(last_day=str(last_day.date()), equity=round(eq, 2), open=int((led.status == "open").sum()) if len(led) else 0,
                 missing=missing)
    state.setdefault("equity_by_day", {})[str(last_day.date())] = round(eq, 2)
    STATE.write_text(json.dumps(state, indent=1))
    render(state, led, days)
    log(f"day {last_day.date()} | equity ${eq:,.2f} | open {state['open']}" + (f" | missing {', '.join(missing)}" if missing else ""))


def _money(v):
    return ("-" if v < 0 else "") + f"${abs(v):,.0f}"


def _cls(v):
    return "pos" if v > 0 else "neg" if v < 0 else ""


def render(state, led, days):
    held = set(led.coin[led.status == "open"]) if len(led) else set()
    lv = []
    for c, d in days.items():
        px, hi, lo = d.close.iloc[-1], d.close.iloc[-simple.BUY_DAYS:].max(), d.close.iloc[-simple.SELL_DAYS:].min()
        status = f"holding · sells below {lo:,.4g}" if c in held else f"flat · buys above {hi:,.4g}"
        lv.append(f"<tr><td>{c}</td><td>{px:,.4g}</td><td>{status}</td><td class='{_cls(px / hi - 1)}'>{px / hi - 1:+.1%} from 20-day high</td></tr>")
    op = led[led.status == "open"] if len(led) else led
    cl = led[led.status == "closed"].sort_values("exit_day", ascending=False) if len(led) else led
    op_html = "".join(f"<tr><td>{x.coin}</td><td>{x.entry_day:%Y-%m-%d}</td><td>{x.entry:,.4g}</td><td>{x.last:,.4g}</td>"
                      f"<td>{_money(x.usd)}</td><td class='{_cls(x.ret)}'>{x.ret:+.1%}</td><td class='{_cls(x.pnl)}'>{_money(x.pnl)}</td></tr>"
                      for x in op.itertuples()) or "<tr><td colspan='7' class='muted'>Nothing held</td></tr>"
    cl_html = "".join(f"<tr><td>{x.coin}</td><td>{x.entry_day:%Y-%m-%d}</td><td>{x.exit_day:%Y-%m-%d}</td><td>{x.entry:,.4g}</td><td>{x.exit:,.4g}</td>"
                      f"<td class='{_cls(x.ret)}'>{x.ret:+.1%}</td><td class='{_cls(x.pnl)}'>{_money(x.pnl)}</td></tr>"
                      for x in cl.itertuples()) or "<tr><td colspan='7' class='muted'>No closed trades yet</td></tr>"
    eq = state["equity"]
    kpis = [("Equity", _money(eq), ""), ("P&L", f"{_money(eq - START_CASH)} ({eq / START_CASH - 1:+.2%})", _cls(eq - START_CASH)),
            ("Holding", f"{state['open']} of 10", ""), ("Closed trades", str(len(cl)), ""), ("Live since", state["go_live"], "")]
    kpi_html = "".join(f"<div class='kpi'><div class='lab'>{a}</div><div class='val {c}'>{b}</div></div>" for a, b, c in kpis)
    PAGE.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="900"><title>Breakout paper sleeve</title>
<style>
:root{{--bg:#fafaf8;--card:#fff;--ink:#111;--mut:#777;--line:#e5e4de;--pos:#13804b;--neg:#c23b3b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1e1e1c;--ink:#eee;--mut:#999;--line:#333;--pos:#3fbf7f;--neg:#ef6b6b}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}} main{{max-width:1000px;margin:auto;padding:16px}}
h1{{font-size:20px;font-weight:600;margin:0}} h2{{font-size:15px;font-weight:600;margin:24px 0 8px}} .muted{{color:var(--mut)}} a{{color:inherit}}
.rules{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:14px;font-size:15px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}} .lab{{color:var(--mut);font-size:12px}} .val{{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--mut);font-weight:500;font-size:12px}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
</style></head><body><main>
<h1>Breakout sleeve <span class="muted" style="font-size:13px;font-weight:400">· PAPER · 10 Kraken perps · <a href="index.html">main bot</a></span></h1>
<div class="muted">Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC · last closed day {state['last_day']}
{(' · <b class="neg">no data: ' + ', '.join(state['missing']) + '</b>') if state.get('missing') else ''}</div>
<div class="rules"><b>Buy</b> when the daily close is the highest in 20 days. <b>Sell</b> when it's the lowest in 10 days. 10% of the account per coin, long only.</div>
<div class="kpis">{kpi_html}</div>
<h2>Each coin now</h2><div class="card"><table><tr><th>Coin</th><th>Last close</th><th>Status</th><th>Distance</th></tr>{''.join(lv)}</table></div>
<h2>Holding</h2><div class="card"><table><tr><th>Coin</th><th>Bought</th><th>Price</th><th>Now</th><th>Size</th><th>Return</th><th>P&amp;L</th></tr>{op_html}</table></div>
<h2>Closed trades</h2><div class="card"><table><tr><th>Coin</th><th>Bought</th><th>Sold</th><th>Buy</th><th>Sell</th><th>Return</th><th>P&amp;L</th></tr>{cl_html}</table></div>
<p class="muted">Backtest 2020–2026 (Binance daily, fees and funding): +61%/yr, worst drawdown −34%, vs buy &amp; hold +69%/yr and −76%.</p>
</main></body></html>""", encoding="utf-8")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log(f"ERROR {type(e).__name__}: {e}")
        raise
