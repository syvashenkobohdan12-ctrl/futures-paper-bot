"""PAPER sleeve: daily Donchian breakout (donchian.py) on 10 Kraken perps. Separate $10k paper account; no real orders.
Stateless on purpose: each run fetches ~300 days of Kraken 1h candles + funding, builds closed UTC daily bars, runs the
exact backtest simulator (donchian.trades) and keeps the trades that opened on or after the go-live day. So the
paper record is what the backtest would have done on Kraken prices, and a missed or failed run loses nothing.
Sizing like the backtest's portfolio(): 0.5% of equity risked per trade, at most 10 positions open at once.
Writes donchian_state.json (go-live day + last snapshot), donchian_trades.csv and donchian.html.
shortcut: fills are the daily bar's open/stop/target/close as in the backtest, not live hourly fills."""
import json, datetime as dt, time
from pathlib import Path
import numpy as np, pandas as pd
import donchian
from live_bot import fetch_hourly

HERE = Path(__file__).parent
STATE, TRADES, PAGE = HERE / "donchian_state.json", HERE / "donchian_trades.csv", HERE / "donchian.html"
PRODUCTS = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD", "SOL": "PF_SOLUSD", "XRP": "PF_XRPUSD", "BNB": "PF_BNBUSD",
            "DOGE": "PF_DOGEUSD", "ADA": "PF_ADAUSD", "AVAX": "PF_AVAXUSD", "LINK": "PF_LINKUSD", "SUI": "PF_SUIUSD"}
START_CASH, RISK, MAX_OPEN, HISTORY_DAYS = 10_000, 0.005, 10, 300
AGG = dict(open="first", high="max", low="min", close="last", volume="sum", funding="sum")


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}  donchian  {msg}")


def daily_bars(h1):
    """Closed UTC days only (today's bar is still forming)."""
    d = h1.resample("1D").agg(AGG).dropna()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return d[d.index < today]


def book(trades, last_px, last_day):
    """Replay entries/exits in time order with fixed-fractional sizing. Open = still running at the last closed day."""
    rows, eq, open_ = [], START_CASH, {}
    trades = trades.sort_values("ts").reset_index(drop=True)
    # same timestamp: entries first, so a trade stopped out on its entry day still closes
    events = sorted([(r.ts, 0, i) for i, r in trades.iterrows()] + [(r.exit_ts, 1, i) for i, r in trades.iterrows() if not r.is_open])
    for ts, kind, i in events:
        r = trades.loc[i]
        if kind == 0:
            if len(open_) >= MAX_OPEN:
                rows.append(dict(r, status="skipped (10 open)", risk_usd=0.0, pnl=0.0))
                continue
            open_[i] = eq * RISK
        elif i in open_:
            usd = open_.pop(i)
            eq += usd * r.R
            rows.append(dict(r, status="closed", risk_usd=usd, pnl=usd * r.R))
    unreal = 0.0
    for i, usd in open_.items():
        r = trades.loc[i]
        px = last_px[r.coin]
        risk_px = abs(r.entry - r.stop)
        upnl_r = r.side * (px - r.entry) / risk_px
        unreal += usd * upnl_r
        rows.append(dict(r, status="open", risk_usd=usd, pnl=usd * upnl_r, exit=px))
    out = pd.DataFrame(rows)
    return out, eq, unreal


def run():
    cfg = donchian.LIVE
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
    if "go_live" not in state:                       # first run: trade from the next closed day on
        state["go_live"] = str((last_day + pd.Timedelta(days=1)).date())
        log(f"new paper sleeve ${START_CASH:,}; go-live {state['go_live']}")
    go = pd.Timestamp(state["go_live"])
    parts = []
    for c, d in days.items():
        t = donchian.trades(d, cfg["exit"], cfg["sides"])
        if len(t):
            hold = donchian.EXITS[cfg["exit"]]["hold"]
            held = (t.exit_ts - t.ts).dt.days
            # the simulator closes anything still running at the last bar as "time": those are open positions
            t["is_open"] = (t.exit_ts == d.index[-1]) & (t.why == "time") & (held < hold - 1)
            parts.append(t.assign(coin=c))
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts", "exit_ts", "is_open", "coin"])
    t = t[t.ts >= go]
    last_px = {c: d.close.iloc[-1] for c, d in days.items()}
    ledger, realized, unreal = book(t, last_px, last_day) if len(t) else (pd.DataFrame(), START_CASH, 0.0)
    if len(ledger):
        ledger.drop(columns=["is_open"], errors="ignore").to_csv(TRADES, index=False)
    eq = realized + unreal
    state.update(last_day=str(last_day.date()), equity=round(eq, 2), realized=round(realized, 2),
                 open=int((ledger.status == "open").sum()) if len(ledger) else 0, missing=missing, config=cfg)
    hist = state.setdefault("equity_by_day", {})
    hist[str(last_day.date())] = round(eq, 2)
    STATE.write_text(json.dumps(state, indent=1))
    render(state, ledger, days, cfg)
    log(f"day {last_day.date()} | equity ${eq:,.2f} | open {state['open']} | coins {len(days)}"
        + (f" | missing {', '.join(missing)}" if missing else ""))


def _money(v):
    return ("-" if v < 0 else "") + f"${abs(v):,.0f}"


def _cls(v):
    return "pos" if v > 0 else "neg" if v < 0 else ""


def render(state, ledger, days, cfg):
    ex = donchian.EXITS[cfg["exit"]]
    rows = []
    for c, d in days.items():
        hi = d.high.iloc[-donchian.N:].max()
        lo = d.low.iloc[-donchian.N:].min()
        px = d.close.iloc[-1]
        side = "long only" if cfg["sides"] == "long" else "long / short"
        rows.append(f"<tr><td>{c}</td><td>{px:,.4g}</td><td>{hi:,.4g}</td><td class='{_cls(px / hi - 1)}'>{px / hi - 1:+.1%}</td>"
                    + (f"<td>{lo:,.4g}</td><td>{px / lo - 1:+.1%}</td>" if cfg["sides"] != "long" else "<td colspan='2' class='muted'>longs only</td>")
                    + "</tr>")
    lv_html = "".join(rows)

    op = ledger[ledger.status == "open"] if len(ledger) else ledger
    cl = ledger[ledger.status == "closed"].sort_values("exit_ts", ascending=False) if len(ledger) else ledger
    fmt = lambda v: f"{v:,.4g}"
    op_html = "".join(f"<tr><td>{x.coin}</td><td class='{_cls(x.side)}'>{'LONG' if x.side > 0 else 'SHORT'}</td><td>{x.ts:%Y-%m-%d}</td>"
                      f"<td>{fmt(x.entry)}</td><td>{fmt(x.exit)}</td><td>{fmt(x.stop)}</td><td>{fmt(x.tgt) if x.tgt == x.tgt and x.tgt else 'trail'}</td>"
                      f"<td>{_money(x.risk_usd)}</td><td class='{_cls(x.pnl)}'>{_money(x.pnl)}</td></tr>" for x in op.itertuples()) \
        or "<tr><td colspan='9' class='muted'>No open positions</td></tr>"
    cl_html = "".join(f"<tr><td>{x.coin}</td><td class='{_cls(x.side)}'>{'LONG' if x.side > 0 else 'SHORT'}</td><td>{x.ts:%Y-%m-%d}</td>"
                      f"<td>{x.exit_ts:%Y-%m-%d}</td><td>{fmt(x.entry)}</td><td>{fmt(x.exit)}</td><td>{x.why}</td>"
                      f"<td class='{_cls(x.R)}'>{x.R:+.2f}R</td><td class='{_cls(x.pnl)}'>{_money(x.pnl)}</td></tr>" for x in cl.itertuples()) \
        or "<tr><td colspan='9' class='muted'>No closed trades yet</td></tr>"
    eq = state["equity"]
    wins = int((cl.R > 0).sum()) if len(cl) else 0
    kpis = [("Equity", _money(eq), ""), ("P&L", f"{_money(eq - START_CASH)} ({eq / START_CASH - 1:+.2%})", _cls(eq - START_CASH)),
            ("Open", str(state["open"]), ""), ("Closed", f"{len(cl)} · {wins / len(cl):.0%} won" if len(cl) else "0", ""),
            ("Live since", state["go_live"], "")]
    kpi_html = "".join(f"<div class='kpi'><div class='lab'>{a}</div><div class='val {c}'>{b}</div></div>" for a, b, c in kpis)
    exit_txt = "3R target or 72 days" if cfg["exit"] == "fixed" else f"close below the prior {ex['trail']}-day low (no target), or the stop"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="900"><title>Donchian paper sleeve</title>
<style>
:root{{--bg:#fafaf8;--card:#fff;--ink:#111;--mut:#777;--line:#e5e4de;--pos:#13804b;--neg:#c23b3b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1e1e1c;--ink:#eee;--mut:#999;--line:#333;--pos:#3fbf7f;--neg:#ef6b6b}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}} main{{max-width:1100px;margin:auto;padding:16px}}
h1{{font-size:20px;font-weight:600;margin:0}} h2{{font-size:15px;font-weight:600;margin:24px 0 8px}} .muted{{color:var(--mut)}} a{{color:inherit}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}} .lab{{color:var(--mut);font-size:12px}} .val{{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--mut);font-weight:500;font-size:12px}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
</style></head><body><main>
<h1>Donchian breakout sleeve <span class="muted" style="font-size:13px;font-weight:400">· PAPER · 10 Kraken perps · daily bars · <a href="index.html">main bot</a></span></h1>
<div class="muted">Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC · last closed day {state['last_day']}
{(' · <b class="neg">no data: ' + ', '.join(state['missing']) + '</b>') if state.get('missing') else ''}</div>
<div class="kpis">{kpi_html}</div>
<h2>Open positions</h2><div class="card"><table><tr><th>Coin</th><th>Side</th><th>Opened</th><th>Entry</th><th>Last close</th><th>Stop</th><th>Target</th><th>Risked</th><th>Unrealized</th></tr>{op_html}</table></div>
<h2>Distance to the breakout levels</h2><div class="card"><table><tr><th>Coin</th><th>Last close</th><th>20-day high</th><th>vs high</th><th>20-day low</th><th>vs low</th></tr>{lv_html}</table></div>
<h2>Closed trades</h2><div class="card"><table><tr><th>Coin</th><th>Side</th><th>Opened</th><th>Closed</th><th>Entry</th><th>Exit</th><th>Reason</th><th>R</th><th>P&amp;L</th></tr>{cl_html}</table></div>
<p class="muted">Rules: daily close above the prior 20-day high{' (or below the 20-day low for shorts)' if cfg['sides'] != 'long' else ''} → enter at the next day's open ·
stop 2×ATR(14) · exit: {exit_txt} · 0.5% of equity risked per trade · max 10 open. Backtest 2020–2026 on Binance data: see the smc-backtest branch.</p>
</main></body></html>"""
    PAGE.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log(f"ERROR {type(e).__name__}: {e}")
        raise
