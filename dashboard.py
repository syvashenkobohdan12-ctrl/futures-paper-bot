"""Writes dashboard.html after every bot run: account summary, equity + drawdown charts, open positions with
stops/targets, per-strategy signal status, market prices/funding and the recent trade log. Self-refreshes every 5 min."""
import json, datetime as dt
from pathlib import Path
import pandas as pd
from combo import AGG, to_hourly
from strategies import adx

HERE = Path(__file__).parent
OUT = HERE / "dashboard.html"


def _money(v):
    return ("-" if v < 0 else "") + f"${abs(v):,.0f}"


def _cls(v):
    return "pos" if v > 0 else "neg" if v < 0 else ""


def render(state, h1, sig, cfg):
    px = {c: d.close.iloc[-1] for c, d in h1.items()}
    start = state.get("start_cash", 100_000)
    eq_now = state["cash"] + sum(t["d"] * t["u"] * (px[k.split("|")[0]] - t["e"]) for k, t in state["open"].items())
    curve = pd.read_csv(HERE / "paper_equity.csv", parse_dates=["ts"]) if (HERE / "paper_equity.csv").exists() else pd.DataFrame(columns=["ts", "equity", "exposure"])
    peak = max([start, *curve.equity.tolist(), eq_now])
    events = pd.read_csv(HERE / "paper_events.csv") if (HERE / "paper_events.csv").exists() else pd.DataFrame()
    closed = events[events.event == "close"] if len(events) else pd.DataFrame(columns=["pnl"])
    wins = int((closed.pnl > 0).sum()) if len(closed) else 0
    notional = sum(t["u"] * px[k.split("|")[0]] for k, t in state["open"].items())
    last_bar = pd.Timestamp(state["last_ts"])

    # open positions
    pos_rows = []
    for k, t in state["open"].items():
        c, sleeve = k.split("|")
        p, d = px[c], t["d"]
        tp = t["e"] * (1 + d * cfg["tp_r"] * cfg["stop_pct"])
        left = pd.Timestamp(t["entry"]) + pd.Timedelta(hours=cfg["max_hold"]) - last_bar
        upnl = d * t["u"] * (p - t["e"])
        pos_rows.append(f"<tr><td>{c}</td><td>{sleeve}</td><td class='{_cls(d)}'>{'LONG' if d > 0 else 'SHORT'}</td>"
                        f"<td>{t['entry'][:16]}</td><td>{t['e']:,.2f}</td><td>{p:,.2f}</td><td>{_money(t['u'] * p)}</td>"
                        f"<td>{t['sp']:,.2f}{' (BE)' if t.get('be') else ''}</td><td>{tp:,.2f}</td>"
                        f"<td>{max(left.total_seconds(), 0) / 86400:.1f}d</td><td class='{_cls(upnl)}'>{_money(upnl)}</td></tr>")
    pos_html = "".join(pos_rows) or "<tr><td colspan='11' class='muted'>No open positions</td></tr>"

    # signal status per strategy leg
    sig_rows = []
    for k, s in sig.items():
        c, sleeve = k.split("|")
        now = int(s.iloc[-1])
        d1 = h1[c].resample("1D").agg(AGG).dropna()
        a = to_hourly(adx(d1), 24, h1[c].index).iloc[-1]
        if k in state["open"]:
            status = "in trade"
        elif state["blocked"].get(k):
            status = "waiting for signal flip (exited on stop/target)"
        elif now == 0:
            status = "skipped: this trend began while daily ADX was > 40 (stretched); waits for the next flip"
        else:
            status = "waiting for a fresh flip (joined mid-trend not allowed)"
        sig_rows.append(f"<tr><td>{c}</td><td>{sleeve}</td><td class='{_cls(now)}'>{['SHORT', 'FLAT', 'LONG'][now + 1]}</td>"
                        f"<td>{a:.0f}</td><td>{status}</td></tr>")

    mkt = "".join(f"<tr><td>{c}</td><td>{d.close.iloc[-1]:,.2f}</td><td class='{_cls(d.close.iloc[-1] / d.close.iloc[-25] - 1)}'>"
                  f"{d.close.iloc[-1] / d.close.iloc[-25] - 1:+.2%}</td><td>{d.funding.iloc[-24:].sum() * 365:+.1%}</td></tr>"
                  for c, d in h1.items())

    ev_rows = []
    for e in events.tail(25).iloc[::-1].itertuples() if len(events) else []:
        pnl = getattr(e, "pnl", float("nan"))
        ev_rows.append(f"<tr><td>{str(e.ts)[:16]}</td><td>{e.event.upper()}</td><td>{e.leg}</td>"
                       f"<td class='{_cls(e.side)}'>{'LONG' if e.side > 0 else 'SHORT'}</td><td>{e.px:,.2f}</td>"
                       f"<td>{getattr(e, 'why', '') if isinstance(getattr(e, 'why', ''), str) else ''}</td>"
                       f"<td class='{_cls(pnl) if pnl == pnl else ''}'>{_money(pnl) if pnl == pnl else ''}</td></tr>")
    ev_html = "".join(ev_rows) or "<tr><td colspan='7' class='muted'>No trades yet</td></tr>"

    data = dict(t=[str(x)[:16] for x in curve.ts], eq=[round(v, 2) for v in curve.equity],
                dd=[round(v, 2) for v in (curve.equity / curve.equity.cummax().clip(lower=start) - 1) * 100] if len(curve) else [])
    kpis = [("Equity", _money(eq_now), ""), ("P&L", f"{_money(eq_now - start)} ({eq_now / start - 1:+.2%})", _cls(eq_now - start)),
            ("Drawdown", f"{eq_now / peak - 1:.2%}", _cls(eq_now / peak - 1)), ("Exposure", f"{notional / eq_now:.2f}x", ""),
            ("Closed trades", f"{len(closed)} · {wins / len(closed):.0%} won" if len(closed) else "0", ""),
            ("Funding P&L", _money(-state.get("funding_paid", 0.0)), _cls(-state.get("funding_paid", 0.0)))]
    kpi_html = "".join(f"<div class='kpi'><div class='lab'>{a}</div><div class='val {c}'>{b}</div></div>" for a, b, c in kpis)

    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300"><title>Futures paper bot</title>
<style>
:root{{--bg:#fafaf8;--card:#fff;--ink:#111;--mut:#777;--line:#e5e4de;--pos:#13804b;--neg:#c23b3b;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1e1e1c;--ink:#eee;--mut:#999;--line:#333;--pos:#3fbf7f;--neg:#ef6b6b;--acc:#3987e5}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}} main{{max-width:1100px;margin:auto;padding:16px}}
h1{{font-size:20px;font-weight:600;margin:0}} h2{{font-size:15px;font-weight:600;margin:24px 0 8px}} .muted{{color:var(--mut)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}} .lab{{color:var(--mut);font-size:12px}} .val{{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--mut);font-weight:500;font-size:12px}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}} .chart{{position:relative;height:260px}}
</style></head><body><main>
<h1>Futures paper bot <span class="muted" style="font-size:13px;font-weight:400">· PAPER · Kraken perps BTC/ETH/SOL · no real orders</span></h1>
<div class="muted">Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC · last closed bar {state['last_ts'][:16]} UTC · page refreshes every 5 min
{' · <b class="neg">LIQUIDATED ' + state['liquidated'] + '</b>' if state.get('liquidated') else ''}</div>
<div class="kpis">{kpi_html}</div>
<h2>Equity</h2><div class="card"><div class="chart"><canvas id="eq"></canvas></div><div id="eqmsg" class="muted"></div></div>
<h2>Drawdown from peak</h2><div class="card"><div class="chart" style="height:160px"><canvas id="dd"></canvas></div></div>
<h2>Open positions</h2><div class="card"><table><tr><th>Coin</th><th>Strategy</th><th>Side</th><th>Opened (UTC)</th><th>Entry</th><th>Now</th><th>Size</th><th>Stop</th><th>Target</th><th>Time left</th><th>Unrealized</th></tr>{pos_html}</table></div>
<h2>Signals</h2><div class="card"><table><tr><th>Coin</th><th>Strategy</th><th>Signal</th><th>Daily ADX</th><th>Status</th></tr>{''.join(sig_rows)}</table></div>
<h2>Market</h2><div class="card"><table><tr><th>Perp</th><th>Price</th><th>24h</th><th>Funding (annualized, last 24h; + = longs pay)</th></tr>{mkt}</table></div>
<h2>Recent activity</h2><div class="card"><table><tr><th>Bar (UTC)</th><th>Event</th><th>Leg</th><th>Side</th><th>Price</th><th>Exit reason</th><th>P&amp;L</th></tr>{ev_html}</table></div>
<p class="muted">Rules: 2% risk/trade · 5% stop · 15% target · max 7 days (re-entry if signal holds) · stop to breakeven after +7.5% · 10x cap.</p>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const D={json.dumps(data)}, css=getComputedStyle(document.documentElement), acc=css.getPropertyValue('--acc').trim(), mut=css.getPropertyValue('--mut').trim(), line=css.getPropertyValue('--line').trim();
if(D.eq.length<2){{document.getElementById('eqmsg').textContent='The curve fills in as hours are processed (one point per hour).';}}
const opt=(fmt)=>({{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{display:false}}}},
 scales:{{x:{{ticks:{{color:mut,maxTicksLimit:8,maxRotation:0}},grid:{{display:false}}}},y:{{ticks:{{color:mut,callback:fmt}},grid:{{color:line}}}}}}}});
new Chart(eq,{{type:'line',data:{{labels:D.t,datasets:[{{data:D.eq,borderColor:acc,borderWidth:2,pointRadius:0,tension:.1}}]}},options:opt(v=>'$'+Math.round(v).toLocaleString())}});
new Chart(dd,{{type:'line',data:{{labels:D.t,datasets:[{{data:D.dd,borderColor:acc,backgroundColor:acc+'22',fill:'origin',borderWidth:1.5,pointRadius:0}}]}},options:opt(v=>v.toFixed(1)+'%')}});
</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
