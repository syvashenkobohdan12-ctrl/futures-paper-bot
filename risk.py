"""Combined bot with fixed-fractional sizing: each trade risks RISK of equity at its stop, total notional capped
at MAX_LEV x equity. Signals = combo.py sleeves (incl. ADX gate), direction only.
Each sleeve x coin is its own trade: fixed coin units from entry until signal flip or stop. After a stop,
stay flat until that sleeve's signal flips. Fees+slippage on entry/exit notional, funding on open notional.
step() is the single source of trading logic: the backtest below and live_bot.py both call it.
shortcut: the two sleeves on one coin are separate trades (hedge mode); a one-way account would net them and pay less fees.
shortcut: no contract-size rounding; at $10k, nano contracts (~$1k BTC) round each trade by up to ~10%."""
import numpy as np, pandas as pd
import data
from backtest import SYMS, COST
from combo import sleeve_positions

START_USD, RISK, STOP, MAX_LEV, MAINT = 10_000, 0.02, 0.05, 10, 0.005


def new_state(cash=START_USD):
    return dict(cash=cash, open={}, blocked={}, prev={}, liquidated=None)


def equity(state, px):
    return state["cash"] + sum(t["d"] * t["u"] * (px[k.split("|")[0]] - t["e"]) for k, t in state["open"].items())


def step(state, ts, bars, sigs, risk=RISK, stop_pct=STOP, max_lev=MAX_LEV, cost=COST, max_hold=None, tp_r=None, reenter=(), be_r=None, be_lock=0.002, signal_exit=True):
    """Advance one closed bar. bars: {coin: dict(o,h,l,c,f)}; sigs: {"coin|sleeve": -1/0/1} known at this close.
    max_hold: close after this many hours; tp_r: take profit at tp_r x the stop distance. Either exit waits
    for the next signal flip before re-entering, unless its reason is in `reenter` ("time"/"target"): then the bot
    may re-enter at a later close while the signal still points the same way (fresh stop/target).
    be_r: once price has moved be_r x the stop distance in our favour, move the stop to entry + be_lock (covers fees).
    signal_exit=False: an opposite signal no longer closes an open trade (only stop/target/time do).
    Mutates state; returns list of events (dicts)."""
    ev, opn = [], state["open"]
    rearm = state.setdefault("rearm", {})
    close_px = {c: b["c"] for c, b in bars.items()}

    def close(k, px, why):
        t = opn.pop(k)
        pnl = t["d"] * t["u"] * (px - t["e"]) - t["u"] * px * cost
        state["cash"] += pnl
        ev.append(dict(ts=str(ts), event="close", leg=k, side=t["d"], units=t["u"], px=px, why=why,
                       pnl=pnl + t["acc"], entry_ts=t["entry"]))

    # 1) intrabar: funding, stops, liquidation check
    worst = None
    for k in list(opn):
        t, b = opn[k], bars[k.split("|")[0]]
        if b["f"]:
            fee = t["d"] * t["u"] * b["c"] * b["f"]
            state["cash"] -= fee
            state["funding_paid"] = state.get("funding_paid", 0.0) + fee
            t["acc"] -= fee
        adverse = b["l"] if t["d"] > 0 else b["h"]
        if (t["d"] > 0 and adverse <= t["sp"]) or (t["d"] < 0 and adverse >= t["sp"]):
            close(k, min(t["sp"], b["o"]) if t["d"] > 0 else max(t["sp"], b["o"]), "stop")   # gaps fill worse
            state["blocked"][k] = True
            continue
        if be_r and not t.get("be"):   # applies from the next bar: conservative when stop and trigger share a bar
            fav = (b["h"] / t["e"] - 1) if t["d"] > 0 else (1 - b["l"] / t["e"])
            if fav >= be_r * stop_pct:
                t["sp"], t["be"] = t["e"] * (1 + t["d"] * be_lock), True
        tp = t["e"] * (1 + t["d"] * tp_r * stop_pct) if tp_r else None
        if tp and ((t["d"] > 0 and b["h"] >= tp) or (t["d"] < 0 and b["l"] <= tp)):   # same bar as stop: stop wins
            close(k, max(tp, b["o"]) if t["d"] > 0 else min(tp, b["o"]), "target")
            state["blocked"][k] = "target" not in reenter
            rearm[k] = "target" in reenter
    worst = state["cash"] + sum(t["d"] * t["u"] * ((bars[k.split("|")[0]]["l"] if t["d"] > 0 else bars[k.split("|")[0]]["h"]) - t["e"])
                                for k, t in opn.items())
    notional = sum(t["u"] * close_px[k.split("|")[0]] for k, t in opn.items())
    if worst <= MAINT * notional:
        state["liquidated"] = str(ts)
        ev.append(dict(ts=str(ts), event="LIQUIDATED"))
        return ev
    # 2) at the close: signal flips -> exit / enter
    if max_hold:
        for k in [k for k, t in opn.items() if pd.Timestamp(ts) - pd.Timestamp(t["entry"]) >= pd.Timedelta(hours=max_hold)]:
            close(k, close_px[k.split("|")[0]], "time")
            state["blocked"][k] = "time" not in reenter
            rearm[k] = "time" in reenter
    eq = equity(state, close_px)
    for k, s in sigs.items():
        px, prev = close_px[k.split("|")[0]], state["prev"].get(k, 0)
        if k in opn and s != opn[k]["d"] and signal_exit:
            close(k, px, "signal")
        if s != prev:
            state["blocked"][k] = False
            rearm[k] = False
        if s != 0 and k not in opn and not state["blocked"].get(k) and (s != prev or rearm.get(k)):
            rearm[k] = False
            notional = sum(t["u"] * close_px[j.split("|")[0]] for j, t in opn.items())
            n = min(risk * eq / stop_pct, max(0.0, max_lev * eq - notional))   # $ so a stop-out loses `risk`
            if n > 0:
                u = n / px
                state["cash"] -= u * px * cost
                opn[k] = dict(d=int(s), u=u, e=px, sp=px * (1 - s * stop_pct), entry=str(ts), acc=-u * px * cost)
                ev.append(dict(ts=str(ts), event="open", leg=k, side=int(s), units=u, px=px, stop=opn[k]["sp"]))
        state["prev"][k] = int(s)
    return ev


def signals(h1_by_coin):
    """{coin: hourly df} -> {"coin|sleeve": hourly sign series}"""
    return {f"{c}|{k}": np.sign(p).astype(int) for c, h1 in h1_by_coin.items() for k, p in sleeve_positions(h1).items()}


def simulate(start, stop_pct, risk=RISK, max_lev=MAX_LEV, syms=SYMS, **exits):
    h1s = {s.removesuffix("USDT"): data.load(s) for s in syms}
    common = sorted(set.intersection(*(set(h.index) for h in h1s.values())))   # stocks: only bars all symbols share
    h1s = {c: h.loc[common] for c, h in h1s.items()}
    sig = {k: v.loc[start:].values for k, v in signals(h1s).items()}
    h1s = {c: h.loc[start:] for c, h in h1s.items()}
    idx = next(iter(h1s.values())).index
    arr = {c: {f: h[col].values for f, col in zip("ohlcf", ["open", "high", "low", "close", "funding"])} for c, h in h1s.items()}
    state, trades, curve, max_used = new_state(), [], np.zeros(len(idx)), 0.0
    for i, ts in enumerate(idx):
        bars = {c: {f: a[f][i] for f in "ohlcf"} for c, a in arr.items()}
        for e in step(state, ts, bars, {k: v[i] for k, v in sig.items()}, risk, stop_pct, max_lev, **exits):
            if e["event"] == "close":
                trades.append(e)
        if state["liquidated"]:
            break
        px = {c: b["c"] for c, b in bars.items()}
        eq = equity(state, px)
        notional = sum(t["u"] * px[k.split("|")[0]] for k, t in state["open"].items())
        max_used = max(max_used, notional / eq if eq > 0 else 0)
        curve[i] = eq
    if not state["liquidated"]:
        for k in list(state["open"]):
            t = state["open"].pop(k)
            px = arr[k.split("|")[0]]["c"][-1]
            pnl = t["d"] * t["u"] * (px - t["e"]) - t["u"] * px * COST
            state["cash"] += pnl
            trades.append(dict(leg=k, side=t["d"], pnl=pnl + t["acc"], why="open", entry_ts=t["entry"], ts=str(idx[-1])))
    t = pd.DataFrame(trades)
    t[["coin", "sleeve"]] = t.leg.str.split("|", expand=True)
    t = t.rename(columns={"why": "exit_why", "entry_ts": "entry", "ts": "exit"})
    t["side"] = np.where(t.side > 0, "long", "short")
    t["hours"] = (pd.to_datetime(t.exit) - pd.to_datetime(t.entry)).dt.total_seconds() / 3600
    return pd.Series(curve, idx), t[["coin", "sleeve", "side", "entry", "exit", "hours", "pnl", "exit_why"]], max_used, state["liquidated"]


def report(eq, t, max_used, liq):
    w, l = t[t.pnl > 0], t[t.pnl <= 0]
    m = eq.resample("ME").last()
    return dict(final=eq.iloc[-1], max_dd=(eq / eq.cummax() - 1).min(), low=eq.min(),
                trades=len(t), win=len(w) / len(t), rr=w.pnl.mean() / -l.pnl.mean(),
                pf=w.pnl.sum() / -l.pnl.sum(), stops=int((t.exit_why == "stop").sum()),
                worst_trade=t.pnl.min(), max_lev_used=max_used, avg_hold_h=t.hours.mean(),
                pos_months=f"{int((m.diff().fillna(m.iloc[0] - START_USD) > 0).sum())}/{len(m)}",
                liquidated=str(liq)[:10] if liq is not None else "")


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20)
    for start in ["2025-01-01", "2026-01-01", "2026-07-01"]:
        rows = {}
        for sp in [0.05, 0.08, 0.12]:
            eq, t, mu, liq = simulate(start, sp)
            rows[f"stop {sp:.0%}"] = report(eq, t, mu, liq)
            if start == "2025-01-01" and sp == STOP:
                t.to_csv("risk_trades.csv", index=False)
                eq.resample("1D").last().to_csv("risk_equity_10k.csv")
        df = pd.DataFrame(rows).T
        for c in ["final", "low", "worst_trade"]:
            df[c] = df[c].map(lambda v: f"${v:,.0f}")
        for c in ["max_dd", "win"]:
            df[c] = df[c].map(lambda v: f"{v:.0%}")
        for c in ["rr", "pf"]:
            df[c] = df[c].map(lambda v: f"{v:.2f}")
        df["max_lev_used"] = df["max_lev_used"].map(lambda v: f"{v:.2f}x")
        print(f"\n=== $10,000 from {start}, risk 2%/trade, max 10x")
        print(df)
