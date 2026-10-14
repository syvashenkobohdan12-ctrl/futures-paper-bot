"""Liquidity sweep -> break of structure -> FVG retrace (long; shorts run the same code on negated prices).
1. liquidity: last confirmed swing low (pivot of k bars, known only k bars after it prints)
2. sweep: a bar wicks below it and closes back above; sweep extreme = lowest low until the BOS
3. BOS: within max_bos bars a close above the last confirmed swing high from before the sweep
4. FVG: bullish gap low[t] > high[t-2] (>= fvg_min x ATR) inside the sweep->BOS leg, untouched since
5. entry: limit at the gap top or middle, valid `expiry` bars; stop below the sweep low; target = next
   untaken confirmed swing high (needs >= min_rr) or a fixed R multiple; time stop after `hold` bars.
Fills: limit/stop/target fill at their own prices (gap-throughs at the open); a bar that touches both
stop and target, or fills and stops on the same bar, counts as the stop. Costs: COST per side + funding.
Usage: python smc.py [SYMBOL ...]   (default BTCUSDT ETHUSDT SOLUSDT, pooled)"""
import itertools, sys
import numpy as np, pandas as pd
from strategies import atr
from patterns import _pivots

COST = 0.0005 + 0.0002          # same as backtest.COST: taker + slippage per side (limit entries charged as taker too)
TRAIN = ("2024-01-01", "2025-12-31 23:00")
TEST = ("2026-01-01", "2099")
FIXED = dict(expiry=24, hold=72, min_rr=1.5, buf=0.1, fvg_min=0.25)


def _flip(df):
    """Mirror prices so the long logic trades shorts: swing highs become swing lows and so on."""
    return pd.DataFrame(dict(open=-df.open, high=-df.low, low=-df.high, close=-df.close, funding=-df.funding,
                             volume=df.get("volume", 0.0)), df.index)


def find_setups(df, k, max_bos, entry, tp, liq="last", fresh=True, ok=None, fvg_min=0.25, buf=0.1, min_rr=1.5, expiry=24, **_):
    """Long setups only. Returns dicts with the bar the order goes live (after the BOS close) and its levels.
    liq: "last" = only the latest swing low counts as liquidity; "any" = any of the last 3 not yet taken.
    fresh: the gap must be untouched between its candle and the BOS. ok: optional bool per bar (trend filter)."""
    O, H, L, C = (df[c].values for c in ("open", "high", "low", "close"))
    a = atr(df, 14).values
    ph, pl = _pivots(df, k)
    highs, lows, out = [], [], []
    swept = set()                      # swing lows already used
    sweep = None                       # (bar, swept level, BOS level)
    for i in range(len(C)):
        if ph[i]: highs.append((i - k, H[i - k]))
        if pl[i]: lows.append((i - k, L[i - k]))
        if sweep:
            s, lvl, bos = sweep
            if C[i] < lvl or i - s > max_bos:   # closed back below the swept low or no BOS in time
                sweep = None
            elif C[i] > bos:                    # BOS: look for the gap that the displacement left
                sweep = None
                lo = L[s:i + 1].min()
                for t in range(i, s + 1, -1):   # most recent gap first
                    bot, top = H[t - 2], L[t]
                    if top - bot < fvg_min * a[i] or (fresh and t < i and L[t + 1:i + 1].min() <= top):
                        continue
                    px = top if entry == "top" else (top + bot) / 2
                    stop = lo - buf * a[i]
                    risk = px - stop
                    if tp == "liq":           # nearest confirmed swing high above, not taken since it formed
                        tgts = [h for j, h in highs[-50:] if h > C[i] and H[j + 1:i + 1].max(initial=-np.inf) < h]
                        tgts = [h for h in sorted(tgts) if (h - px) / risk >= min_rr]
                        if not tgts:
                            break
                        tgt = tgts[0]
                    else:
                        tgt = px + float(tp) * risk
                    if ok is None or ok[i]:   # features describe the setup at the BOS close (bar i), for filters
                        out.append(dict(bar=i + 1, entry=px, stop=stop, tgt=tgt, expiry=expiry, sweep_bar=s,
                                        depth=(lvl - lo) / a[i], gap=(top - bot) / a[i], body=abs(C[i] - O[i]) / a[i]))
                    break
                continue
        for j, lvl in lows[-3:]:              # a low closed through is spent, not liquidity any more
            if C[i] < lvl: swept.add(j)
        if not sweep and lows:
            pool = [(j, lvl) for j, lvl in (lows[-1:] if liq == "last" else lows[-3:]) if j not in swept and L[i] < lvl < C[i]]
            if pool:
                j, lvl = min(pool, key=lambda p: p[1])          # deepest pool taken
                prior = [h for jj, h in highs[-50:] if jj < j]  # BOS level: last swing high before that low
                swept.update(jj for jj, ll in lows[-3:] if L[i] < ll)
                if prior:
                    sweep = (i, lvl, prior[-1])
    return out


def simulate(df, setups, hold, cost=COST, fees=None, be=None, through=0.0):
    """One position or pending order at a time per side. Returns trade dicts with R and % net of costs.
    fees: (entry, target exit, stop/time exit) per side; default `cost` for all three.
    be: once price reaches be x risk in our favour, the stop moves to entry + 0.05% (from the next bar on).
    through: the limit only fills if price trades this fraction beyond it (queue position)."""
    O, H, L, C, F = (df[c].values for c in ("open", "high", "low", "close", "funding"))
    fe, ft, fs = fees or (cost, cost, cost)
    trades, free = [], 0
    for su in setups:
        b = su["bar"]
        if b < free or b >= len(C):
            continue
        e, st, tg = su["entry"], su["stop"], su["tgt"]
        kind = su.get("kind", "limit")       # limit: buy a dip to e; stop: buy a break of e; market: next open
        lim = e - through * abs(e)
        fill = None
        for i in range(b, min(b + su["expiry"], len(C))):
            if kind == "market":
                fill = (i, O[i])
                break
            if kind == "stop":
                if L[i] <= st:                # invalidated before it triggered
                    break
                if H[i] >= e:
                    fill = (i, max(O[i], e))
                    break
                continue
            if H[i] >= tg and L[i] > lim:     # ran to target without us: cancel
                break
            if L[i] <= lim:
                fill = (i, min(O[i], e))
                break
        if not fill or fill[1] <= st or fill[1] >= tg:   # gapped past the stop or the target: no trade
            free = (fill[0] if fill else i) + 1
            continue
        f, px = fill
        fe_i = su.get("fee_in", fe)
        st0 = st
        why, ex, fund = "time", None, 0.0
        for i in range(f, min(f + hold, len(C))):
            if i > f: fund += F[i]
            if L[i] <= st:
                ex, why = (min(O[i], st) if i > f else st), ("stop" if st < px else "breakeven")
                break
            if H[i] >= tg and (i > f or C[i] >= e):   # fill bar: only trust the target if it closed back up
                ex, why = (max(O[i], tg) if i > f else tg), "target"
                break
            if be is not None and H[i] >= px + be * (px - st0):
                st = max(st, px + 0.0005 * abs(px))
        if ex is None:
            ex = C[i]
        dn = abs(px)
        pct = (ex - px) / dn - fe_i - (ft if why == "target" else fs) - fund
        trades.append(dict(ts=df.index[f], exit_ts=df.index[i], entry=px, exit=ex, stop=st0, tgt=tg, why=why,
                           pct=pct, R=pct / ((px - st0) / dn), risk_pct=(px - st0) / dn, setup_bar=b - 1))
        free = i + 1
    return trades


def trend_up(df, n=50):
    """4h EMA(n) of closes, using only 4h bars that have fully closed; True where the 1h close is above it."""
    c4 = df.close.resample("4h").last()
    e = c4.ewm(span=n, adjust=False).mean().shift(1)            # bar labelled 00:00 is known from 04:00 on
    return (df.close > e.reindex(df.index, method="ffill")).values


def run(df, p, cost=COST):
    out = []
    up = trend_up(df) if p.get("trend") else None
    for side, d in ((1, df), (-1, _flip(df))):
        ok = None if up is None else (up if side > 0 else ~up)
        for t in simulate(d, find_setups(d, ok=ok, **p), p["hold"], cost):
            t.update(side=side, entry=abs(t["entry"]), exit=abs(t["exit"]), stop=abs(t["stop"]), tgt=abs(t["tgt"]))
            out.append(t)
    return pd.DataFrame(out).sort_values("ts").reset_index(drop=True) if out else pd.DataFrame()


def random_baseline(df, trades, hold, n=200, seed=0, cost=COST, **sim):
    """Same side, stop % and target % as each real trade, but entered at the open of a random bar:
    does the setup's timing add anything over the stop/target geometry alone? `sim` passes exit/fee options
    (be, fees) through so the comparison uses the same exits; market entries keep `cost` as their entry fee."""
    rng = np.random.default_rng(seed)
    flipped = _flip(df)
    if "fees" in sim:
        sim = dict(sim, fees=(cost,) + tuple(sim["fees"][1:]))
    sim.pop("through", None)
    means = []
    for _ in range(n):
        rs = []
        for _, t in trades.iterrows():
            d = df if t.side > 0 else flipped
            b = int(rng.integers(20, len(d) - hold))
            e = d.open.iloc[b]
            r = simulate(d, [dict(bar=b, entry=e, stop=e - t.risk_pct * abs(e),
                                  tgt=e + abs(t.tgt - t.entry) / abs(t.entry) * abs(e), expiry=1)], hold, cost, **sim)
            rs += [r[0]["R"]] if r else []
        means.append(np.mean(rs) if rs else np.nan)
    return np.array(means)


def summary(t):
    if t.empty:
        return dict(n=0)
    w = t[t.R > 0]
    eq = (1 + 0.01 * t.R).cumprod()            # 1% of equity risked per trade, one after another
    return dict(n=len(t), win=len(w) / len(t), avgR=t.R.mean(), totR=t.R.sum(),
                pf=w.R.sum() / -t.R[t.R <= 0].sum() if (t.R <= 0).any() else np.inf,
                ret_1pct=eq.iloc[-1] - 1, maxdd_1pct=(eq / eq.cummax() - 1).min(),
                stop_med=t.risk_pct.median(), longs=int((t.side > 0).sum()), shorts=int((t.side < 0).sum()))


GRID = dict(k=[3, 5], max_bos=[12, 24], entry=["top", "mid"], tp=["liq", "2"],
            liq=["last", "any"], fresh=[True, False], trend=[False, True])
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MIN_TRAIN = 60                       # pooled train trades needed before a config can be picked


def fmt(d):
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def window(t, w):
    return t[(t.ts >= w[0]) & (t.ts <= w[1])] if len(t) else t


def run_all(dfs, p):
    parts = []
    for sym, df in dfs.items():
        t = run(df, dict(p, **FIXED))
        if len(t):
            parts.append(t.assign(sym=sym))
    return pd.concat(parts).sort_values("ts").reset_index(drop=True) if parts else pd.DataFrame()


# Picked by family on 2024-25 only: (k=3, fresh gap, fixed 2R) had the best mean train avgR of the
# k x tp x fresh families, then the best train config inside it. 2026 stayed out of the choice.
FINAL = dict(k=3, max_bos=12, entry="top", tp="2", liq="any", fresh=True, trend=False)


def final_report(dfs):
    from pathlib import Path
    t = run_all(dfs, FINAL)
    print("== FINAL:", fmt(FINAL), fmt(FIXED))
    years = [("2024", ("2024-01-01", "2024-12-31 23:00")), ("2025", ("2025-01-01", "2025-12-31 23:00")),
             ("2026 (unseen)", TEST), ("all", ("2000", "2099"))]
    for nm, w in years:
        s = window(t, w)
        print(f"  {nm:14s} {fmt(summary(s))}")
        if len(s) > 1:
            print(f"  {'':14s} t-stat={s.R.mean() / (s.R.std() / np.sqrt(len(s))):.2f}  "
                  f"avgR at 2x costs={(s.R - 2 * COST / s.risk_pct).mean():+.3f}")
    for s in dfs:
        ts = t[t.sym == s]
        print(f"  {s}: 2024-25 {fmt(summary(window(ts, TRAIN)))}\n  {' ' * len(s)}  2026    {fmt(summary(window(ts, TEST)))}")
    for nm, w in (("2024-25", TRAIN), ("2026", TEST)):
        sub = window(t, w)
        rb = np.nanmean([random_baseline(dfs[s], sub[sub.sym == s], FIXED["hold"], n=200) for s in dfs if (sub.sym == s).any()], axis=0)
        print(f"  random entries, same stops/targets ({nm}, 200 runs): mean avgR={np.nanmean(rb):+.3f} "
              f"p95={np.nanpercentile(rb, 95):+.3f}; strategy beats {np.mean(rb < sub.R.mean()):.0%}")
    print("  exits:", t.why.value_counts().to_dict(), " median hours held:",
          float((t.exit_ts - t.ts).dt.total_seconds().median() / 3600))
    out = Path("smc_results"); out.mkdir(exist_ok=True)
    t.to_csv(out / "final_trades.csv", index=False)
    print(t.tail(20)[["ts", "sym", "side", "entry", "stop", "tgt", "exit", "why", "R"]].to_string())


if __name__ == "__main__":
    import data
    from pathlib import Path
    if "--final" in sys.argv:
        sys.argv.remove("--final")
        final_report({s: data.load(s) for s in (sys.argv[1:] or SYMS)})
        sys.exit()
    syms = sys.argv[1:] or SYMS
    dfs = {s: data.load(s) for s in syms}
    for s, d in dfs.items():
        print(s, d.index[0], "->", d.index[-1], len(d), "bars")
    rows = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        t = run_all(dfs, p)
        rows.append((p, summary(window(t, TRAIN)), summary(window(t, TEST)), t))
    print(f"\n== {len(rows)} configs, pooled {'+'.join(syms)}: train 2024-25 | test 2026 (sorted by train avgR) ==")
    for p, a, b, _ in sorted(rows, key=lambda r: -r[1].get("avgR", -9))[:25]:
        print(f"{fmt(p):70s} | train n={a.get('n',0):4d} win={a.get('win',0):.2f} avgR={a.get('avgR',0):+.3f} "
              f"| test n={b.get('n',0):4d} win={b.get('win',0):.2f} avgR={b.get('avgR',0):+.3f}")
    print("\n== each rule variant, averaged over the core grid (how much does the idea help on its own?) ==")
    for key in ("liq", "fresh", "trend"):
        for v in GRID[key]:
            sel = [r for r in rows if r[0][key] == v]
            tr = np.mean([r[1].get("avgR", np.nan) for r in sel]); te = np.mean([r[2].get("avgR", np.nan) for r in sel])
            print(f"  {key}={v!s:5s}: mean train avgR={tr:+.3f}  mean test avgR={te:+.3f}  "
                  f"test positive in {np.mean([r[2].get('avgR', -1) > 0 for r in sel]):.0%} of configs")
    pos_tr = np.mean([r[1].get("avgR", -1) > 0 for r in rows]); pos_te = np.mean([r[2].get("avgR", -1) > 0 for r in rows])
    print(f"  all configs: train positive {pos_tr:.0%}, test positive {pos_te:.0%}")
    best = max((r for r in rows if r[1].get("n", 0) >= MIN_TRAIN), key=lambda r: r[1]["avgR"])
    p, a, b, t = best
    print(f"\n== PICKED ON TRAIN ONLY (n>={MIN_TRAIN}): {fmt(p)} {fmt(FIXED)}")
    print("  train:", fmt(a)); print("  test: ", fmt(b))
    for s in syms:
        ts = t[t.sym == s]
        print(f"  {s}: train {fmt(summary(window(ts, TRAIN)))}\n  {' ' * len(s)}  test  {fmt(summary(window(ts, TEST)))}")
    print("  exits train:", window(t, TRAIN).why.value_counts().to_dict(), " test:", window(t, TEST).why.value_counts().to_dict())
    for nm, w in (("train", TRAIN), ("test", TEST)):
        sub = window(t, w)
        rb = np.nanmean([random_baseline(dfs[s], sub[sub.sym == s], FIXED["hold"], n=100) for s in syms if (sub.sym == s).any()], axis=0)
        print(f"  random-entry baseline ({nm}, same stops/targets, 100 runs): mean avgR={np.nanmean(rb):+.3f} "
              f"p95={np.nanpercentile(rb, 95):+.3f}; strategy beats {np.mean(rb < sub.R.mean()):.0%} of runs")
    out = Path("smc_results"); out.mkdir(exist_ok=True)
    t.assign(**{k: str(v) for k, v in p.items()}).to_csv(out / "picked_trades.csv", index=False)
    pd.DataFrame([dict(**r[0], **{f"train_{k}": v for k, v in r[1].items()}, **{f"test_{k}": v for k, v in r[2].items()})
                  for r in rows]).to_csv(out / "grid.csv", index=False)
