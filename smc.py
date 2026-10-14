"""Liquidity sweep -> break of structure -> FVG retrace (long; shorts run the same code on negated prices).
1. liquidity: last confirmed swing low (pivot of k bars, known only k bars after it prints)
2. sweep: a bar wicks below it and closes back above; sweep extreme = lowest low until the BOS
3. BOS: within max_bos bars a close above the last confirmed swing high from before the sweep
4. FVG: bullish gap low[t] > high[t-2] (>= fvg_min x ATR) inside the sweep->BOS leg, untouched since
5. entry: limit at the gap top or middle, valid `expiry` bars; stop below the sweep low; target = next
   untaken confirmed swing high (needs >= min_rr) or a fixed R multiple; time stop after `hold` bars.
Fills: limit/stop/target fill at their own prices (gap-throughs at the open); a bar that touches both
stop and target, or fills and stops on the same bar, counts as the stop. Costs: COST per side + funding.
Usage: python smc.py [SYMBOL]   (default ETHUSDT)"""
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
    return pd.DataFrame(dict(open=-df.open, high=-df.low, low=-df.high, close=-df.close, funding=-df.funding), df.index)


def find_setups(df, k, max_bos, entry, tp, fvg_min=0.25, buf=0.1, min_rr=1.5, expiry=24, **_):
    """Long setups only. Returns dicts with the bar the order goes live (after the BOS close) and its levels."""
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
                    if top - bot < fvg_min * a[i] or (t < i and L[t + 1:i + 1].min() <= top):
                        continue
                    px = top if entry == "top" else (top + bot) / 2
                    stop = lo - buf * a[i]
                    risk = px - stop
                    if tp == "liq":           # nearest confirmed swing high above, not taken since it formed
                        tgts = [h for j, h in highs if h > C[i] and H[j + 1:i + 1].max(initial=-np.inf) < h]
                        tgts = [h for h in sorted(tgts) if (h - px) / risk >= min_rr]
                        if not tgts:
                            break
                        tgt = tgts[0]
                    else:
                        tgt = px + float(tp) * risk
                    out.append(dict(bar=i + 1, entry=px, stop=stop, tgt=tgt, expiry=expiry))
                    break
                continue
        if not sweep and lows:
            j, lvl = lows[-1]
            prior = [h for jj, h in highs if jj < j]   # BOS level: last confirmed swing high before the liquidity low
            if j not in swept and prior and L[i] < lvl < C[i]:
                swept.add(j)
                sweep = (i, lvl, prior[-1])
    return out


def simulate(df, setups, hold, cost=COST):
    """One position or pending order at a time per side. Returns trade dicts with R and % net of costs."""
    O, H, L, C, F = (df[c].values for c in ("open", "high", "low", "close", "funding"))
    trades, free = [], 0
    for su in setups:
        b = su["bar"]
        if b < free or b >= len(C):
            continue
        e, st, tg = su["entry"], su["stop"], su["tgt"]
        fill = None
        for i in range(b, min(b + su["expiry"], len(C))):
            if H[i] >= tg and L[i] > e:       # ran to target without us: cancel
                break
            if L[i] <= e:
                fill = (i, min(O[i], e))
                break
        if not fill:
            free = i + 1
            continue
        f, px = fill
        why, ex, fund = "time", None, 0.0
        for i in range(f, min(f + hold, len(C))):
            if i > f: fund += F[i]
            if L[i] <= st:
                ex, why = (min(O[i], st) if i > f else st), "stop"
                break
            if H[i] >= tg and (i > f or C[i] >= e):   # fill bar: only trust the target if it closed back up
                ex, why = (max(O[i], tg) if i > f else tg), "target"
                break
        if ex is None:
            ex = C[i]
        dn = abs(px)
        pct = (ex - px) / dn - 2 * cost - fund
        trades.append(dict(ts=df.index[f], exit_ts=df.index[i], entry=px, exit=ex, stop=st, tgt=tg, why=why,
                           pct=pct, R=pct / ((px - st) / dn), risk_pct=(px - st) / dn))
        free = i + 1
    return trades


def run(df, p):
    out = []
    for side, d in ((1, df), (-1, _flip(df))):
        for t in simulate(d, find_setups(d, **p), p["hold"]):
            t.update(side=side, entry=abs(t["entry"]), exit=abs(t["exit"]), stop=abs(t["stop"]), tgt=abs(t["tgt"]))
            out.append(t)
    return pd.DataFrame(out).sort_values("ts").reset_index(drop=True) if out else pd.DataFrame()


def random_baseline(df, trades, hold, n=200, seed=0):
    """Same side, stop % and target % as each real trade, but entered at the open of a random bar:
    does the setup's timing add anything over the stop/target geometry alone?"""
    rng = np.random.default_rng(seed)
    flipped = _flip(df)
    means = []
    for _ in range(n):
        rs = []
        for _, t in trades.iterrows():
            d = df if t.side > 0 else flipped
            b = int(rng.integers(20, len(d) - hold))
            e = d.open.iloc[b]
            r = simulate(d, [dict(bar=b, entry=e, stop=e - t.risk_pct * abs(e),
                                  tgt=e + abs(t.tgt - t.entry) / t.entry * abs(e), expiry=1)], hold)
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


GRID = dict(k=[3, 5, 8], max_bos=[12, 24], entry=["top", "mid"], tp=["liq", "2", "3"])
DEFAULT = dict(k=5, max_bos=24, entry="mid", tp="liq")   # chosen before looking at results


def fmt(d):
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


if __name__ == "__main__":
    import data
    sym = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    df = data.load(sym)
    print(sym, df.index[0], "->", df.index[-1], len(df), "bars; train", TRAIN, "test", TEST)
    rows = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals), **FIXED)
        t = run(df, p)
        tr = t[(t.ts >= TRAIN[0]) & (t.ts <= TRAIN[1])] if len(t) else t
        te = t[t.ts >= TEST[0]] if len(t) else t
        rows.append((dict(zip(GRID, vals)), summary(tr), summary(te), t))
    print("\n== grid (train 2024-25 | test 2026) ==")
    for p, a, b, _ in sorted(rows, key=lambda r: -r[1].get("avgR", -9)):
        print(f"{fmt(p):38s} | train n={a.get('n',0):4d} win={a.get('win',0):.2f} avgR={a.get('avgR',0):+.3f} "
              f"| test n={b.get('n',0):4d} win={b.get('win',0):.2f} avgR={b.get('avgR',0):+.3f}")
    best = max((r for r in rows if r[1].get("n", 0) >= 30), key=lambda r: r[1]["avgR"])
    for label, r in (("DEFAULT (pre-chosen)", next(r for r in rows if r[0] == DEFAULT)), ("BEST ON TRAIN", best)):
        p, a, b, t = r
        print(f"\n== {label}: {fmt(p)} {fmt(FIXED)}")
        print("  train:", fmt(a))
        print("  test: ", fmt(b))
        print("  all:  ", fmt(summary(t)))
        print("  exits:", t.why.value_counts().to_dict())
        for nm, w in (("train", TRAIN), ("test", (TEST[0], "2099"))):
            s = t[(t.ts >= w[0]) & (t.ts <= w[1])]
            if len(s):
                rb = random_baseline(df, s, FIXED["hold"])
                print(f"  random-entry baseline ({nm}, same stops/targets, 200 runs): mean avgR={np.nanmean(rb):+.3f} "
                      f"p95={np.nanpercentile(rb, 95):+.3f}; strategy beats {np.mean(rb < s.R.mean()):.0%} of runs")
    print("\n== last 15 trades (default) ==")
    t = next(r for r in rows if r[0] == DEFAULT)[3]
    print(t.tail(15)[["ts", "side", "entry", "stop", "tgt", "exit", "why", "R"]].to_string())
