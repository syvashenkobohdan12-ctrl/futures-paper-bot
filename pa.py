"""Classic price-action setups through the same honest protocol as smc_lab.py. No indicators: ATR only sizes stops.
Every detector is written for longs; shorts run the same code on mirrored prices (smc._flip).
  pd_sweep   1h: trades below the previous UTC day's low, closes back above it (first time that day) -> market;
                 stop under the day's low so far
  inside     a bar inside the previous bar -> stop order above the mother bar's high; stop under its low
  pin        lower wick >= 60% of the range, close in the top third, low testing the last swing low -> stop order
             above the pin high; stop under the pin low
  engulf     green body engulfs the previous red body, low testing the last swing low -> market; stop under the pair
  retest     close breaks the last swing high -> limit back at that level; stop 1 ATR under it
  structure  higher high + higher low, then a close above the last swing high -> market; stop under the last swing low
  donchian   close above the prior n-bar high -> market; stop 2 ATR
Every stop gets a 0.1 ATR buffer and sits at least 0.5 ATR from entry (tighter stops just pay fees). Target = tp x risk. Time stop 72 bars. Orders live 3 bars (retest 12).
Protocol: DEV = BTC ETH SOL XRP BNB DOGE until 2025-12; HOLD_A = ADA AVAX LINK SUI same years; HOLD_B = all, 2026.
For each setup the (tf, params) with the best DEV t-stat is picked; holdouts are only reported.
Usage: python pa.py            real data (needs data.binance.vision)
       python pa.py --null N   the same search on N random-walk universes (offline) -> what luck alone finds"""
import itertools, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import smc, smc_data
from patterns import _pivots

DEV = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
HOLD = ["ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
DEV_END = "2026-01-01"
START = {"1h": "2022-01-01", "4h": "2022-01-01", "1d": "2020-01-01"}
TAKER, MAKER = 0.0005 + 0.0002, 0.0002
HOLD_BARS = 72
MIN_DEV = {"1h": 150, "4h": 60, "1d": 30}
TPS = [1.5, 2.0, 3.0]
OUT = Path("smc_results")


def _floor(e, st, a):
    return min(st, e - 0.5 * a)


def _swing_lows(df, k):
    """Last confirmed swing low level known at each bar (NaN before the first)."""
    _, pl = _pivots(df, k)
    lv = np.full(len(df), np.nan)
    cur = np.nan
    L = df.low.values
    for i in range(len(df)):
        if pl[i]: cur = L[i - k]
        lv[i] = cur
    return lv


def det_pd_sweep(df):
    O, H, L, C = (df[c].values for c in ("open", "high", "low", "close"))
    a = smc.atr(df, 14).values
    day = df.index.normalize()
    dlow = df.low.groupby(day).min()
    pdl = pd.Series(dlow.shift(1).reindex(day).values, df.index).values
    out, fired, lo, d0 = [], set(), np.inf, None
    for i in range(1, len(C)):
        if day[i] != d0:
            d0, lo = day[i], np.inf
        lo = min(lo, L[i])
        if d0 in fired or not np.isfinite(pdl[i]):
            continue
        if L[i] < pdl[i] < C[i]:
            fired.add(d0)
            st = _floor(C[i], lo - 0.1 * a[i], a[i])
            out.append(dict(bar=i + 1, kind="market", entry=C[i], stop=st, risk=C[i] - st, expiry=1))
    return out


def det_inside(df):
    H, L = df.high.values, df.low.values
    a = smc.atr(df, 14).values
    out = []
    for i in range(2, len(H)):
        if H[i] < H[i - 1] and L[i] > L[i - 1]:
            e = H[i - 1]; st = _floor(e, L[i - 1] - 0.1 * a[i], a[i])
            out.append(dict(bar=i + 1, kind="stop", entry=e, stop=st, risk=e - st, expiry=3))
    return out


def det_pin(df, k):
    O, H, L, C = (df[c].values for c in ("open", "high", "low", "close"))
    a = smc.atr(df, 14).values
    lv = _swing_lows(df, k)
    out = []
    for i in range(1, len(C)):
        r = H[i] - L[i]
        if r <= 0 or np.isnan(lv[i]):
            continue
        if (min(O[i], C[i]) - L[i] >= 0.6 * r and C[i] >= L[i] + 0.66 * r
                and lv[i] - a[i] <= L[i] <= lv[i] + 0.25 * a[i]):
            e = H[i]; st = _floor(e, L[i] - 0.1 * a[i], a[i])
            out.append(dict(bar=i + 1, kind="stop", entry=e, stop=st, risk=e - st, expiry=3))
    return out


def det_engulf(df, k):
    O, H, L, C = (df[c].values for c in ("open", "high", "low", "close"))
    a = smc.atr(df, 14).values
    lv = _swing_lows(df, k)
    out = []
    for i in range(1, len(C)):
        if np.isnan(lv[i]):
            continue
        lo = min(L[i], L[i - 1])
        if (C[i - 1] < O[i - 1] and C[i] > O[i] and O[i] <= C[i - 1] and C[i] >= O[i - 1]
                and lv[i] - a[i] <= lo <= lv[i] + 0.25 * a[i]):
            st = _floor(C[i], lo - 0.1 * a[i], a[i])
            out.append(dict(bar=i + 1, kind="market", entry=C[i], stop=st, risk=C[i] - st, expiry=1))
    return out


def det_retest(df, k):
    C = df.close.values
    H = df.high.values
    a = smc.atr(df, 14).values
    ph, _ = _pivots(df, k)
    out, lvl = [], np.nan
    for i in range(1, len(C)):
        if ph[i]: lvl = H[i - k]
        if np.isfinite(lvl) and C[i] > lvl >= C[i - 1]:
            st = lvl - 1.1 * a[i]
            out.append(dict(bar=i + 1, kind="limit", entry=lvl, stop=st, risk=lvl - st, expiry=12))
            lvl = np.nan                      # one setup per level
    return out


def det_structure(df, k):
    H, L, C = df.high.values, df.low.values, df.close.values
    a = smc.atr(df, 14).values
    ph, pl = _pivots(df, k)
    hs, ls, out, armed = [], [], [], True
    for i in range(1, len(C)):
        if ph[i]: hs.append(H[i - k]); armed = True
        if pl[i]: ls.append(L[i - k])
        if len(hs) >= 2 and len(ls) >= 2 and armed and hs[-1] > hs[-2] and ls[-1] > ls[-2] and C[i] > hs[-1] >= C[i - 1]:
            st = _floor(C[i], ls[-1] - 0.1 * a[i], a[i])
            out.append(dict(bar=i + 1, kind="market", entry=C[i], stop=st, risk=C[i] - st, expiry=1))
            armed = False
    return out


def det_donchian(df, n):
    H, C = df.high.values, df.close.values
    a = smc.atr(df, 14).values
    hi = df.high.rolling(n).max().shift(1).values
    out = []
    for i in range(n + 2, len(C)):
        if C[i] > hi[i] and C[i - 1] <= hi[i - 1]:
            st = C[i] - 2 * a[i]
            out.append(dict(bar=i + 1, kind="market", entry=C[i], stop=st, risk=C[i] - st, expiry=1))
    return out


# name -> (detector, {param: values}, timeframes)
SETUPS = {
    "pd_sweep":  (det_pd_sweep,  {},             ["1h"]),
    "inside":    (det_inside,    {},             ["1h", "4h", "1d"]),
    "pin":       (det_pin,       {"k": [3, 5]},  ["1h", "4h", "1d"]),
    "engulf":    (det_engulf,    {"k": [3, 5]},  ["1h", "4h", "1d"]),
    "retest":    (det_retest,    {"k": [3, 5]},  ["1h", "4h", "1d"]),
    "structure": (det_structure, {"k": [3, 5]},  ["1h", "4h", "1d"]),
    "donchian":  (det_donchian,  {"n": [20, 55]}, ["1h", "4h", "1d"]),
}


def fees_for(kind):
    entry = MAKER if kind == "limit" else TAKER
    return entry


def load_real(sym):
    out = {}
    for tf in ("1h", "4h", "1d"):
        df = smc_data.crypto(sym, tf)
        out[tf] = df[df.index >= START[tf]]
    return out


def synth(seed, start="2020-01-01", end="2026-10-14"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(idx)
    vol = 0.007 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol)); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 1, n)) * vol * .6)
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 1, n)) * vol * .6)
    df = pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=1.0,
                           funding=np.where(idx.hour % 8 == 0, 0.0001, 0.0)), idx)
    out = {"1h": df[df.index >= START["1h"]]}
    for tf in ("4h", "1d"):
        r = df.resample(tf).agg(smc_data.AGG).dropna()
        out[tf] = r[r.index >= START[tf]]
    return out


def setups_for(frames):
    """All raw setups for one coin: {(setup, tf, params): [(side, mirrored df, setups)]}."""
    res = {}
    flips = {tf: smc._flip(df) for tf, df in frames.items()}
    for name, (det, grid, tfs) in SETUPS.items():
        for tf in tfs:
            for vals in itertools.product(*grid.values()) if grid else [()]:
                p = dict(zip(grid, vals))
                res[(name, tf, tuple(p.items()))] = [(d, x, det(x, **p)) for d, x in ((1, frames[tf]), (-1, flips[tf]))]
    return res


def trades_for(sym, sides, tp):
    rows = []
    for d, x, su in sides:
        orders = [dict(s, tgt=s["entry"] + tp * s["risk"], fee_in=fees_for(s["kind"])) for s in su if s["risk"] > 0]
        for t in smc.simulate(x, orders, HOLD_BARS, fees=(TAKER, MAKER, TAKER), through=0.0002):
            t.update(sym=sym, side=d)
            rows.append(t)
    return rows


def tstat(r):
    r = np.asarray(r, float)
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else np.nan


def search(universe, say=print):
    """universe: {sym: {key: sides}}. Per setup: pick (tf, params, tp) on DEV, report holdouts."""
    keys = next(iter(universe.values())).keys()
    picks = []
    for name in SETUPS:
        best = None
        for key in [k for k in keys if k[0] == name]:
            for tp in TPS:
                rows = [t for sym, u in universe.items() for t in trades_for(sym, u[key], tp)]
                t = pd.DataFrame(rows)
                if t.empty:
                    continue
                t["set"] = np.where(t.ts >= DEV_END, "hold_B", np.where(t.sym.isin(DEV), "dev", "hold_A"))
                dev = t.R[t.set == "dev"]
                score = tstat(dev) if len(dev) >= MIN_DEV[key[1]] else -np.inf
                if best is None or score > best[0]:
                    best = (score, key, tp, t)
        score, key, tp, t = best
        r = {s: t.R[t.set == s] for s in ("dev", "hold_A", "hold_B")}
        hold = t.R[t.set != "dev"]
        row = dict(setup=name, tf=key[1], params=dict(key[2]), tp=tp,
                   dev_n=len(r["dev"]), dev_R=r["dev"].mean(), dev_t=score,
                   A_n=len(r["hold_A"]), A_R=r["hold_A"].mean(), A_t=tstat(r["hold_A"]),
                   B_n=len(r["hold_B"]), B_R=r["hold_B"].mean(), B_t=tstat(r["hold_B"]),
                   hold_R=hold.mean(), hold_t=tstat(hold))
        picks.append((row, t))
        say(f"{name:10s} pick tf={key[1]} {dict(key[2])} tp={tp} | dev n={row['dev_n']:4d} R={row['dev_R']:+.3f} t={score:+.2f} "
            f"| A n={row['A_n']:4d} R={row['A_R']:+.3f} | B n={row['B_n']:4d} R={row['B_R']:+.3f} | holdouts R={row['hold_R']:+.3f} t={row['hold_t']:+.2f}")
    return picks


def _prep_real(sym):
    return sym, setups_for(load_real(sym))


def _null_universe(u):
    uni = {sym: setups_for(synth(u * 100 + j)) for j, sym in enumerate(DEV + HOLD)}
    picks = search(uni, say=lambda *a: None)
    return [dict(p[0], u=u) for p in picks]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    if "--null" in sys.argv:
        n = int(sys.argv[sys.argv.index("--null") + 1])
        with ProcessPoolExecutor(4) as ex:
            rows = [r for rs in ex.map(_null_universe, range(n)) for r in rs]
        nul = pd.DataFrame(rows)
        nul.to_csv(OUT / "pa_null.csv", index=False)
        print(f"== null: {n} random-walk universes, same search ==")
        print(nul.groupby("setup")[["dev_t", "hold_R", "hold_t"]].quantile(0.95).round(3).rename(columns=lambda c: c + "_p95"))
        best = nul.groupby("u").hold_t.max()
        print(f"best-of-7 holdout t across setups: median {best.median():.2f}, p95 {best.quantile(.95):.2f}")
        sys.exit()
    with ProcessPoolExecutor(4) as ex:
        uni = dict(ex.map(_prep_real, DEV + HOLD))
    print("== real data: pick per setup on DEV, holdouts reported ==")
    picks = search(uni)
    summ = pd.DataFrame([p[0] for p in picks])
    summ.to_csv(OUT / "pa_summary.csv", index=False)
    print("\n" + summ.round(3).to_string(index=False))
    for row, t in picks:
        t.assign(setup=row["setup"], tf=row["tf"]).to_csv(OUT / f"pa_trades_{row['setup']}.csv", index=False)
        yr = t.groupby(t.ts.dt.year).R.agg(["count", "mean"]).round(3)
        print(f"\n{row['setup']} by year: " + " ".join(f"{y}:{int(c)}/{m:+.2f}" for y, (c, m) in yr.iterrows()))
