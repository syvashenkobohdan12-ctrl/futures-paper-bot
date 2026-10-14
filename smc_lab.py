"""Make-it-work lab for the sweep -> BOS -> FVG setup, with a fixed protocol so the answer isn't fitted to noise.
  DEV   : BTC ETH SOL XRP BNB DOGE, 1h, 2022-01 .. 2025-12   <- every choice is made here, nowhere else
  HOLD-A: ADA AVAX LINK SUI over the same years              <- coins never used for a choice
  HOLD-B: all 10 coins, 2026                                 <- months never used for a choice
Steps (all scored on DEV by t-stat of R, so a filter has to keep enough trades to be worth it):
  1. execution: taker everywhere vs maker limit entries/targets (fill only on a 0.02% trade-through)
  2. exits: target 1.5/2/3R x breakeven off/at 1R
  3. greedy filters: add the one that lifts DEV t most, stop when nothing adds >= 0.25
Holdouts are printed for every step for transparency, but they never drive a choice.
Usage: python smc_lab.py"""
import itertools, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import smc, smc_data
from strategies import adx

DEV = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
HOLD = ["ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
DEV_END = "2026-01-01"
TAKER = 0.0005 + 0.0002                   # fee + slippage per side
MAKER = 0.0002                            # Binance USD-M maker; limit orders don't slip
EXEC = {"taker": dict(fees=(TAKER, TAKER, TAKER), through=0.0),
        "maker": dict(fees=(MAKER, MAKER, TAKER), through=0.0002)}
BASES = {"bos12": dict(smc.FINAL), "bos24": dict(smc.FINAL, max_bos=24)}
OUT = Path("smc_results")


def features(df):
    """Per-bar context in the ORIGINAL orientation, all known at that bar's close."""
    c = df.close
    ema4 = c.resample("4h").last().ewm(span=50, adjust=False).mean().shift(1).reindex(df.index, method="ffill")
    emad = c.resample("1D").last().ewm(span=50, adjust=False).mean().shift(1).reindex(df.index, method="ffill")
    a = smc.atr(df, 14)
    volr = df.volume / df.volume.rolling(20).mean().shift(1)
    vpct = (a / c).rolling(720, min_periods=200).rank(pct=True)
    return pd.DataFrame(dict(up4=(c > ema4).values, upd=(c > emad).values, adx=adx(df, 14).values,
                             vpct=vpct.values, volr=volr.values, hour=df.index.hour), df.index)


# filter name -> fn(setup, features row arrays, side) -> bool. Thresholds fixed up front, not tuned.
FILTERS = {
    "trend_4h":   lambda s, F, d: F["up4"][s["i"]] == (d > 0),
    "trend_1d":   lambda s, F, d: F["upd"][s["i"]] == (d > 0),
    "killzone":   lambda s, F, d: F["hour"][s["i"]] in (7, 8, 9, 10, 12, 13, 14, 15, 16),
    "displace":   lambda s, F, d: s["body"] >= 1.0,
    "big_gap":    lambda s, F, d: s["gap"] >= 0.5,
    "deep_sweep": lambda s, F, d: s["depth"] >= 0.3,
    "vol_spike":  lambda s, F, d: F["volr"][s["sweep_bar"]] >= 1.5,
    "adx_trend":  lambda s, F, d: F["adx"][s["i"]] >= 20,
    "adx_range":  lambda s, F, d: F["adx"][s["i"]] < 20,
    "hi_vol":     lambda s, F, d: F["vpct"][s["i"]] >= 0.5,
    "lo_vol":     lambda s, F, d: F["vpct"][s["i"]] < 0.5,
}


def prep(sym):
    df, _ = smc_data.load(sym, "1h")
    F = {k: v.values for k, v in features(df).items()}
    out = {}
    for name, p in BASES.items():
        sides = []
        for d, x in ((1, df), (-1, smc._flip(df))):
            su = smc.find_setups(x, **dict(p, **smc.FIXED))
            for s in su:
                s["i"] = s["bar"] - 1
            sides.append((d, x, su))
        out[name] = sides
    return sym, df, F, out


def evaluate(data, base, ex, tp, be, filt):
    rows = []
    for sym, (df, F, bases) in data.items():
        for d, x, su in bases[base]:
            keep = []
            for s in su:
                if all(FILTERS[f](s, F, d) for f in filt):
                    risk = s["entry"] - s["stop"]
                    keep.append(dict(s, tgt=s["entry"] + tp * risk))
            for t in smc.simulate(x, keep, smc.FIXED["hold"], be=be, **EXEC[ex]):
                t.update(sym=sym, side=d, entry=abs(t["entry"]), exit=abs(t["exit"]), stop=abs(t["stop"]), tgt=abs(t["tgt"]))
                rows.append(t)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["set"] = np.where(t.ts >= DEV_END, "hold_B", np.where(t.sym.isin(DEV), "dev", "hold_A"))
    return t


def st(r):
    r = np.asarray(r)
    if len(r) < 2:
        return dict(n=len(r), avgR=float(r.mean()) if len(r) else np.nan, t=np.nan)
    return dict(n=len(r), avgR=r.mean(), t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def line(label, t):
    parts = []
    for k in ("dev", "hold_A", "hold_B"):
        s = st(t.R[t.set == k]) if len(t) else st([])
        parts.append(f"{k} n={s['n']:4d} avgR={s['avgR']:+.3f} t={s['t']:+.2f}")
    return f"{label:58s} | " + " | ".join(parts)


def dev_t(t, min_n=150):
    r = t.R[t.set == "dev"] if len(t) else []
    s = st(r)
    return s["t"] if s["n"] >= min_n else -np.inf


def protocol(data, say=print):
    """Steps 1-4 on DEV only. Returns the chosen config and its trades over all sets."""
    say("\n== 1. execution (base rules, 2R, no breakeven) ==")
    best = None
    for base, ex in itertools.product(BASES, EXEC):
        t = evaluate(data, base, ex, 2.0, None, ())
        say(line(f"{base} {ex}", t))
        if best is None or dev_t(t) > best[0]:
            best = (dev_t(t), base, ex)
    _, base, ex = best
    say(f"-> picked on DEV: base={base} exec={ex}")

    say("\n== 2. exits ==")
    best = None
    for tp, be in itertools.product([1.5, 2.0, 3.0], [None, 1.0]):
        t = evaluate(data, base, ex, tp, be, ())
        say(line(f"tp={tp} be={be}", t))
        if best is None or dev_t(t) > best[0]:
            best = (dev_t(t), tp, be)
    _, tp, be = best
    say(f"-> picked on DEV: tp={tp} be={be}")

    say("\n== 3. filters, one at a time on top of the above ==")
    cur = evaluate(data, base, ex, tp, be, ())
    say(line("no filter", cur))
    for f in FILTERS:
        say(line(f"+ {f}", evaluate(data, base, ex, tp, be, (f,))))

    say("\n== 3b. greedy (DEV t-stat, >= 150 DEV trades, stop when gain < 0.25) ==")
    chosen, cur_t = [], dev_t(cur)
    while True:
        cands = [(dev_t(evaluate(data, base, ex, tp, be, tuple(chosen + [f]))), f) for f in FILTERS if f not in chosen
                 and not (f.endswith("_range") and "adx_trend" in chosen) and not (f == "adx_trend" and "adx_range" in chosen)
                 and not (f == "lo_vol" and "hi_vol" in chosen) and not (f == "hi_vol" and "lo_vol" in chosen)]
        if not cands:
            break
        bt, bf = max(cands)
        if bt - cur_t < 0.25:
            break
        chosen.append(bf); cur_t = bt
        say(line(f"step {len(chosen)}: + {bf} (dev t {bt:+.2f})", evaluate(data, base, ex, tp, be, tuple(chosen))))
    say(f"-> filters picked on DEV: {chosen or 'none'}")

    say("\n== 4. re-check exits with the filters ==")
    best = None
    for tp2, be2 in itertools.product([1.5, 2.0, 3.0], [None, 1.0]):
        t = evaluate(data, base, ex, tp2, be2, tuple(chosen))
        say(line(f"tp={tp2} be={be2}", t))
        if dev_t(t, 100) > (best[0] if best else -np.inf):
            best = (dev_t(t, 100), tp2, be2)
    _, tp, be = best

    final = evaluate(data, base, ex, tp, be, tuple(chosen))
    cfg = dict(base=base, exec=ex, tp=tp, be=be, filters=chosen)
    return cfg, final


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(4) as pool:
        data = {sym: (df, F, b) for sym, df, F, b in pool.map(prep, DEV + HOLD)}
    print("loaded", {s: len(v[0]) for s, v in data.items()})

    cfg, final = protocol(data)
    base, ex, tp, be, chosen = cfg["base"], cfg["exec"], cfg["tp"], cfg["be"], cfg["filters"]
    print(f"\n== FINAL (chosen on DEV only): {cfg}")
    print(line("final", final))
    for k in ("dev", "hold_A", "hold_B"):
        sub = final[final.set == k]
        print(f"  {k}: win={np.mean(sub.R > 0):.2f} exits={sub.why.value_counts().to_dict()} longs={int((sub.side > 0).sum())} shorts={int((sub.side < 0).sum())}")
    yr = final.groupby([final.ts.dt.year, "set"]).R.agg(["count", "mean"]).round(3)
    print("  by year:\n" + yr.to_string())
    print("  by coin:\n" + final.groupby("sym").R.agg(["count", "mean"]).round(3).to_string())
    for k in ("hold_A", "hold_B"):
        sub = final[final.set == k]
        runs = []
        for sym in sub.sym.unique():
            ts = sub[sub.sym == sym]
            df = data[sym][0]
            rb = smc.random_baseline(df, ts, smc.FIXED["hold"], n=100, cost=TAKER, be=be, **EXEC[ex])
            runs.append(rb * len(ts))
        rb = np.nansum(runs, axis=0) / len(sub)
        print(f"  random entries, same stops/targets ({k}): mean avgR={np.nanmean(rb):+.3f} p95={np.nanpercentile(rb, 95):+.3f} "
              f"-> strategy beats {np.mean(rb < sub.R.mean()):.0%}")
    eq = (1 + 0.01 * final.sort_values("exit_ts").R).cumprod()
    print(f"  all sets, 1% risk/trade: return={eq.iloc[-1] - 1:+.1%} maxDD={(eq / eq.cummax() - 1).min():.1%} trades/yr="
          f"{len(final) / ((final.ts.max() - final.ts.min()).days / 365):.0f}")
    final.to_csv(OUT / "lab_trades.csv", index=False)
    pd.Series(cfg, dtype=object).to_json(OUT / "lab_config.json")
