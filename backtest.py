"""Walk-forward research: tune every strategy on 2026 H1 (train), judge it on 2026-07-01..now (test, unseen).
Costs: taker fee + slippage per side on every position change, plus perp funding while held.
One param set per strategy across all symbols (less overfit than per-symbol tuning)."""
import itertools, sys
import numpy as np, pandas as pd
import data
from strategies import STRATS, buy_hold
import patterns, waves  # noqa: F401  register chart-pattern and Elliott-wave strategies

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
COST = 0.0005 + 0.0002          # 0.05% taker + 0.02% slippage per side
TRAIN = ("2026-01-01", "2026-06-30 23:00")
TEST = ("2026-07-01", "2099")
FULL = ("2026-01-01", "2099")
Y2025 = ("2025-01-01", "2025-12-31 23:00")  # never used for tuning: a second out-of-sample year
VOL_TARGET, MAX_LEV = 0.5, 2.0  # annualised vol target for the "vt" polish
TF = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("1h", "4h", "1d") else "1h"   # 1h | 4h | 1d; grids are in bars, so they span longer time on higher TFs
TF_HOURS = {"1h": 1, "4h": 4, "1d": 24}[TF]
HOURS = 24 * 365 // TF_HOURS                      # bars per year
MIN_TRADES = 3 if TF == "1d" else 6               # daily H1 has only ~180 bars


def apply_polish(df, pos, side, vt, tf_hours=None):
    tf_hours = tf_hours or TF_HOURS
    if side == "long":
        pos = pos.clip(lower=0)
    if vt:
        rv = df.close.pct_change().rolling(max(168 // tf_hours, 20)).std() * np.sqrt(24 * 365 // tf_hours)
        pos = pos * ((VOL_TARGET / rv).clip(upper=MAX_LEV) * 4).round().div(4).fillna(0)  # 0.25 steps limit churn
    return pos


def pnl(df, pos, cost=COST):
    """Hourly strategy return. Position set at close t earns bar t+1."""
    pos = pos.fillna(0)
    held = pos.shift().fillna(0)
    trade = pos.diff().abs().fillna(pos.abs())
    return held * df.close.pct_change().fillna(0) - trade * cost - held * df.funding


def stats(r, pos=None):
    eq = (1 + r).cumprod()
    sd = r.std()
    out = dict(ret=eq.iloc[-1] - 1,
               sharpe=r.mean() / sd * np.sqrt(HOURS) if sd > 0 else 0.0,
               maxdd=(eq / eq.cummax() - 1).min())
    if pos is not None:
        p = np.sign(pos.loc[r.index])
        out["trades"] = int(((p != p.shift()) & (p != 0)).sum())
    return out


def window(s, w): return s.loc[w[0]:w[1]]


def run():
    agg = dict(open="first", high="max", low="min", close="last", volume="sum", funding="sum")
    dfs = {s: data.load(s).resample(TF).agg(agg).dropna() for s in SYMS}
    rows, best, wf = [], {}, {}
    for name, (fn, grid) in STRATS.items():
        keys = list(grid)
        cands = []
        for combo in itertools.product(*grid.values()):
            p = dict(zip(keys, combo))
            if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
                continue
            raw = {s: fn(df, **p) for s, df in dfs.items()}
            for side, vt in itertools.product(["both", "long"], [False, True]):
                per = {}
                for s, df in dfs.items():
                    pos = apply_polish(df, raw[s], side, vt)
                    r = pnl(df, pos)
                    per[s] = (r, pos)
                tr = [stats(window(r, TRAIN), pos) for r, pos in per.values()]
                if min(t["trades"] for t in tr) < MIN_TRADES:   # too few trades = luck, not edge
                    continue
                score = np.mean([t["sharpe"] for t in tr])
                cands.append((score, {**p, "side": side, "vt": vt}, per))
        if not cands:
            print(f"{name:15s} skipped: too few train trades per symbol for every setting", flush=True)
            continue
        cands.sort(key=lambda c: c[0], reverse=True)
        score, params, per = cands[0]
        best[name] = (params, per)
        wf[name] = walk_forward(cands)
        for s, (r, pos) in per.items():
            rows.append(dict(strategy=name, sym=s, params=params,
                             **{f"train_{k}": v for k, v in stats(window(r, TRAIN), pos).items()},
                             **{f"test_{k}": v for k, v in stats(window(r, TEST), pos).items()},
                             **{f"y2026_{k}": v for k, v in stats(window(r, FULL), pos).items()},
                             **{f"y2025_{k}": v for k, v in stats(window(r, Y2025), pos).items()}))
        print(f"{name:15s} train Sharpe {score:5.2f}  {params}", flush=True)
    for s, df in dfs.items():
        pos = buy_hold(df); r = pnl(df, pos)
        rows.append(dict(strategy="buy_hold", sym=s, params={},
                         **{f"train_{k}": v for k, v in stats(window(r, TRAIN), pos).items()},
                         **{f"test_{k}": v for k, v in stats(window(r, TEST), pos).items()},
                         **{f"y2026_{k}": v for k, v in stats(window(r, FULL), pos).items()},
                             **{f"y2025_{k}": v for k, v in stats(window(r, Y2025), pos).items()}))
    res = pd.DataFrame(rows)
    res.to_csv(f"results_{TF}.csv", index=False)
    return res, best, dfs, wf


def walk_forward(cands, lookback=3):
    """Every month from April: pick the candidate with the best equal-weight-portfolio Sharpe over the
    previous `lookback` months, trade it for the month. Returns the stitched out-of-sample return series."""
    port = [(c[1], pd.concat([r for r, _ in c[2].values()], axis=1).mean(axis=1)) for c in cands]
    months = pd.date_range("2026-04-01", port[0][1].index[-1], freq="MS")
    out, picks = [], []
    for m in months:
        lo, hi = m - pd.DateOffset(months=lookback), m - pd.Timedelta(hours=1)
        params, r = max(port, key=lambda pr: stats(pr[1].loc[lo:hi])["sharpe"])
        out.append(r.loc[m:m + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)])
        picks.append((f"{m:%Y-%m}", params))
    return pd.concat(out), picks


def summarize(res):
    g = res.groupby("strategy")[["train_sharpe", "test_sharpe", "test_ret", "test_maxdd", "y2025_ret", "y2025_sharpe", "y2025_maxdd", "test_trades"]].mean()
    return g.sort_values("test_sharpe", ascending=False)


if __name__ == "__main__":
    res, best, dfs, wf = run()
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("\nAverages across BTC/ETH/SOL (test = unseen Jul-Oct 2026):")
    with pd.option_context("display.max_columns", 20):
        print(summarize(res))
    print("\nPer symbol, test window:")
    print(res.pivot(index="strategy", columns="sym", values="test_ret"))
    print("\nRolling walk-forward (re-tune monthly on prior 3 months), equal-weight BTC/ETH/SOL, Apr-Oct 2026:")
    bh = pd.concat([pnl(df, buy_hold(df)) for df in dfs.values()], axis=1).mean(axis=1)
    rows = {n: stats(r) for n, (r, _) in wf.items()}
    rows["buy_hold"] = stats(bh.loc[wf["tsmom"][0].index])
    rows["pattern_basket"] = stats(pd.concat([wf[n][0] for n in ["double_top_bot", "head_shoulders", "triangle", "range_breakout"] if n in wf], axis=1).mean(axis=1))
    trend = ["donchian", "tsmom", "ema_cross", "sma_cross", "macd", "supertrend"]
    rows["trend_basket"] = stats(pd.concat([wf[n][0] for n in trend if n in wf], axis=1).mean(axis=1))
    print(pd.DataFrame(rows).T.sort_values("sharpe", ascending=False))
    pd.DataFrame({n: r for n, (r, _) in wf.items()}).to_csv(f"walkforward_returns_{TF}.csv")
    sys.exit(0)
