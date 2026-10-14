"""Daily Donchian breakout: the one price-action setup that beat its noise threshold in pa.py.
Rule: a daily close above the prior 20-day high -> buy the next open; stop 2 ATR(14) under the signal close.
Exit variants chosen on DEV only (BTC ETH SOL XRP BNB DOGE, 2020-2025), holdouts just reported:
  fixed  : 3R target or 72 days, whichever first (what pa.py tested)
  trail10: no target; exit on a close below the prior 10-day low (Turtle system 1), stop still in place
  trail20: same with the prior 20-day low
  sides  : long+short (shorts mirror) or long only
trades() is shared by this research and donchian_bot.py, so paper trades are exactly what the backtest does.
Usage: python donchian.py            real data -> smc_results/donchian_report.txt
       python donchian.py --null N   same variant choice on N random-walk universes (offline)"""
import itertools, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd


def _ar1(eps, phi):
    out, x = np.empty(len(eps)), 0.0
    for i, e in enumerate(eps):
        x = phi * x + e
        out[i] = x
    return out
import smc

N, STOP_ATR = 20, 2.0
TAKER, MAKER = 0.0005 + 0.0002, 0.0002
EXITS = {"fixed": dict(tp=3.0, hold=72, trail=None), "trail10": dict(tp=None, hold=365, trail=10),
         "trail20": dict(tp=None, hold=365, trail=20)}
SIDES = {"both": (1, -1), "long": (1,)}
DEV = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
HOLD = ["ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
DEV_END = "2026-01-01"
LIVE = dict(exit="fixed", sides="long")   # the DEV pick (smc_results/donchian_report.txt); donchian_bot.py trades this
OUT = Path("smc_results")


def _long_trades(d, exit_):
    H, C = d.high.values, d.close.values
    a = smc.atr(d, 14).values
    hi = d.high.rolling(N).max().shift(1).values
    orders = []
    for i in range(N + 2, len(C)):
        if C[i] > hi[i] and C[i - 1] <= hi[i - 1]:
            st = C[i] - STOP_ATR * a[i]
            tgt = C[i] + exit_["tp"] * (C[i] - st) if exit_["tp"] else np.inf
            orders.append(dict(bar=i + 1, kind="market", entry=C[i], stop=st, tgt=tgt, expiry=1))
    trail = d.low.rolling(exit_["trail"]).min().shift(1).values if exit_["trail"] else None
    return smc.simulate(d, orders, exit_["hold"], fees=(TAKER, MAKER, TAKER), trail=trail)


def trades(daily, exit_name="fixed", sides="both"):
    """daily OHLC+funding df -> trades with real (positive) prices, side, R."""
    rows = []
    for s in SIDES[sides]:
        d = daily if s > 0 else smc._flip(daily)
        for t in _long_trades(d, EXITS[exit_name]):
            t.update(side=s, entry=abs(t["entry"]), exit=abs(t["exit"]), stop=abs(t["stop"]),
                     tgt=abs(t["tgt"]) if np.isfinite(t["tgt"]) else None)
            rows.append(t)
    return pd.DataFrame(rows)


def tstat(r):
    r = np.asarray(r, float)
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else np.nan


def evaluate(frames, exit_name, sides):
    t = pd.concat([trades(df, exit_name, sides).assign(sym=s) for s, df in frames.items()], ignore_index=True)
    t["set"] = np.where(t.ts >= DEV_END, "hold_B", np.where(t.sym.isin(DEV), "dev", "hold_A"))
    return t


def choose(frames, say=print):
    rows, best = [], None
    for ex, sd in itertools.product(EXITS, SIDES):
        t = evaluate(frames, ex, sd)
        r = {k: t.R[t.set == k] for k in ("dev", "hold_A", "hold_B")}
        row = dict(exit=ex, sides=sd, dev_n=len(r["dev"]), dev_R=r["dev"].mean(), dev_t=tstat(r["dev"]),
                   A_R=r["hold_A"].mean(), B_R=r["hold_B"].mean(), hold_R=t.R[t.set != "dev"].mean(),
                   hold_t=tstat(t.R[t.set != "dev"]))
        rows.append(row)
        say(f"{ex:8s} {sd:5s} | dev n={row['dev_n']:4d} R={row['dev_R']:+.3f} t={row['dev_t']:+.2f} | "
            f"hold_A R={row['A_R']:+.3f} | hold_B R={row['B_R']:+.3f} | holdouts R={row['hold_R']:+.3f} t={row['hold_t']:+.2f}")
        if best is None or row["dev_t"] > best[0]["dev_t"]:
            best = (row, t)
    return best, rows


def portfolio(t, risk=0.005, max_open=10):
    """Fixed-fractional sizing in trade-entry order; skip entries while max_open positions are open."""
    t = t.sort_values("ts").reset_index(drop=True)
    # same timestamp: entries first, so a trade stopped out on its entry day still closes
    events = sorted([(r.ts, 0, i) for i, r in t.iterrows()] + [(r.exit_ts, 1, i) for i, r in t.iterrows()])
    eq, open_, taken, curve = 1.0, {}, [], []
    for ts, kind, i in events:
        if kind == 0:
            if len(open_) < max_open:
                open_[i] = eq * risk
                taken.append(i)
        elif i in open_:
            eq += open_.pop(i) * t.R[i]
            curve.append((ts, eq))
    c = pd.Series([v for _, v in curve], [k for k, _ in curve])
    return c, t.loc[taken]


def synth_daily(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", "2026-10-14", freq="h", inclusive="left")
    n = len(idx)
    vol = 0.007 * np.exp(_ar1(rng.normal(0, 0.01, n), 0.998))   # mean-reverting log-vol: clusters, never explodes
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol)); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 1, n)) * vol * .6)
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 1, n)) * vol * .6)
    df = pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=1.0, funding=np.where(idx.hour % 8 == 0, 0.0001, 0.0)), idx)
    return df.resample("1D").agg(dict(open="first", high="max", low="min", close="last", volume="sum", funding="sum"))


def _null(u):
    frames = {s: synth_daily(u * 100 + j) for j, s in enumerate(DEV + HOLD)}
    (row, _), _ = choose(frames, say=lambda *a: None)
    return dict(row, u=u)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    if "--null" in sys.argv:
        n = int(sys.argv[sys.argv.index("--null") + 1])
        with ProcessPoolExecutor(4) as ex:
            nul = pd.DataFrame(list(ex.map(_null, range(n))))
        nul.to_csv(OUT / "donchian_null.csv", index=False)
        print(nul[["dev_t", "hold_R", "hold_t"]].quantile([.5, .9, .95]).round(3))
        sys.exit()
    import smc_data
    frames = {s: smc_data.crypto(s, "1d") for s in DEV + HOLD}
    print("== exit/side variants (choice by DEV t-stat) ==")
    (row, t), rows = choose(frames)
    print(f"\n-> picked on DEV: exit={row['exit']} sides={row['sides']}")
    print(t.groupby(t.ts.dt.year).R.agg(["count", "mean"]).round(3).T.to_string())
    print(t.groupby("sym").R.agg(["count", "mean"]).round(3).T.to_string())
    print("exits:", t.why.value_counts().to_dict(), " win:", round((t.R > 0).mean(), 3),
          " median days held:", (t.exit_ts - t.ts).dt.days.median())
    for risk, cap in ((0.005, 10), (0.01, 10), (0.005, 99)):
        c, taken = portfolio(t, risk, cap)
        yrs = (c.index[-1] - c.index[0]).days / 365
        print(f"risk {risk:.1%} max_open {cap}: x{c.iloc[-1]:.2f} ({c.iloc[-1] ** (1 / yrs) - 1:+.0%}/yr) "
              f"maxDD {(c / c.cummax() - 1).min():.1%} trades {len(taken)}")
    t.to_csv(OUT / "donchian_trades.csv", index=False)
    pd.Series(dict(exit=row["exit"], sides=row["sides"])).to_json(OUT / "donchian_config.json")
    # overlap with the live bot: monthly realized R of this sleeve vs the live combo's monthly returns
    try:
        import risk as live
        eq, lt, _, _ = live.simulate("2024-01-01", live.STOP, **dict(max_hold=168, tp_r=3, reenter=("time",), be_r=1.5))
        live_m = eq.resample("ME").last().pct_change().dropna()
        c, _ = portfolio(t, 0.005, 10)
        sleeve_m = c.resample("ME").last().ffill().pct_change().reindex(live_m.index).fillna(0)
        print(f"\noverlap with the live bot (monthly returns 2024+): correlation {live_m.corr(sleeve_m):+.2f}; "
              f"live sharpe-ish {live_m.mean() / live_m.std() * np.sqrt(12):.2f}, sleeve {sleeve_m.mean() / sleeve_m.std() * np.sqrt(12):.2f}, "
              f"50/50 blend {((live_m + sleeve_m) / 2).mean() / ((live_m + sleeve_m) / 2).std() * np.sqrt(12):.2f}")
    except Exception as e:
        print("overlap check failed:", repr(e)[:200])
