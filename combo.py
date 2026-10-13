"""Combined bot backtest: 50% daily 72-day momentum + 50% 4h MACD(48,168,24), both long/short and vol-targeted,
on BTC/ETH/SOL equal weight. No new entries while daily ADX(14) > 40 (stretched trend; see improve.py). Positions are netted per coin on the hourly grid (one account), costs on the net."""
import numpy as np, pandas as pd
import data
from backtest import SYMS, apply_polish, pnl, stats
from strategies import tsmom, macd, adx

AGG = dict(open="first", high="max", low="min", close="last", volume="sum", funding="sum")
SLEEVES = {  # name: (timeframe, hours, signal fn)
    "daily_mom": ("1D", 24, lambda d: tsmom(d, look=72)),
    "macd_4h":   ("4h", 4,  lambda d: macd(d, fast=48, slow=168, sig=24)),
}
WEIGHTS = {"daily_mom": 0.5, "macd_4h": 0.5}
ADX_MAX = 40
PERIODS = {"2025 (unseen)": ("2025-01-01", "2025-12-31 23:00"),
           "2026 H1 (tuning)": ("2026-01-01", "2026-06-30 23:00"),
           "Jul-Oct 2026 (unseen)": ("2026-07-01", "2099"),
           "2025-2026 all": ("2025-01-01", "2099")}


def to_hourly(pos, hours, index):
    """A bar's signal is known at its close, i.e. at its last hour; hold it until the next bar's close."""
    pos = pos.copy()
    pos.index = pos.index + pd.Timedelta(hours=hours - 1)
    return pos.reindex(index).ffill().fillna(0)


def gate(pos, close, adx_h, stop, trail, adx_max):
    """Per trade (a run of one sign): block entry if ADX too high, exit on stop/trailing, re-arm on next flip."""
    p, c, a = pos.values.copy(), close.values, adx_h.values
    side = np.sign(p)
    prev, blocked, entry, peak = 0, False, 0.0, 0.0
    for i in range(len(p)):
        s = side[i]
        if s != prev:
            entry, peak, prev = c[i], 0.0, s
            blocked = bool(adx_max and a[i] > adx_max)
        if s == 0:
            continue
        if not blocked:
            r = s * (c[i] / entry - 1)
            peak = max(peak, r)
            if (stop and r <= -stop) or (trail and peak - r >= trail):
                blocked = True
        if blocked:
            p[i] = 0.0
    return pd.Series(p, pos.index)


def sleeve_positions(h1):
    adx_h = to_hourly(adx(h1.resample("1D").agg(AGG).dropna()), 24, h1.index)
    out = {}
    for name, (tf, hrs, fn) in SLEEVES.items():
        d = h1.resample(tf).agg(AGG).dropna()
        pos = to_hourly(apply_polish(d, fn(d), "both", True, hrs), hrs, h1.index)
        out[name] = gate(pos, h1.close, adx_h, None, None, ADX_MAX)
    return out


def run():
    rets = {k: [] for k in [*SLEEVES, "combined", "buy_hold"]}
    corr_in = {k: [] for k in SLEEVES}
    for s in SYMS:
        h1 = data.load(s)
        pos = sleeve_positions(h1)
        for k, p in pos.items():
            rets[k].append(pnl(h1, p))
        rets["combined"].append(pnl(h1, sum(WEIGHTS[k] * p for k, p in pos.items())))
        rets["buy_hold"].append(pnl(h1, pd.Series(1.0, h1.index)))
    port = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in rets.items()}
    rows = []
    for pname, (a, b) in PERIODS.items():
        for k, r in port.items():
            rows.append(dict(period=pname, strategy=k, **stats(r.loc[a:b])))
    res = pd.DataFrame(rows)
    daily = pd.DataFrame(port).loc["2025":].resample("1D").sum()
    return res, daily


if __name__ == "__main__":
    res, daily = run()
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    for p, g in res.groupby("period", sort=False):
        print(f"\n{p}"); print(g.drop(columns="period").set_index("strategy"))
    print("\nDaily return correlation 2025-2026:"); print(daily.corr())
    eq = (1 + daily).cumprod(); eq.to_csv("combo_equity.csv")
    m = daily["combined"].resample("ME").sum()
    print(f"\nCombined: {int((m > 0).sum())}/{len(m)} months positive, worst month {m.min():.1%}, best {m.max():.1%}")
