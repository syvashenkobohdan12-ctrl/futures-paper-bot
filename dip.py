"""Dip bounce: after a daily close down more than 8%, buy at that close and sell at the next day's close.
Long only, 10 coins, each trade gets 10% of the account. 0.07% cost per buy/sell, funding paid while held.
Reports: per year, per coin, dev (6 coins to 2025) vs holdout (4 other coins + 2026), neighbouring settings,
random-day baseline, and how it combines with the breakout sleeve (simple.py).
Usage: python dip.py"""
import numpy as np, pandas as pd
import simple

COINS, DEV = simple.COINS, simple.COINS[:6]
DROP, HOLD, COST = 0.08, 1, 0.0007


def trades(daily, drop=DROP, hold=HOLD):
    rows = []
    for coin, d in daily.items():
        c, f = d.close, d.funding
        r1 = c.pct_change()
        for i in np.flatnonzero((r1 < -drop).values):
            if i + hold >= len(c):
                continue
            ret = c.iloc[i + hold] / c.iloc[i] - 1 - 2 * COST - f.iloc[i + 1:i + hold + 1].sum()
            rows.append(dict(coin=coin, day=c.index[i], ret=ret, drop=r1.iloc[i]))
    t = pd.DataFrame(rows)
    t["set"] = np.where(t.coin.isin(DEV) & (t.day < "2026-01-01"), "dev", "holdout")
    return t


def daily_returns(daily, drop=DROP, hold=HOLD):
    """Account return per day: every open dip trade is a 10% slice."""
    parts = []
    for coin, d in daily.items():
        sig = (d.close.pct_change() < -drop).astype(float)
        pos = sum(sig.shift(k) for k in range(1, hold + 1)).clip(upper=1).fillna(0)
        cost = pos.diff().abs().fillna(0) * COST
        parts.append((pos * d.close.pct_change().fillna(0) - cost - pos * d.funding).rename(coin))
    return pd.concat(parts, axis=1).fillna(0).sum(axis=1) * 0.10


def summary(t):
    w = t.ret > 0
    return f"trades {len(t):4d} | won {w.mean():.0%} | avg {t.ret.mean():+.2%} | avg win {t.ret[w].mean():+.1%} | avg loss {t.ret[~w].mean():+.1%} | worst {t.ret.min():+.1%}"


if __name__ == "__main__":
    import smc_data
    daily = {c: smc_data.crypto(c, "1d") for c in COINS}
    t = trades(daily)
    print(f"== dip bounce: down >{DROP:.0%} close -> buy, sell {HOLD} day later ==")
    print("all     ", summary(t))
    for s in ("dev", "holdout"):
        print(f"{s:8s}", summary(t[t.set == s]))
    print("\nby year:")
    for y, g in t.groupby(t.day.dt.year):
        print(f"  {y}  {summary(g)}")
    print("\nby coin:", t.groupby("coin").ret.mean().map("{:+.2%}".format).to_dict())
    # baseline: the same coins, random days, same 1-day hold
    rng = np.random.default_rng(0)
    base = []
    for _ in range(200):
        rs = []
        for coin, n in t.coin.value_counts().items():
            c = daily[coin].close
            idx = rng.integers(1, len(c) - HOLD, n)
            rs += list(c.values[idx + HOLD] / c.values[idx] - 1 - 2 * COST)
        base.append(np.mean(rs))
    print(f"\nrandom days, same coins and count (200 runs): avg {np.mean(base):+.2%}, best run {np.max(base):+.2%} "
          f"-> strategy avg {t.ret.mean():+.2%} beats {np.mean(np.array(base) < t.ret.mean()):.0%}")
    print("\nneighbouring settings (avg per trade / trades):")
    for d_ in (0.05, 0.06, 0.08, 0.10, 0.12):
        print("  drop >{:.0%}:".format(d_), " | ".join(f"hold {h}d {trades(daily, d_, h).ret.mean():+.2%} ({len(trades(daily, d_, h))})" for h in (1, 2, 3)))
    # account level, alone and next to the breakout sleeve
    dip = daily_returns(daily)
    brk, _ = simple.backtest(daily)
    both = (dip + brk) / 2
    print("\naccount (10% per trade):")
    for name, r in (("dip bounce", dip), ("breakout", brk), ("50/50 both", both)):
        st = simple.stats(r)
        yrs = {y: f"{(1 + g).prod() - 1:+.0%}" for y, g in r.groupby(r.index.year)}
        print(f"  {name:11s} {st['cagr']:+.0%}/yr  worst DD {st['maxdd']:.0%}  sharpe {st['sharpe']:.2f}  {yrs}")
    print(f"  correlation of daily returns dip vs breakout: {dip.corr(brk):+.2f}")
    t.to_csv("smc_results/dip_trades.csv", index=False)
