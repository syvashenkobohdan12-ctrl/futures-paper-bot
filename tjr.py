"""TJR-style session model on 5m crypto, normal and REVERSED.
  1. Asia range: high/low of 00:00-04:00 UTC (8pm-midnight New York)
  2. London/NY window 06:00-16:00 UTC: price sweeps the Asia low (or high)
  3. 5m break of structure the other way: close beyond the last swing high (low) that formed before the sweep
  4. entry: limit at the near edge of the fair value gap the move left (untouched); stop beyond the sweep extreme
  5. target: the opposite side of the Asia range ("liq") or 2R; everything closes at 20:00 UTC; 1 trade/day/side
REVERSED = same trigger, opposite trade: a stop order at the same entry price, stop at the original target,
target at the original stop. Costs: limit entry 0.02% (maker), stop-order entry and stop/time exits 0.07%, target 0.02%.
DEV = BTC ETH SOL XRP BNB DOGE 2024-25; HOLDOUT = ADA AVAX LINK SUI + every coin in 2026.
Usage: python tjr.py"""
import numpy as np, pandas as pd
import smc
from patterns import _pivots

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
DEV = COINS[:6]
TAKER, MAKER = 0.0005 + 0.0002, 0.0002
K = 2                                             # 5m swing: 2 bars each side


def setups(d, target):
    """Long-orientation setups (sweep of the Asia LOW). Run on smc._flip(d) for the high side."""
    H, L, C = d.high.values, d.low.values, d.close.values
    a = smc.atr(d, 14).values
    ph, _ = _pivots(d, K)
    hrs = d.index.hour.values
    day = d.index.normalize()
    out, highs = [], []
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(C)]
    for s0, e0 in zip(starts, ends):
        asia = [i for i in range(s0, e0) if hrs[i] < 4]
        if len(asia) < 40:
            continue
        lvl, top = L[asia].min(), H[asia].max()
        swept, ext, ext_i, bos, done = False, None, None, None, False
        for i in range(s0, e0):
            if ph[i]:
                highs.append((i - K, H[i - K]))       # keep recording swings all day, even after a setup
            if done or not 6 <= hrs[i] < 16:
                continue
            if not swept:
                if L[i] < lvl:
                    prior = [h for j, h in highs[-20:] if j < i]
                    swept, ext, ext_i, bos = True, L[i], i, (prior[-1] if prior else None)
                    done = bos is None
                continue
            if C[i] > bos:                             # break of structure: look for the gap
                done = True                            # one setup per day per side
                for t in range(i, ext_i + 1, -1):
                    if L[t] > H[t - 2] and (t == i or L[t + 1:i + 1].min() > L[t]):
                        e = L[t]
                        st = ext - 0.1 * a[i]
                        tg = top if target == "liq" else e + 2 * (e - st)
                        if tg > e and e > st:
                            last = s0 + int(np.searchsorted(hrs[s0:e0], 16))   # orders live until 16:00
                            out.append(dict(bar=i + 1, entry=e, stop=st, tgt=tg, expiry=max(1, last - i - 1)))
                        break
                continue
            if L[i] < ext:
                ext, ext_i = L[i], i
    return out


def run(d, target, reverse):
    """Trades for one coin, both sides. Reverse = mirror each setup into the opposite trade."""
    eod = np.where(d.index.hour.values >= 20, np.inf, -np.inf)     # forces the 20:00 UTC exit
    rows = []
    for side, x in ((1, d), (-1, smc._flip(d))):
        su = setups(x, target)
        if not reverse:
            tr = smc.simulate(x, su, 400, fees=(MAKER, MAKER, TAKER), through=0.0002, trail=eod)
            sd = side
        else:                                          # trade it the other way, in the opposite frame
            y = smc._flip(x)
            mirrored = [dict(bar=s["bar"], kind="stop", entry=-s["entry"], stop=-s["tgt"], tgt=-s["stop"],
                             expiry=s["expiry"], fee_in=TAKER) for s in su]
            tr = smc.simulate(y, mirrored, 400, fees=(TAKER, MAKER, TAKER), trail=eod)
            sd = -side
        for t in tr:
            t.update(side=sd)
            rows.append(t)
    return rows


def report(t, label):
    t = t.copy()
    t["set"] = np.where(t.sym.isin(DEV) & (t.ts < "2026-01-01"), "dev", "holdout")
    out = []
    for s in ("dev", "holdout"):
        x = t[t.set == s]
        tt = x.R.mean() / (x.R.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
        out.append(f"{s} n={len(x):5d} win={np.mean(x.R > 0):.0%} avgR={x.R.mean():+.3f} t={tt:+.2f}")
    print(f"{label:28s} | " + " | ".join(out) + f" | stop median {t.risk_pct.median():.2%}")
    return t


if __name__ == "__main__":
    import smc_data
    pd.set_option("display.width", 220)
    frames = {s: smc_data.crypto(s, "5m") for s in COINS}
    res = {}
    for target in ("liq", "2R"):
        for reverse in (False, True):
            rows = []
            for s, d in frames.items():
                for t in run(d, target, reverse):
                    t["sym"] = s
                    rows.append(t)
            res[(target, reverse)] = report(pd.DataFrame(rows), f"target={target} {'REVERSED' if reverse else 'normal'}")
    print("\nby year (avg R):")
    for k, t in res.items():
        print(f"  {k[0]:3s} {'REVERSED' if k[1] else 'normal  '}", t.groupby(t.ts.dt.year).R.agg(lambda r: f"{len(r)}/{r.mean():+.2f}").to_dict())
    print("\nexits:", {f"{k[0]} {'rev' if k[1] else 'norm'}": t.why.value_counts().to_dict() for k, t in res.items()})
    print("fees in R (median round trip):", {f"{k[0]} {'rev' if k[1] else 'norm'}": round(float((0.0009 / t.risk_pct).median()), 3) for k, t in res.items()})
    pd.concat([t.assign(target=k[0], reversed=k[1]) for k, t in res.items()]).to_csv("smc_results/tjr_trades.csv", index=False)
