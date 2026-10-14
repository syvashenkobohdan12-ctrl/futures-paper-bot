"""Out-of-sample sweep of the frozen smc.FINAL rules over more markets and every timeframe. No re-tuning:
the rules were picked on BTC/ETH/SOL 1h 2024-25, so everything else here is data they never saw.
Rule lengths stay in bars (k, BOS window, expiry, 72-bar time stop), so they span more time on higher TFs.
Costs per side: crypto 0.05% taker + 0.02% slippage + funding; stocks 0.01% fees + 0.02% slippage.
Usage: python smc_multi.py [tf ...]   (default: 5m 15m 1h 4h 1d)"""
import sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import smc, smc_data

CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
STOCKS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
COSTS = {"crypto": smc.COST, "stocks": 0.0001 + 0.0002}
TFS = ["5m", "15m", "1h", "4h", "1d"]
SEEN = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}          # 1h 2024-25 on these picked the rules
OUT = Path("smc_results")


def job(args):
    sym, tf = args
    group = "crypto" if sym.endswith("USDT") else "stocks"
    try:
        df, src = smc_data.load(sym, tf)
        t = smc.run(df, dict(smc.FINAL, **smc.FIXED), COSTS[group])
        info = dict(sym=sym, tf=tf, group=group, src=src, bars=len(df), first=str(df.index[0])[:10], last=str(df.index[-1])[:10])
        if t.empty:
            return info, t, None
        t = t.assign(sym=sym, tf=tf, group=group)
        t["seen"] = (tf == "1h") & (sym in SEEN) & (t.ts >= "2024-01-01") & (t.ts < "2026-01-01")
        fresh = t[~t.seen]
        rb = smc.random_baseline(df, fresh, smc.FIXED["hold"], n=60, cost=COSTS[group]) if len(fresh) else None
        return info, t, rb
    except Exception as e:
        traceback.print_exc()
        return dict(sym=sym, tf=tf, group=group, error=repr(e)[:200]), pd.DataFrame(), None


def stats(t):
    if len(t) == 0:
        return dict(n=0)
    r = t.R
    return dict(n=len(t), win=(r > 0).mean(), avgR=r.mean(), t=r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 1 else 0.0,
                pf=r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() else np.inf, stop_med=t.risk_pct.median(),
                fee_R=(2 * (COSTS[t.group.iloc[0]]) / t.risk_pct).median())


if __name__ == "__main__":
    tfs = sys.argv[1:] or TFS
    OUT.mkdir(exist_ok=True)
    jobs = [(s, tf) for tf in tfs for s in CRYPTO + STOCKS]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, jobs))
    infos = pd.DataFrame([r[0] for r in res])
    infos.to_csv(OUT / "multi_data.csv", index=False)
    print("== data coverage ==")
    print(infos.to_string(index=False))
    trades = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True)
    trades.to_csv(OUT / "multi_trades.csv", index=False)
    fresh = trades[~trades.seen]

    rows = []
    for (g, tf), sub in fresh.groupby(["group", "tf"], sort=False):
        bl = [(r[2], (fresh[(fresh.sym == r[0]["sym"]) & (fresh.tf == tf)]).shape[0]) for r in res
              if r[0].get("tf") == tf and r[0].get("group") == g and r[2] is not None]
        w = np.array([n for _, n in bl], float)
        runs = np.nansum(np.vstack([b * n for b, n in bl]), axis=0) / w.sum()   # trade-weighted pooled baseline
        s = stats(sub)
        s26 = stats(sub[sub.ts >= "2026-01-01"])
        bys = sub.groupby("sym").R.mean()
        rows.append(dict(group=g, tf=tf, **s, n2026=s26["n"], avgR2026=s26.get("avgR", np.nan),
                         rand_avgR=np.nanmean(runs), beats_random=np.mean(runs < s["avgR"]),
                         syms_pos=f"{(bys > 0).sum()}/{len(bys)}", per_year=round(len(sub) / max(1e-9, (sub.ts.max() - sub.ts.min()).days / 365) / sub.sym.nunique(), 1)))
    summ = pd.DataFrame(rows)
    order = {tf: i for i, tf in enumerate(TFS)}
    summ = summ.sort_values(["group", "tf"], key=lambda c: c.map(order) if c.name == "tf" else c)
    summ.to_csv(OUT / "multi_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== frozen rules, data they never saw (excludes BTC/ETH/SOL 1h 2024-25) ==")
    print(summ.round(3).to_string(index=False))
    bysym = fresh.groupby(["group", "tf", "sym"]).R.agg(["count", "mean"]).round(3)
    bysym.to_csv(OUT / "multi_by_symbol.csv")
    print("\n== by symbol (trades, avg R) ==")
    print(bysym.unstack("tf").to_string())
