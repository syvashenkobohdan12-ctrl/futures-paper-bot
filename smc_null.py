"""Usage: python smc_null.py   (offline, ~1 min on 4 cores)
Null distribution of the whole smc_lab protocol: run it on random-walk universes (no edge by construction)
and record what it reports. A real result has to beat these, not zero."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from concurrent.futures import ProcessPoolExecutor
import smc_lab, smc
def fake(seed, n=41928):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    vol = 0.007 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))            # slowly drifting volatility, like real markets
    c = 100*np.exp(np.cumsum(rng.normal(0, 1, n) * vol)); o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c)*np.exp(np.abs(rng.normal(0, 1, n)) * vol * .6); l = np.minimum(o, c)*np.exp(-np.abs(rng.normal(0, 1, n)) * vol * .6)
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=rng.lognormal(0, .5, n), funding=0.0), idx)
def universe(u):
    data = {}
    for j, sym in enumerate(smc_lab.DEV + smc_lab.HOLD):
        smc_lab.smc_data.load = lambda s, tf, _j=j: (fake(u * 100 + _j), "fake")
        _, df, F, b = smc_lab.prep(sym)
        data[sym] = (df, F, b)
    cfg, t = smc_lab.protocol(data, say=lambda *a: None)
    r = lambda k: smc_lab.st(t.R[t.set == k])
    d, a, b = r("dev"), r("hold_A"), r("hold_B")
    return dict(u=u, nf=len(cfg["filters"]), dev_t=d["t"], dev_R=d["avgR"], A_R=a["avgR"], A_t=a["t"], B_R=b["avgR"], B_t=b["t"],
                AB_R=t.R[t.set != "dev"].mean(), AB_n=int((t.set != "dev").sum()))
with ProcessPoolExecutor(4) as ex:
    res = pd.DataFrame(list(ex.map(universe, range(40))))
res.to_csv("smc_results/null.csv", index=False)
print(res.round(3).to_string(index=False))
print("\nquantiles:"); print(res[["dev_t", "dev_R", "A_R", "B_R", "AB_R", "A_t", "B_t"]].quantile([.5, .9, .95]).round(3))
