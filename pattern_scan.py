"""Which chart patterns 'work most of the time'? For ~30 simple patterns on daily and 4h bars, measure how often
price is higher N bars later, against the normal rate for the same coin (crypto rises on ~51% of days anyway).
DEV  = BTC ETH SOL XRP BNB DOGE, until 2025-12   (where a pattern has to show an edge first)
HOLD = ADA AVAX LINK SUI all years, plus every coin in 2026   (where it has to show it again)
Confirmed = DEV edge >= +3 points with z >= 3, AND holdout edge >= +2 points in the same direction.
'Bearish' patterns are scored as shorts: success = price LOWER after N bars.
Usage: python pattern_scan.py"""
import numpy as np, pandas as pd

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
DEV = COINS[:6]
HORIZONS = {"1d": [1, 3, 10], "4h": [1, 6, 18]}


def patterns(d, daily):
    o, h, l, c = d.open, d.high, d.low, d.close
    body, rng = (c - o).abs(), (h - l).replace(0, np.nan)
    up, dn = c > o, c < o
    lw, uw = np.minimum(o, c) - l, h - np.maximum(o, c)
    ph, pl = h.shift(), l.shift()
    hi20, lo20 = c.rolling(20).max(), c.rolling(20).min()
    ma50 = c.rolling(50).mean()
    ret = c.pct_change()
    avg_rng = rng.rolling(20).mean()
    P = {  # name: (signal, side) ; side +1 = expect up, -1 = expect down
        "bullish engulfing":     (up & dn.shift() & (o <= c.shift()) & (c >= o.shift()), 1),
        "bearish engulfing":     (dn & up.shift() & (o >= c.shift()) & (c <= o.shift()), -1),
        "hammer":                ((lw >= 2 * body) & (uw <= 0.3 * rng) & (c < c.shift(5)), 1),
        "shooting star":         ((uw >= 2 * body) & (lw <= 0.3 * rng) & (c > c.shift(5)), -1),
        "doji":                  (body <= 0.1 * rng, 1),
        "inside bar":            ((h < ph) & (l > pl), 1),
        "outside bar up":        ((h > ph) & (l < pl) & up, 1),
        "3 green candles":       (up & up.shift() & up.shift(2), 1),
        "3 red candles":         (dn & dn.shift() & dn.shift(2), 1),
        "5 red candles":         (dn & dn.shift() & dn.shift(2) & dn.shift(3) & dn.shift(4), 1),
        "close at 20-bar high":  (c >= hi20, 1),
        "close at 20-bar low":   (c <= lo20, -1),
        "break above prev high": ((c > ph) & (c.shift() <= h.shift(2)), 1),
        "break below prev low":  ((c < pl) & (c.shift() >= l.shift(2)), -1),
        "sweep prev low, close back above": ((l < pl) & (c > pl), 1),
        "sweep prev high, close back below": ((h > ph) & (c < ph), -1),
        "big up bar (>2x range)":   ((rng > 2 * avg_rng) & up, 1),
        "big down bar (>2x range)": ((rng > 2 * avg_rng) & dn, 1),
        "narrowest range of 7":  (rng <= rng.rolling(7).min(), 1),
        "up >8% in one bar":     (ret > 0.08, 1),
        "down >8% in one bar":   (ret < -0.08, 1),
        "above 50-bar average":  (c > ma50, 1),
        "below 50-bar average":  (c < ma50, -1),
        "cross above 50-bar avg": ((c > ma50) & (c.shift() <= ma50.shift()), 1),
        "pullback in uptrend (red bar above avg)": (dn & (c > ma50) & (c.shift(20) < c), 1),
        "rally in downtrend (green bar below avg)": (up & (c < ma50) & (c.shift(20) > c), -1),
        "double bottom area (low near 20-bar low twice)": ((l <= lo20 * 1.01) & (l.shift(3).rolling(15).min() <= lo20 * 1.01) & up, 1),
    }
    if daily:
        for k, name in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]):
            P[f"day: {name}"] = (pd.Series(d.index.dayofweek == k, d.index), 1)
    return P


def scan(frames, tf):
    rows = []
    pats = {sym: patterns(d, tf == "1d") for sym, d in frames.items()}
    for name in next(iter(pats.values())):
        for H in HORIZONS[tf]:
            acc = {"dev": [0, 0, 0, 0.0], "hold": [0, 0, 0, 0.0]}   # hits, n, base hits (same coins/periods), sum ret
            base = {"dev": [0, 0], "hold": [0, 0]}
            for sym, d in frames.items():
                sig, side = pats[sym][name]
                fwd = d.close.shift(-H) / d.close - 1
                ok = fwd.notna()
                is_dev = (sym in DEV) & (d.index < "2026-01-01")
                for part, mask in (("dev", is_dev), ("hold", ~is_dev)):
                    m = mask & ok
                    s = sig.fillna(False).astype(bool) & m
                    win = (side * fwd > 0)
                    acc[part][0] += int(win[s].sum()); acc[part][1] += int(s.sum()); acc[part][3] += float((side * fwd)[s].sum())
                    base[part][0] += int(win[m].sum()); base[part][1] += int(m.sum())
            r = dict(pattern=name, side="up" if side > 0 else "down", tf=tf, bars_ahead=H)
            for part in ("dev", "hold"):
                hits, n, _, sret = acc[part]
                b = base[part][0] / max(base[part][1], 1)
                p = hits / n if n else np.nan
                r.update({f"{part}_n": n, f"{part}_hit": p, f"{part}_base": b, f"{part}_edge": p - b,
                          f"{part}_z": (p - b) / np.sqrt(b * (1 - b) / n) * np.sqrt(1 / H) if n else np.nan,   # overlap-adjusted
                          f"{part}_avg_move": sret / n if n else np.nan})
            r["confirmed"] = bool(r["dev_edge"] >= 0.03 and r["dev_z"] >= 3 and r["hold_edge"] >= 0.02)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import smc_data
    pd.set_option("display.width", 250, "display.max_rows", 500)
    out = []
    for tf in ("1d", "4h"):
        frames = {s: smc_data.crypto(s, tf) for s in COINS}
        out.append(scan(frames, tf))
    res = pd.concat(out, ignore_index=True)
    res.to_csv("smc_results/pattern_scan.csv", index=False)
    show = res.sort_values("dev_edge", ascending=False)
    cols = ["pattern", "side", "tf", "bars_ahead", "dev_n", "dev_hit", "dev_base", "dev_edge", "dev_z", "hold_n", "hold_hit", "hold_edge", "hold_avg_move", "confirmed"]
    print(f"== {len(res)} pattern x timeframe x horizon tests; confirmed = {int(res.confirmed.sum())} ==\n")
    print(show[cols].round(3).to_string(index=False))
