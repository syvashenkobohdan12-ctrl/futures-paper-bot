"""Chart-geometry patterns built on swing pivots. A pivot at bar p is only known k bars later
(confirmed at p+k), so nothing here peeks ahead. Entry = close breaks the pattern's neckline;
exit = ATR stop, R-multiple target, or time stop. Registers itself into strategies.STRATS."""
import numpy as np, pandas as pd
from strategies import STRATS, atr


def _pivots(df, k):
    w = 2 * k + 1
    ph = (df.high.shift(k) == df.high.rolling(w).max()).values
    pl = (df.low.shift(k) == df.low.rolling(w).min()).values
    return ph, pl


def scan(df, k, detect, tol=None):
    """Walk bars; when a new pivot confirms, ask `detect` for armed setups [(dir, level)].
    Fire +1/-1 on the bar whose close crosses an armed level."""
    H, L, C = df.high.values, df.low.values, df.close.values
    ph, pl = _pivots(df, k)
    highs, lows, armed = [], [], []
    ent = np.zeros(len(C))
    for i in range(len(C)):
        if ph[i] or pl[i]:
            if ph[i]: highs.append((i - k, H[i - k]))
            if pl[i]: lows.append((i - k, L[i - k]))
            armed = detect(highs[-3:], lows[-3:], tol)
        for d, lvl in armed:
            if (d > 0 and C[i] > lvl >= C[i - 1]) or (d < 0 and C[i] < lvl <= C[i - 1]):
                ent[i], armed = d, []
                break
    return ent


def manage(df, ent, stop_atr, rr, hold, stops=None, tgts=None):
    """Turn entry signals into a position with stop/target/time exits.
    stops/tgts: optional per-entry structural levels (override the ATR ones).
    shortcut: exits fill at the bar close, not at the stop price; model intrabar fills if stops get tight."""
    H, L, C = df.high.values, df.low.values, df.close.values
    a = atr(df, 14).values
    pos = np.zeros(len(C)); cur = 0
    for i in range(len(C)):
        if cur:
            hit_stop = L[i] <= stop if cur > 0 else H[i] >= stop
            hit_tgt = H[i] >= tgt if cur > 0 else L[i] <= tgt
            if hit_stop or hit_tgt or i - start >= hold:
                cur = 0
        if not cur and ent[i]:
            cur, start = ent[i], i
            stop = C[i] - cur * stop_atr * a[i]
            tgt = C[i] + cur * rr * stop_atr * a[i]
            if stops is not None:
                stop, tgt = stops[i], tgts[i]
        pos[i] = cur
    return pd.Series(pos, df.index)


def _between(piv, lo, hi):
    return [p for i, p in piv if lo < i < hi]


# --- detectors: (last 3 highs, last 3 lows, tol) -> [(dir, neckline)] ---
def det_double(highs, lows, tol):
    out = []
    if len(lows) >= 2:  # double bottom -> long above the peak between
        (i1, a), (i2, b) = lows[-2:]
        mid = _between(highs, i1, i2)
        if mid and abs(a - b) / max(a, b) < tol:
            out.append((1, max(mid)))
    if len(highs) >= 2:  # double top -> short below the trough between
        (i1, a), (i2, b) = highs[-2:]
        mid = _between(lows, i1, i2)
        if mid and abs(a - b) / max(a, b) < tol:
            out.append((-1, min(mid)))
    return out


def det_hs(highs, lows, tol):
    out = []
    if len(highs) == 3:  # head & shoulders -> short below neckline
        (i1, s1), (_, h), (i3, s3) = highs
        neck = _between(lows, i1, i3)
        if neck and h > max(s1, s3) * (1 + tol) and abs(s1 - s3) / h < tol:
            out.append((-1, min(neck)))
    if len(lows) == 3:  # inverse H&S -> long above neckline
        (i1, s1), (_, h), (i3, s3) = lows
        neck = _between(highs, i1, i3)
        if neck and h < min(s1, s3) * (1 - tol) and abs(s1 - s3) / h < tol:
            out.append((1, max(neck)))
    return out


def det_triangle(highs, lows, tol):
    """Lower highs + higher lows (converging). Trade the break of the last swing either way."""
    if len(highs) >= 2 and len(lows) >= 2:
        h1, h2 = highs[-2][1], highs[-1][1]
        l1, l2 = lows[-2][1], lows[-1][1]
        if h2 < h1 * (1 - tol) and l2 > l1 * (1 + tol):
            return [(1, h2), (-1, l2)]
    return []


def det_range(highs, lows, tol):
    """Horizontal support/resistance: break of the last confirmed swing high/low."""
    return ([(1, highs[-1][1])] if highs else []) + ([(-1, lows[-1][1])] if lows else [])


def _strat(detect):
    def fn(df, k, tol, stop_atr, rr, hold):
        return manage(df, scan(df, k, detect, tol), stop_atr, rr, hold)
    return fn


EXITS = dict(stop_atr=[1.5, 3], rr=[1.5, 3], hold=[48, 168])
STRATS.update({
    "double_top_bot":  (_strat(det_double),   dict(k=[6, 12, 24], tol=[0.005, 0.01, 0.02], **EXITS)),
    "head_shoulders":  (_strat(det_hs),       dict(k=[6, 12, 24], tol=[0.005, 0.01, 0.02], **EXITS)),
    "triangle":        (_strat(det_triangle), dict(k=[6, 12, 24], tol=[0.0, 0.005, 0.01], **EXITS)),
    "range_breakout":  (_strat(det_range),    dict(k=[6, 12, 24, 48], tol=[0], **EXITS)),
})
