"""Mechanical Elliott-wave setups on an alternating zigzag of confirmed swing pivots.
wave3: impulse 1 then a Fibonacci pullback 2 (not below wave-1 start) -> enter on break of wave-1 extreme,
       stop at wave-2 end, target = wave-2 end + ext * wave 1.
wave5: waves 1-4 with 3 longer than 1 and 4 not overlapping 1 -> enter on break of wave-3 extreme,
       stop at wave-4 end, target = wave-4 end + ext * wave 1.
Registers itself into strategies.STRATS."""
import numpy as np
from strategies import STRATS
from patterns import _pivots, manage

BANDS = {"38-62": (0.382, 0.618), "38-79": (0.382, 0.786), "24-89": (0.236, 0.886)}


def zigzag_scan(df, k, setup, band, ext):
    """Returns entry signals plus the structural stop/target attached to each entry bar."""
    H, L, C = df.high.values, df.low.values, df.close.values
    ph, pl = _pivots(df, k)
    lo_r, hi_r = BANDS[band]
    zz, armed = [], None          # zz: alternating [(+1 high | -1 low, price)]
    n = len(C)
    ent, stops, tgts = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        new = [(1, H[i - k])] if ph[i] else []
        new += [(-1, L[i - k])] if pl[i] else []
        for t, p in new:
            if zz and zz[-1][0] == t:   # same side twice: keep the more extreme one
                if t * p > t * zz[-1][1]:
                    zz[-1] = (t, p)
            else:
                zz.append((t, p))
        if new:
            armed = setup(zz, lo_r, hi_r, ext)
        if armed:
            d, lvl, stop, tgt = armed
            if (d > 0 and C[i] > lvl >= C[i - 1]) or (d < 0 and C[i] < lvl <= C[i - 1]):
                ent[i], stops[i], tgts[i], armed = d, stop, tgt, None
    return ent, stops, tgts


def setup_wave3(zz, lo_r, hi_r, ext):
    if len(zz) < 3:
        return None
    (_, p0), (_, p1), (t2, p2) = zz[-3:]
    d = -t2                     # last pivot is a low -> bullish count
    w1 = (p1 - p0) * d
    if w1 <= 0:
        return None
    r = (p1 - p2) * d / w1
    if lo_r <= r <= hi_r:       # r < 1 also means wave 2 stayed above wave-1 start
        return d, p1, p2, p2 + d * ext * w1
    return None


def setup_wave5(zz, lo_r, hi_r, ext):
    if len(zz) < 5:
        return None
    (_, p0), (_, p1), (_, p2), (_, p3), (t4, p4) = zz[-5:]
    d = -t4
    w1, w3 = (p1 - p0) * d, (p3 - p2) * d
    if w1 <= 0 or w3 <= w1 or (p2 - p0) * d <= 0:   # wave 3 longer than 1; wave 2 above wave-1 start
        return None
    if (p4 - p1) * d <= 0:                           # wave 4 may not overlap wave 1
        return None
    r = (p3 - p4) * d / w3
    if lo_r <= r <= hi_r:
        return d, p3, p4, p4 + d * ext * w1
    return None


def _strat(setup):
    def fn(df, k, band, ext, hold):
        ent, s, t = zigzag_scan(df, k, setup, band, ext)
        return manage(df, ent, 0, 0, hold, s, t)
    return fn


STRATS.update({
    "elliott_wave3": (_strat(setup_wave3), dict(k=[6, 12, 24], band=list(BANDS), ext=[1.0, 1.618, 2.618], hold=[48, 168, 336])),
    "elliott_wave5": (_strat(setup_wave5), dict(k=[4, 6, 12, 24], band=list(BANDS), ext=[0.618, 1.0, 1.618], hold=[48, 168, 336])),
})
