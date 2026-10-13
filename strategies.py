"""Strategy library. Each fn(df, **params) -> target position per bar in [-1, 1], decided at bar close.
The backtester applies it to the NEXT bar, so signals may use the current close."""
import numpy as np, pandas as pd


def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def atr(df, n):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def legs(idx, long_in, long_out, short_in, short_out):
    """Stateful entries/exits without a loop: each leg holds from entry until its exit."""
    def leg(i, o):
        return pd.Series(np.where(i, 1.0, np.where(o, 0.0, np.nan)), idx).ffill().fillna(0.0)
    return leg(long_in, long_out) - leg(short_in, short_out)


# --- trend following ---
def sma_cross(df, fast, slow):
    return np.sign(df.close.rolling(fast).mean() - df.close.rolling(slow).mean()).fillna(0)


def ema_cross(df, fast, slow):
    return np.sign(ema(df.close, fast) - ema(df.close, slow))


def macd(df, fast, slow, sig):
    m = ema(df.close, fast) - ema(df.close, slow)
    return np.sign(m - ema(m, sig))


def tsmom(df, look):
    return np.sign(df.close.pct_change(look)).fillna(0)


def donchian(df, entry, exit):
    hi, lo = df.high.rolling(entry).max().shift(), df.low.rolling(entry).min().shift()
    xhi, xlo = df.high.rolling(exit).max().shift(), df.low.rolling(exit).min().shift()
    c = df.close
    return legs(df.index, c > hi, c < xlo, c < lo, c > xhi)


def supertrend(df, n, mult):
    a = atr(df, n).values
    mid = ((df.high + df.low) / 2).values
    c = df.close.values
    up, dn = mid + mult * a, mid - mult * a
    pos = np.zeros(len(c)); fu, fd, d = up[0], dn[0], 1
    for i in range(1, len(c)):
        fu = up[i] if (up[i] < fu or c[i - 1] > fu) else fu
        fd = dn[i] if (dn[i] > fd or c[i - 1] < fd) else fd
        if c[i] > fu: d = 1
        elif c[i] < fd: d = -1
        pos[i] = d
    return pd.Series(pos, df.index)


# --- mean reversion ---
def bollinger_mr(df, n, k):
    m, sd = df.close.rolling(n).mean(), df.close.rolling(n).std()
    c = df.close
    return legs(df.index, c < m - k * sd, c >= m, c > m + k * sd, c <= m)


def rsi_mr(df, n, lo):
    r = rsi(df.close, n)
    return legs(df.index, r < lo, r > 50, r > 100 - lo, r < 50)


# --- positioning ---
def funding_contra(df, n, z):
    """Fade crowded positioning: very high funding -> short, very negative -> long."""
    f = df.funding.where(df.funding != 0).ffill()
    zs = (f - f.rolling(n * 8).mean()) / f.rolling(n * 8).std()
    return legs(df.index, zs < -z, zs > 0, zs > z, zs < 0)


def buy_hold(df):
    return pd.Series(1.0, df.index)


# name -> (fn, grid). Grids stay coarse on purpose: fine grids just fit noise.
STRATS = {
    "sma_cross":      (sma_cross,      dict(fast=[12, 24, 48, 96], slow=[96, 168, 336, 504])),
    "ema_cross":      (ema_cross,      dict(fast=[12, 24, 48, 96], slow=[96, 168, 336, 504])),
    "macd":           (macd,           dict(fast=[12, 24, 48], slow=[26, 72, 168], sig=[9, 24])),
    "tsmom":          (tsmom,          dict(look=[24, 72, 168, 336, 720])),
    "donchian":       (donchian,       dict(entry=[24, 72, 168, 336], exit=[12, 24, 72, 168])),
    "supertrend":     (supertrend,     dict(n=[10, 24, 72], mult=[2, 3, 4, 6])),
    "bollinger_mr":   (bollinger_mr,   dict(n=[24, 48, 96, 168], k=[1.5, 2, 2.5, 3])),
    "rsi_mr":         (rsi_mr,         dict(n=[6, 14, 24], lo=[15, 20, 25, 30])),
    "funding_contra": (funding_contra, dict(n=[7, 14, 30], z=[1, 1.5, 2, 2.5])),
}


def adx(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / a
    return (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / n, adjust=False).mean()
