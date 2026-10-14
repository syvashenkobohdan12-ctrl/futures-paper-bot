"""Bars at any timeframe for the SMC study, cached in ./cache.
Crypto: Binance USD-M perp klines + funding from data.binance.vision (5m/15m/1h downloaded, 4h/1d resampled from 1h).
Stocks: Alpaca IEX bars when ALPACA_PAPER_KEY_ID/_SECRET_KEY are set, else Yahoo Finance
(Yahoo only serves ~730 days of 1h and 60 days of 5m/15m, so those come back short)."""
import datetime as dt, os
import pandas as pd, requests
from data import BASE, CACHE, DEFAULT_FUNDING, _get_csv as _raw_csv


def _get_csv(url, tries=5):
    """Older Binance files have no header row, newer ones do: number the columns so they line up when concatenated.
    Retries network errors (runners occasionally fail DNS) with backoff."""
    import time
    for k in range(tries):
        try:
            df = _raw_csv(url)
            break
        except requests.exceptions.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    if df is not None:
        df.columns = range(df.shape[1])
    return df

AGG = dict(open="first", high="max", low="min", close="last", volume="sum", funding="sum")
START = {"5m": "2024-01-01", "15m": "2024-01-01", "1h": "2022-01-01", "4h": "2022-01-01", "1d": "2020-01-01"}


def _months(start, today):
    m = pd.Timestamp(start).date().replace(day=1)
    while m <= today:
        yield m
        m = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def _cached(name, build):
    path = CACHE / name
    if path.exists() and dt.date.fromtimestamp(path.stat().st_mtime) == dt.date.today():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = build()
    CACHE.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def _binance_klines(sym, iv, start):
    today = dt.date.today()
    parts = []
    for m in _months(start, today):
        df = _get_csv(f"{BASE}/monthly/klines/{sym}/{iv}/{sym}-{iv}-{m:%Y-%m}.zip")
        if df is None and (today - m).days < 62:      # recent month not packed yet: daily files
            for d in pd.date_range(m, min(today - dt.timedelta(days=1), (m + dt.timedelta(days=31)).replace(day=1) - dt.timedelta(days=1))):
                dd = _get_csv(f"{BASE}/daily/klines/{sym}/{iv}/{sym}-{iv}-{d:%Y-%m-%d}.zip")
                if dd is not None:
                    parts.append(dd)
        elif df is not None:
            parts.append(df)
    df = pd.concat(parts).iloc[:, :6].astype(float)
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms")
    return df[~df.index.duplicated()].sort_index()


def _binance_funding(sym, start):
    today = dt.date.today()
    parts = [p for m in _months(start, today)
             if (p := _get_csv(f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")) is not None]
    df = pd.concat(parts)
    f = pd.Series(df.iloc[:, 2].astype(float).values,
                  index=pd.to_datetime(df.iloc[:, 0].astype("int64"), unit="ms").dt.floor("h"), name="funding")
    return f[~f.index.duplicated()].sort_index().to_frame()


def crypto(sym, tf):
    start = START[tf]
    iv = tf if tf in ("5m", "15m") else "1h"
    k = _cached(f"smc_{sym}_{iv}_{start}.csv", lambda: _binance_klines(sym, iv, start))
    f = _cached(f"smc_{sym}_funding_{START['1d']}.csv", lambda: _binance_funding(sym, START["1d"])).iloc[:, 0]
    k["funding"] = f.reindex(k.index).fillna(0.0)
    # shortcut: months Binance hasn't published funding for yet get the baseline rate every 8h
    gap = (k.index > f.index[-1]) & (k.index.hour % 8 == 0) & (k.index.minute == 0)
    k.loc[gap, "funding"] = DEFAULT_FUNDING
    if tf in ("4h", "1d"):
        k = k.resample(tf).agg(AGG).dropna()
    return k


def _alpaca(sym, tf, start):
    tfmap = {"5m": "5Min", "15m": "15Min", "1h": "1Hour", "4h": "1Hour", "1d": "1Day"}
    hdr = {"APCA-API-KEY-ID": os.environ["ALPACA_PAPER_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_PAPER_SECRET_KEY"]}
    rows, token = [], None
    while True:
        q = dict(timeframe=tfmap[tf], start=f"{start}T00:00:00Z", limit=10000, feed="iex", adjustment="all")
        if token:
            q["page_token"] = token
        r = requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/bars", params=q, headers=hdr, timeout=60)
        r.raise_for_status()
        j = r.json()
        rows += j.get("bars") or []
        token = j.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows).rename(columns=dict(o="open", h="high", l="low", c="close", v="volume"))
    df.index = pd.to_datetime(df.pop("t")).dt.tz_convert("America/New_York")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    if tf != "1d":   # regular session only: IEX prints sparse pre/after-market bars
        df = df.between_time("09:30" if tf in ("5m", "15m") else "09:00", "15:59")
    df.index = df.index.tz_convert(None)
    return df


def _yahoo(sym, tf, start):
    import yfinance as yf
    period = {"5m": "60d", "15m": "60d", "1h": "730d", "4h": "730d"}.get(tf)
    iv = {"4h": "1h"}.get(tf, tf)
    kw = dict(period=period) if period else dict(start=start)
    df = yf.Ticker(sym).history(interval=iv, auto_adjust=True, **kw)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype(float)
    df.index = pd.DatetimeIndex(df.index).tz_convert(None)
    return df


def stock(sym, tf):
    src = "alpaca" if os.environ.get("ALPACA_PAPER_KEY_ID") else "yahoo"
    start = START[tf]
    base_tf = "1h" if tf == "4h" else tf
    df = _cached(f"smc_{sym}_{base_tf}_{src}_{start}.csv",
                 lambda: (_alpaca if src == "alpaca" else _yahoo)(sym, base_tf, start))
    df["funding"] = 0.0
    if tf == "4h":   # two bars a session: open-13:00 and 13:00-close New York, stamped at their first hourly bar
        ny = df.index.tz_localize("UTC").tz_convert("America/New_York")
        key = pd.Series(ny.normalize().tz_localize(None) + pd.to_timedelta((ny.hour >= 13) * 13, unit="h"), index=df.index)
        first = pd.Series(df.index, index=df.index).groupby(key.values).first()
        df = df.groupby(key.values).agg(AGG)
        df.index = pd.DatetimeIndex(first.loc[df.index].values)
        df = df.dropna()
    return df, src


def load(sym, tf):
    """-> (df, source). Crypto symbols end in USDT."""
    if sym.endswith("USDT"):
        return crypto(sym, tf), "binance"
    return stock(sym, tf)
