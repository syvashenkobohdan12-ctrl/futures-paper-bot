"""Download Binance USD-M perp 1h klines + funding from data.binance.vision (public, no key) into ./cache."""
import io, zipfile, datetime as dt
from pathlib import Path
import pandas as pd, requests

BASE = "https://data.binance.vision/data/futures/um"
CACHE = Path(__file__).parent / "cache"
START = dt.date(2024, 1, 1)  # warmup before 2026 (daily bars need ~500 days)
DEFAULT_FUNDING = 0.0001  # Binance baseline 0.01%/8h, used where funding files aren't published yet


def _get_csv(url):
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if not str(df.iloc[0, 0]).isdigit():  # newer files carry a header row
        df.columns = df.iloc[0]; df = df.iloc[1:]
    return df


def _months(today):
    m = START.replace(day=1)
    while m <= today:
        yield m
        m = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def klines(sym, today=None):
    today = today or dt.date.today()
    path = CACHE / f"{sym}_1h.csv"
    if path.exists() and dt.date.fromtimestamp(path.stat().st_mtime) == today:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    parts = []
    for m in _months(today):
        df = _get_csv(f"{BASE}/monthly/klines/{sym}/1h/{sym}-1h-{m:%Y-%m}.zip")
        if df is None:  # current month: daily files
            for d in pd.date_range(m, today - dt.timedelta(days=1)):
                dd = _get_csv(f"{BASE}/daily/klines/{sym}/1h/{sym}-1h-{d:%Y-%m-%d}.zip")
                if dd is not None:
                    parts.append(dd)
        else:
            parts.append(df)
    df = pd.concat(parts).iloc[:, :6].astype(float)
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms")
    df = df[~df.index.duplicated()].sort_index()
    CACHE.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def funding(sym, index, today=None):
    """Funding rate aligned to the hourly index (rate at settlement hours, 0 elsewhere)."""
    today = today or dt.date.today()
    path = CACHE / f"{sym}_funding.csv"
    if path.exists() and dt.date.fromtimestamp(path.stat().st_mtime) == today:
        f = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        parts = [p for m in _months(today)
                 if (p := _get_csv(f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")) is not None]
        df = pd.concat(parts)
        f = pd.Series(df.iloc[:, 2].astype(float).values,
                      index=pd.to_datetime(df.iloc[:, 0].astype("int64"), unit="ms").dt.floor("h"), name="funding")
        f = f[~f.index.duplicated()].sort_index()
        f.to_csv(path)
    out = f.reindex(index).fillna(0.0)
    # shortcut: unpublished recent months get the baseline rate every 8h; fetch fapi history if this period matters
    gap = (index > f.index[-1]) & (index.hour % 8 == 0)
    out[gap] = DEFAULT_FUNDING
    return out


def alpaca_hourly(sym, today=None):
    """US stock/ETF 1h bars from Alpaca's free IEX feed (split/dividend adjusted). Keys from ALPACA_PAPER_KEY_ID/_SECRET_KEY."""
    import os
    today = today or dt.date.today()
    path = CACHE / f"{sym}_1h.csv"
    if path.exists() and dt.date.fromtimestamp(path.stat().st_mtime) == today:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    hdr = {"APCA-API-KEY-ID": os.environ["ALPACA_PAPER_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_PAPER_SECRET_KEY"]}
    rows, token = [], None
    while True:
        q = dict(timeframe="1Hour", start=f"{START}T00:00:00Z", limit=10000, feed="iex", adjustment="all")
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
    df.index = pd.to_datetime(df.pop("t")).dt.tz_convert(None)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    CACHE.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def load(sym):
    if not sym.endswith("USDT"):  # stocks/ETFs: no funding (shortcut: short borrow fees and margin interest ignored)
        df = alpaca_hourly(sym)
        df["funding"] = 0.0
        return df
    df = klines(sym)
    df["funding"] = funding(sym, df.index)
    return df


if __name__ == "__main__":
    for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT"]:
        d = load(s)
        print(s, d.index[0], d.index[-1], len(d), "funding sum", round(d.funding.sum(), 4))
