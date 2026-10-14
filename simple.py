"""The simple strategy. Two rules, daily closes, long only:
  BUY  when the close is the highest close of the last 20 days
  SELL when the close is the lowest close of the last 10 days
Each of the 10 coins gets an equal 10% slice of the account. No stops, no targets, no indicators.
Backtest: Binance daily bars 2020-now, 0.07% cost per buy/sell, funding paid while holding.
Usage: python simple.py"""
import numpy as np, pandas as pd

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
BUY_DAYS, SELL_DAYS, COST = 20, 10, 0.0007


def position(close):
    """1 while holding, 0 while flat; decided at each daily close."""
    hi = close.rolling(BUY_DAYS).max()
    lo = close.rolling(SELL_DAYS).min()
    pos, held = [], 0
    for c, h, l in zip(close, hi, lo):
        if not held and c >= h:
            held = 1
        elif held and c <= l:
            held = 0
        pos.append(held)
    return pd.Series(pos, close.index)


def backtest(daily):
    """daily: {coin: df with close, funding}. Returns (daily portfolio return, trades list)."""
    rets, trades = [], []
    for coin, d in daily.items():
        pos = position(d.close).shift(1).fillna(0)          # act on the next day
        r = d.close.pct_change().fillna(0)
        cost = pos.diff().abs().fillna(0) * COST
        rets.append((pos * r - cost - pos * d.funding).rename(coin))
        held, entry = 0, None
        for ts, p, c in zip(pos.index, pos, d.close):
            if p and not held: entry, t0 = c, ts
            if not p and held: trades.append(dict(coin=coin, entry=t0, exit=ts, ret=c / entry - 1 - 2 * COST))
            held = p
    port = pd.concat(rets, axis=1).fillna(0).mean(axis=1)   # equal 1/10 slices; a coin not listed yet sits in cash
    return port, pd.DataFrame(trades)


def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, maxdd=(eq / eq.cummax() - 1).min(), sharpe=r.mean() / r.std() * np.sqrt(365))


if __name__ == "__main__":
    import smc_data
    daily = {c: smc_data.crypto(c, "1d") for c in COINS}
    port, trades = backtest(daily)
    hold = pd.concat([d.close.pct_change().rename(c) for c, d in daily.items()], axis=1).fillna(0).mean(axis=1)
    print(f"== simple: buy {BUY_DAYS}-day high close, sell {SELL_DAYS}-day low close, 10 coins equal weight, long only ==")
    print(f"{'':10s} {'strategy':>10s} {'buy&hold':>10s}")
    for y in sorted(set(port.index.year)):
        s, h = port[port.index.year == y], hold[hold.index.year == y]
        print(f"{y:<10d} {(1 + s).prod() - 1:>+10.0%} {(1 + h).prod() - 1:>+10.0%}")
    a, b = stats(port), stats(hold)
    print(f"{'per year':10s} {a['cagr']:>+10.0%} {b['cagr']:>+10.0%}")
    print(f"{'worst DD':10s} {a['maxdd']:>10.0%} {b['maxdd']:>10.0%}")
    print(f"{'sharpe':10s} {a['sharpe']:>10.2f} {b['sharpe']:>10.2f}")
    print(f"\ntrades {len(trades)} | won {(trades.ret > 0).mean():.0%} | avg trade {trades.ret.mean():+.1%} | "
          f"avg win {trades.ret[trades.ret > 0].mean():+.1%} | avg loss {trades.ret[trades.ret <= 0].mean():+.1%} | "
          f"median days held {(trades.exit - trades.entry).dt.days.median():.0f}")
    print("per coin avg trade:", trades.groupby("coin").ret.mean().map("{:+.1%}".format).to_dict())
    # robustness: neighbouring settings, not used to choose anything
    print("\nneighbouring settings (per year / worst DD):")
    for b_, s_ in [(10, 5), (20, 10), (30, 15), (55, 20)]:
        BUY_DAYS, SELL_DAYS = b_, s_
        p, _ = backtest(daily)
        st = stats(p)
        print(f"  buy {b_:2d} / sell {s_:2d}: {st['cagr']:+.0%} / {st['maxdd']:.0%} / sharpe {st['sharpe']:.2f}")
