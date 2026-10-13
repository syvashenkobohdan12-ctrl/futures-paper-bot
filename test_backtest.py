"""Self-checks for the engine: no lookahead, costs charged, legs hold state."""
import numpy as np, pandas as pd
from backtest import pnl, stats
from strategies import legs

idx = pd.date_range("2026-01-01", periods=4, freq="h")
df = pd.DataFrame({"close": [100, 110, 121, 121.0], "funding": [0, 0, 0.01, 0]}, idx)
pos = pd.Series([0, 1, 1, 0.0], idx)
r = pnl(df, pos, cost=0.001)
# bar1: buy at close (fee only, no return yet); bar2: +10% minus 1% funding; bar3: flat price, exit fee
assert np.allclose(r.values, [0, -0.001, 0.10 - 0.01, -0.001]), r.values
assert stats(r)["maxdd"] < 0

l = legs(idx, [True, False, False, False], [False, False, True, False], [False] * 4, [False] * 4)
assert list(l) == [1, 1, 0, 0], list(l)
print("ok")

# double bottom: lows ~100 at bars 3 and 9, peak 110 between; breakout above 110 at bar 14 only
from patterns import scan, det_double
c = [105, 103, 101, 100, 104, 108, 110, 106, 102, 100.5, 103, 106, 108, 109, 112, 113]
d = pd.DataFrame({"close": c, "high": c, "low": c}, pd.date_range("2026-01-01", periods=len(c), freq="h"))
e = scan(d, 2, det_double, 0.01)
assert list(np.nonzero(e)[0]) == [14] and e[14] == 1, e
print("patterns ok")

# wave 3: low 100 -> high 120 (wave 1) -> 110 (50% pullback, wave 2) -> break of 120 enters long
from waves import zigzag_scan, setup_wave3
c = [104, 102, 100, 106, 112, 120, 116, 113, 110, 114, 117, 119, 121, 124]
d = pd.DataFrame({"close": c, "high": c, "low": c}, pd.date_range("2026-01-01", periods=len(c), freq="h"))
e, s, t = zigzag_scan(d, 2, setup_wave3, "38-62", 1.618)
assert list(np.nonzero(e)[0]) == [12] and s[12] == 110 and abs(t[12] - (110 + 1.618 * 20)) < 1e-9, (e, s, t)
print("waves ok")

# ADX gate: a trade signalled while ADX > 40 stays flat for its whole run; the next signal (ADX ok) trades
from combo import gate
i5 = pd.date_range("2026-01-01", periods=6, freq="h")
p = gate(pd.Series([1, 1, 1, -1, -1, 0.0], i5), pd.Series([100.0] * 6, i5), pd.Series([45, 30, 30, 30, 30, 30.0], i5), None, None, 40)
assert list(p) == [0, 0, 0, -1, -1, 0], list(p)
print("gate ok")

# swing exits: take profit at 3R fills at the target; time exit after max_hold hours
import risk
st = risk.new_state(10_000)
b = lambda o, h, l, c: {"BTC": dict(o=o, h=h, l=l, c=c, f=0.0)}
t0 = pd.Timestamp("2026-01-01")
risk.step(st, t0, b(100, 100, 100, 100), {"BTC|x": 1}, tp_r=3, max_hold=168)
ev = risk.step(st, t0 + pd.Timedelta(hours=1), b(101, 116, 100, 110), {"BTC|x": 1}, tp_r=3, max_hold=168)
assert ev[0]["why"] == "target" and abs(ev[0]["px"] - 115) < 1e-9, ev
st = risk.new_state(10_000)
risk.step(st, t0, b(100, 100, 100, 100), {"BTC|x": 1}, tp_r=3, max_hold=2)
risk.step(st, t0 + pd.Timedelta(hours=1), b(100, 101, 99, 100), {"BTC|x": 1}, tp_r=3, max_hold=2)
ev = risk.step(st, t0 + pd.Timedelta(hours=2), b(100, 101, 99, 100), {"BTC|x": 1}, tp_r=3, max_hold=2)
assert ev[0]["why"] == "time" and not st["open"], ev
print("swing exits ok")

# re-entry after a time exit: same signal -> reopened at the same close with a fresh stop
st = risk.new_state(10_000)
risk.step(st, t0, b(100, 100, 100, 100), {"BTC|x": 1}, max_hold=2, reenter=("time",))
risk.step(st, t0 + pd.Timedelta(hours=1), b(100, 101, 99, 100), {"BTC|x": 1}, max_hold=2, reenter=("time",))
ev = risk.step(st, t0 + pd.Timedelta(hours=2), b(104, 110, 104, 110), {"BTC|x": 1}, max_hold=2, reenter=("time",))
assert [e["event"] for e in ev] == ["close", "open"] and abs(st["open"]["BTC|x"]["sp"] - 104.5) < 1e-9, ev
print("re-entry ok")

# breakeven: after +1.5R (7.5%) the stop moves to entry +0.2%; a later dip exits there, not at -5%
st = risk.new_state(10_000)
risk.step(st, t0, b(100, 100, 100, 100), {"BTC|x": 1}, be_r=1.5)
risk.step(st, t0 + pd.Timedelta(hours=1), b(100, 108, 100, 107), {"BTC|x": 1}, be_r=1.5)
assert abs(st["open"]["BTC|x"]["sp"] - 100.2) < 1e-9, st["open"]
ev = risk.step(st, t0 + pd.Timedelta(hours=2), b(104, 104, 99, 99), {"BTC|x": 1}, be_r=1.5)
assert ev[0]["why"] == "stop" and abs(ev[0]["px"] - 100.2) < 1e-9, ev
print("breakeven ok")
