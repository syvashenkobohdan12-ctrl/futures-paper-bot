# Futures paper bot

PAPER trading only, no real orders. Daily 72-day momentum + 4h MACD on Kraken perpetuals (BTC/ETH/SOL),
$100k paper account, 2% risk per trade, 5% stop, 15% target, max 7 days per trade, breakeven after +7.5%.

GitHub Actions runs `live_bot.py` every hour (`.github/workflows/bot.yml`) and commits the updated
`paper_state.json`, `paper_events.csv`, `paper_equity.csv`, `paper_bot.log` and `dashboard.html`.

## Donchian breakout sleeve (separate paper account)

`donchian_bot.py` runs in the same hourly job as its own $10k paper account: daily close above the prior 20-day high,
enter at the next day's open, stop 2×ATR(14), target 3R or 72 days, long only, 0.5% risk per trade, max 10 open, on 10
Kraken perps (BTC ETH SOL XRP BNB DOGE ADA AVAX LINK SUI). It is stateless: every run replays the backtest simulator
(`donchian.trades`) on Kraken history and keeps trades opened since go-live, so the paper record is exactly what the
backtest does. Page: `donchian.html`. Research (dev/holdout protocol, noise tests): branch `smc-backtest`.
