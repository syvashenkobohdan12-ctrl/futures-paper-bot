# Futures paper bot

PAPER trading only, no real orders. Daily 72-day momentum + 4h MACD on Kraken perpetuals (BTC/ETH/SOL),
$100k paper account, 2% risk per trade, 5% stop, 15% target, max 7 days per trade, breakeven after +7.5%.

GitHub Actions runs `live_bot.py` every hour (`.github/workflows/bot.yml`) and commits the updated
`paper_state.json`, `paper_events.csv`, `paper_equity.csv`, `paper_bot.log` and `dashboard.html`.

## Breakout sleeve (separate paper account)

Two rules on daily closes, long only, 10 Kraken perps (BTC ETH SOL XRP BNB DOGE ADA AVAX LINK SUI), 10% of a $10k paper account each:
- **Buy** when the close is the highest close of the last 20 days.
- **Sell** when the close is the lowest close of the last 10 days.

Backtest 2020–2026 (Binance daily, fees + funding): +61%/yr, worst drawdown −34%, vs buy & hold +69%/yr and −76% (`simple.py`).
`donchian_bot.py` runs in the hourly job, rebuilds the trades since go-live from Kraken data each run, and writes `donchian.html`.

## Dip sleeve (separate paper account)

- **Buy** when a coin closes the day more than 8% below the previous close; **sell** at the next day's close.
- 10% of a $10k paper account per trade, at most 3 trades a day (the biggest drops), long only, same 10 coins.

Backtest 2020–2026 (`dip.py`): 58% of trades win, +1.96% per trade; with the cap +16%/yr, worst drawdown −15%.
Most of the profit came in 2020–21. `sleeves.py` fetches Kraken data once per hour and runs both sleeves.
Research history: branch `smc-backtest`.
