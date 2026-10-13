# Futures paper bot

PAPER trading only, no real orders. Daily 72-day momentum + 4h MACD on Kraken perpetuals (BTC/ETH/SOL),
$100k paper account, 2% risk per trade, 5% stop, 15% target, max 7 days per trade, breakeven after +7.5%.

GitHub Actions runs `live_bot.py` every hour (`.github/workflows/bot.yml`) and commits the updated
`paper_state.json`, `paper_events.csv`, `paper_equity.csv`, `paper_bot.log` and `dashboard.html`.
