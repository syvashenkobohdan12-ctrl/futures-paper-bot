"""Build backtests.xlsx from smc_results/ (branch smc-backtest). Summary stats are live formulas over the trade tabs."""
import re, sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

R = Path(__file__).parent
OUT = Path(sys.argv[1])
F = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1F3A5F")
SEC_FILL = PatternFill("solid", fgColor="E8EEF5")
PASS_FILL, FAIL_FILL, MIX_FILL = (PatternFill("solid", fgColor=c) for c in ("D9F2E3", "F9DEDE", "FFF2CC"))
thin = Side(style="thin", color="D0D4DA")
BORDER = Border(bottom=thin)
BLUE = Font(name=F, size=10, color="0000FF")
BLACK = Font(name=F, size=10)
BOLD = Font(name=F, size=10, bold=True)

wb = Workbook()
wb.remove(wb.active)


def header(ws, row, cols, widths=None):
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row, j, c)
        cell.font = Font(name=F, size=10, bold=True, color="FFFFFF")
        cell.fill = HDR_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 30
    if widths:
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w


def title(ws, text, note):
    ws["A1"] = text
    ws["A1"].font = Font(name=F, size=14, bold=True)
    ws["A2"] = note
    ws["A2"].font = Font(name=F, size=9, italic=True, color="666666")


def fmt_col(ws, col, fmt, r0, r1):
    for r in range(r0, r1 + 1):
        ws.cell(r, col).number_format = fmt


# ---------------- trade tabs (inputs, blue) ----------------
TRADE_COLS = ["Entry time", "Exit time", "Coin", "Side", "Entry", "Stop", "Target", "Exit", "Exit reason", "R", "R²", "Stop %"]


def trade_sheet(name, df, extra, note):
    ws = wb.create_sheet(name)
    title(ws, name, note)
    cols = TRADE_COLS + [e[0] for e in extra]
    header(ws, 4, cols, [17, 17, 11, 7, 12, 12, 12, 12, 11, 8, 8, 8] + [12] * len(extra))
    df = df.reset_index(drop=True)
    for i, t in df.iterrows():
        r = 5 + i
        tgt = t.tgt if pd.notna(t.tgt) and abs(t.tgt) < 1e12 else None
        vals = [t.ts.to_pydatetime(), t.exit_ts.to_pydatetime(), str(t.sym).replace("USDT", ""), "Long" if t.side > 0 else "Short",
                abs(t.entry), abs(t.stop), abs(tgt) if tgt is not None else "none", abs(t.exit), t.why, float(t.R), None, float(t.risk_pct)]
        for j, v in enumerate(vals, 1):
            c = ws.cell(r, j, v)
            c.font = BLUE
        ws.cell(r, 11, f"=J{r}^2").font = BLACK
        for j, (_, fn) in enumerate(extra, len(TRADE_COLS) + 1):
            ws.cell(r, j, fn(t)).font = BLUE
    last = 4 + len(df)
    fmt_col(ws, 1, "yyyy-mm-dd hh:mm", 5, last); fmt_col(ws, 2, "yyyy-mm-dd hh:mm", 5, last)
    for col in (5, 6, 7, 8):
        fmt_col(ws, col, "#,##0.0000", 5, last)
    fmt_col(ws, 10, "+0.00;-0.00;0.00", 5, last); fmt_col(ws, 11, "0.00", 5, last); fmt_col(ws, 12, "0.0%", 5, last)
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:{get_column_letter(len(cols))}{last}"
    return ws, last


def year_set(t):
    return str(pd.Timestamp(t.ts).year)


smc = pd.read_csv(R / "final_trades.csv", parse_dates=["ts", "exit_ts"])
ws_smc, n_smc = trade_sheet("SMC 3-coin trades", smc, [("Year", year_set)],
                            "Round 3 final SMC config on BTC/ETH/SOL 1h (k=3, BOS<=12 bars, entry gap top, 2R). Rules picked on 2024-25; 2026 untouched. Source: smc_results/final_trades.csv")

multi = pd.read_csv(R / "multi_trades.csv", parse_dates=["ts", "exit_ts"])
ws_mu, n_mu = trade_sheet("Multi-market trades", multi, [("Market", lambda t: t.group), ("TF", lambda t: t.tf),
                                                         ("Rules saw it?", lambda t: "yes" if t.seen else "no")],
                          "Round 4: frozen SMC rules on 10 crypto perps + 10 US stocks, 5m-1d. 'Rules saw it?' = yes only for BTC/ETH/SOL 1h 2024-25. Source: smc_results/multi_trades.csv")

lab = pd.read_csv(R / "lab_trades.csv", parse_dates=["ts", "exit_ts"])
ws_lab, n_lab = trade_sheet("SMC lab trades", lab, [("Set", lambda t: t.set)],
                            "Round 5 final lab config (maker fills, 1.5R, breakeven at 1R, no filters). dev = 6 coins 2022-25; hold_A = ADA/AVAX/LINK/SUI; hold_B = 2026. Source: smc_results/lab_trades.csv")

pa = pd.concat([pd.read_csv(f, parse_dates=["ts", "exit_ts"]) for f in sorted(R.glob("pa_trades_*.csv"))], ignore_index=True)
ws_pa, n_pa = trade_sheet("Price-action trades", pa, [("Setup", lambda t: t.setup), ("TF", lambda t: t.tf), ("Set", lambda t: t.set)],
                          "Round 6: the DEV-best version of each of 7 price-action setups. Source: smc_results/pa_trades_*.csv")

don = pd.read_csv(R / "donchian_trades.csv", parse_dates=["ts", "exit_ts"])
ws_don, n_don = trade_sheet("Donchian trades", don, [("Set", lambda t: t.set), ("Year", year_set)],
                            "Round 7: live sleeve rules (daily 20-day breakout, long only, 2xATR stop, 3R or 72 days). Source: smc_results/donchian_trades.csv")

# ---------------- summary ----------------
ws = wb.create_sheet("Summary", 0)
title(ws, "Backtest log: liquidity sweep → BOS → FVG, then price action",
      "Crypto perps (Binance data) unless noted. R = profit per trade in units of the amount risked, net of fees and funding. "
      "Blue = inputs typed from CI reports; black = formulas over the trade tabs. Noise p95 = 95th percentile of the same search run on random walks.")
cols = ["Round", "Experiment", "Rules / variant", "Sample", "Trades", "Win rate", "Avg R", "t-stat", "Noise p95 (avg R)", "Verdict"]
header(ws, 4, cols, [7, 30, 40, 34, 9, 9, 9, 9, 12, 46])
row = [5]


def section(text):
    r = row[0]
    ws.cell(r, 1, text).font = BOLD
    for j in range(1, len(cols) + 1):
        ws.cell(r, j).fill = SEC_FILL
    row[0] += 1


def put(rnd, exp, rule, sample, sheet, last, crit, noise=None, verdict="", tone=None, typed=None, note=None):
    """crit: list of (column letter, criterion string) for COUNTIFS on the trade sheet."""
    r = row[0]
    for j, v in enumerate([rnd, exp, rule, sample], 1):
        ws.cell(r, j, v).font = BLACK
    if typed:                                            # numbers only exist in a CI log: typed inputs
        for j, v in zip((5, 6, 7), typed):
            ws.cell(r, j, v).font = BLUE
        ws.cell(r, 8, "n/a").font = BLACK
        ws.cell(r, 5).comment = Comment(note, "Claude")
    else:
        q = f"'{sheet}'!"
        rng = lambda col: f"{q}${col}$5:${col}${last}"
        args = "".join(f",{rng(c)},{v}" for c, v in crit)
        ws.cell(r, 5, f"=COUNTIFS({rng('J')},\"<>\"{args})")
        ws.cell(r, 6, f"=IFERROR(COUNTIFS({rng('J')},\">0\"{args})/E{r},\"-\")")
        ws.cell(r, 7, f"=IFERROR(SUMIFS({rng('J')}{args})/E{r},\"-\")")
        ws.cell(r, 8, f"=IFERROR(G{r}/(SQRT((SUMIFS({rng('K')}{args})-E{r}*G{r}^2)/(E{r}-1))/SQRT(E{r})),\"-\")")
        for j in (5, 6, 7, 8):
            ws.cell(r, j).font = BLACK
    if noise is not None:
        c = ws.cell(r, 9, noise); c.font = BLACK if isinstance(noise, str) and noise.startswith("=") else BLUE
    ws.cell(r, 10, verdict).font = BLACK
    ws.cell(r, 10).alignment = Alignment(wrap_text=True, vertical="top")
    if tone:
        ws.cell(r, 10).fill = {"pass": PASS_FILL, "fail": FAIL_FILL, "mixed": MIX_FILL}[tone]
    ws.cell(r, 5).number_format = "#,##0"; ws.cell(r, 6).number_format = "0%"
    ws.cell(r, 7).number_format = "+0.00;-0.00;0.00"; ws.cell(r, 8).number_format = "+0.00;-0.00;0.00"
    ws.cell(r, 9).number_format = "+0.00;-0.00;0.00"
    for j in range(1, len(cols) + 1):
        ws.cell(r, j).border = BORDER
        if j != 10:
            ws.cell(r, j).alignment = Alignment(vertical="top", wrap_text=j in (2, 3, 4))
    row[0] += 1


S = lambda y0, y1=None: [("A", f"\">=\"&DATE({y0},1,1)")] + ([("A", f"\"<\"&DATE({y1},1,1)")] if y1 else [])

section("Round 1 · SMC on ETHUSDT 1h (36-config grid)")
put(1, "Pre-chosen default", "k=5, BOS≤24 bars, entry gap middle, target next swing high", "2024-25 (tuning)", None, 0, [],
    typed=(15, 0.333, 0.507), note="Typed from CI run 1 log (smc.py ETHUSDT, 2026-10-14 16:24 UTC)")
put(1, "Pre-chosen default", "same", "2026 (unseen)", None, 0, [], typed=(10, 0.20, -0.424), note="Typed from CI run 1 log",
    verdict="Fails: positive when tuned, negative on unseen 2026. 35/36 configs positive in 2024-25, 31/36 negative in 2026.", tone="fail")

section("Round 2 · SMC on BTC+ETH+SOL 1h (128-config grid, see 'SMC grid')")
put(2, "Best single config on 2024-25", "k=3, BOS≤24, gap top, swing-high target, any recent low", "2024-25 (tuning)", None, 0, [],
    typed=(80, 0.50, 0.327), note="Typed from CI run 2 report (smc_results/report.txt)")
put(2, "Best single config on 2024-25", "same", "2026 (unseen)", None, 0, [], typed=(45, 0.356, -0.140), note="Typed from CI run 2 report",
    verdict="Fails out of sample. The family k=3 + untouched gap + 2R target was positive in 16/16 variants in both periods → Round 3.", tone="fail")

section("Round 3 · SMC final config, BTC+ETH+SOL 1h ('SMC 3-coin trades')")
for y in (2024, 2025):
    put(3, "Family pick", "k=3, BOS≤12, gap top, 2R, any recent low, untouched gap", str(y), "SMC 3-coin trades", n_smc, S(y, y + 1))
put(3, "Family pick", "same", "2026 (unseen)", "SMC 3-coin trades", n_smc, S(2026),
    verdict="Looked good (+0.12R in 2026, beat 84% of random entries) but t≈0.5 in 2026; only 30 trades/yr. Not proven.", tone="mixed")
put(3, "Family pick", "same", "All 2024-26", "SMC 3-coin trades", n_smc, S(2000))

section("Round 4 · Same frozen rules on more markets and timeframes ('Multi-market trades', data the rules never saw)")
for g in ("crypto", "stocks"):
    for tf in ("5m", "15m", "1h", "4h", "1d"):
        crit = [("M", f"\"{g}\""), ("N", f"\"{tf}\""), ("O", "\"no\"")]
        v = {("crypto", "5m"): ("Fees eat it: gross ≈ 0, net −0.32R", "fail"), ("crypto", "15m"): ("Fees eat it: gross +0.03, net −0.14R", "fail"),
             ("crypto", "1h"): ("Only survivor: +0.04R net, but −0.19R in 2022, −0.03R in 2023", "mixed"),
             ("crypto", "4h"): ("≈ 0, too few trades", "fail"), ("crypto", "1d"): ("Too few trades (25)", "fail")}.get((g, tf), ("Negative / too little data (Yahoo history only)", "fail"))
        put(4, f"{g.title()} {tf}", "Round 3 rules, unchanged", "All years, unseen data", "Multi-market trades", n_mu, crit, verdict=v[0], tone=v[1])

section("Round 5 · 'Make it work' lab: execution, exits, 11 filters, dev/holdout protocol ('SMC lab trades', 'SMC lab steps')")
NULL_SMC = "=PERCENTILE('Noise tests'!$I$6:$I$45,0.95)"
for s, lab_s in (("dev", "DEV: BTC ETH SOL XRP BNB DOGE 2022-25"), ("hold_A", "Holdout A: ADA AVAX LINK SUI 2022-25"), ("hold_B", "Holdout B: all 10 coins, 2026")):
    put(5, "Lab pick (DEV only)", "maker entries, 1.5R, breakeven at 1R, no filter survived", lab_s, "SMC lab trades", n_lab, [("M", f"\"{s}\"")],
        noise=NULL_SMC if s == "hold_B" else None,
        verdict="No edge: holdouts combined +0.06R vs noise p95 +0.14R (both holdouts pooled)" if s == "hold_B" else "", tone="fail" if s == "hold_B" else None)

section("Round 6 · Seven classic price-action setups, each at its DEV-best tf/params ('Price-action trades', 'Price action')")
names = {"pd_sweep": "Prev-day high/low sweep", "inside": "Inside-bar breakout", "pin": "Pin bar at swing", "engulf": "Engulfing at swing",
         "retest": "Breakout-retest", "structure": "HH/HL structure break", "donchian": "Donchian 20-day breakout"}
psum = pd.read_csv(R / "pa_summary.csv")
null_rows = {}
for _, p in psum.iterrows():
    rule = f"{p.tf}, {p.params if p.params != '{}' else 'no params'}, target {p.tp}R"
    for s, lab_s in (("dev", "DEV 6 coins (1d from 2020)"), ("hold", "Both holdouts")):
        crit = [("M", f"\"{p.setup}\"")] + ([("O", "\"dev\"")] if s == "dev" else [("O", "\"<>dev\"")])
        verdict, tone = "", None
        if s == "hold":
            verdict, tone = (("Passes: beats its noise p95 on dev t, holdout R and holdout t", "pass") if p.setup == "donchian" else
                             ("Fails: within noise range", "fail"))
        put(6, names[p.setup], rule, lab_s, "Price-action trades", n_pa, crit,
            verdict=verdict, tone=tone)
        if s == "hold":
            null_rows[p.setup] = row[0] - 1

section("Round 7 · Donchian sleeve, final rules (now paper trading on main)")
for s, lab_s in (("dev", "DEV 6 coins 2020-25"), ("hold_A", "Holdout A coins 2020-25"), ("hold_B", "Holdout B: 2026")):
    put(7, "Donchian, long only, 3R", "daily close > prior 20-day high; stop 2×ATR; 3R or 72 days", lab_s, "Donchian trades", n_don, [("M", f"\"{s}\"")],
        verdict={"dev": "DEV t well above noise p95 (1.60)", "hold_A": "Holds up on unseen coins",
                 "hold_B": "Flat in 2026 (falling market, long only). Forward test live since 2026-10-14."}[s],
        tone={"dev": "pass", "hold_A": "pass", "hold_B": "mixed"}[s])
ws.freeze_panes = "C5"
summary_last = row[0] - 1

# ---------------- SMC grid ----------------
g = pd.read_csv(R / "grid.csv")
wg = wb.create_sheet("SMC grid")
title(wg, "SMC grid · 128 configs × BTC/ETH/SOL 1h", "Round 2. train = 2024-25 (used to choose), test = 2026. Source: smc_results/grid.csv")
gcols = ["k", "max_bos", "entry", "tp", "liq", "fresh", "trend", "train_n", "train_win", "train_avgR", "train_pf", "train_maxdd_1pct",
         "test_n", "test_win", "test_avgR", "test_pf", "test_maxdd_1pct"]
labels = ["Swing k", "BOS window", "Entry", "Target", "Liquidity", "Untouched gap", "4h trend filter", "Train trades", "Train win", "Train avg R",
          "Train PF", "Train max DD (1%)", "Test trades", "Test win", "Test avg R", "Test PF", "Test max DD (1%)"]
header(wg, 4, labels, [8, 10, 8, 8, 10, 12, 12] + [11] * 10)
g = g.sort_values("train_avgR", ascending=False).reset_index(drop=True)
for i, rr in g.iterrows():
    for j, c in enumerate(gcols, 1):
        v = rr[c]
        v = (str(v) if c in ("tp", "fresh", "trend") else v)
        cell = wg.cell(5 + i, j, None if pd.isna(v) else (float(v) if isinstance(v, (int, float)) and c not in ("k", "max_bos") else v))
        cell.font = BLUE
        if c.endswith("win") or c.endswith("maxdd_1pct"):
            cell.number_format = "0%"
        elif c.endswith("avgR") or c.endswith("pf"):
            cell.number_format = "+0.00;-0.00;0.00"
r0 = 5 + len(g) + 1
wg.cell(r0, 1, "Share of configs with positive avg R").font = BOLD
wg.cell(r0, 10, f"=COUNTIF(J5:J{4 + len(g)},\">0\")/COUNT(J5:J{4 + len(g)})").number_format = "0%"
wg.cell(r0, 15, f"=COUNTIF(O5:O{4 + len(g)},\">0\")/COUNT(O5:O{4 + len(g)})").number_format = "0%"
wg.freeze_panes = "H5"; wg.auto_filter.ref = f"A4:Q{4 + len(g)}"

# ---------------- multi coverage + summary ----------------
wm = wb.create_sheet("Multi-market")
title(wm, "Round 4 · coverage and summary", "Source: smc_results/multi_summary.csv and multi_data.csv. Gross = before fees; fee R = median round-trip fees in R.")
ms = pd.read_csv(R / "multi_summary.csv")
mcols = ["group", "tf", "n", "win", "avgR", "t", "fee_R", "n2026", "avgR2026", "rand_avgR", "beats_random", "syms_pos"]
mlabels = ["Market", "TF", "Trades", "Win", "Avg R", "t-stat", "Fees (R)", "2026 trades", "2026 avg R", "Random-entry avg R", "Beats random", "Symbols positive"]
header(wm, 4, mlabels + ["Avg R before fees"], [9, 6, 9, 8, 9, 9, 9, 10, 10, 12, 11, 11, 12])
for i, rr in ms.iterrows():
    r = 5 + i
    for j, c in enumerate(mcols, 1):
        v = rr[c]
        cell = wm.cell(r, j, v if isinstance(v, str) else float(v)); cell.font = BLUE
        cell.number_format = {"win": "0%", "beats_random": "0%"}.get(c, "+0.00;-0.00;0.00" if c in ("avgR", "t", "fee_R", "avgR2026", "rand_avgR") else "General")
    wm.cell(r, 13, f"=E{r}+G{r}").number_format = "+0.00;-0.00;0.00"
cov = pd.read_csv(R / "multi_data.csv")
r = 5 + len(ms) + 2
wm.cell(r, 1, "Data coverage").font = BOLD
header(wm, r + 1, ["Symbol", "TF", "Market", "Source", "Bars", "First", "Last"])
for i, rr in cov.iterrows():
    for j, c in enumerate(["sym", "tf", "group", "src", "bars", "first", "last"], 1):
        v = rr.get(c); wm.cell(r + 2 + i, j, None if pd.isna(v) else v).font = BLUE

# ---------------- lab steps ----------------
wl = wb.create_sheet("SMC lab steps")
title(wl, "Round 5 · every step of the lab protocol", "Each choice made on DEV only; holdouts printed for transparency. Source: smc_results/lab_report.txt")
header(wl, 4, ["Step", "Variant", "DEV trades", "DEV avg R", "DEV t", "Hold-A trades", "Hold-A avg R", "Hold-A t", "Hold-B trades", "Hold-B avg R", "Hold-B t"],
       [44, 34, 10, 10, 9, 11, 11, 9, 11, 11, 9])
pat = re.compile(r"^(?P<label>.+?)\s+\| dev n=\s*(?P<dn>\d+) avgR=(?P<dR>[-+.\d]+) t=(?P<dt>[-+.\d]+) \| hold_A n=\s*(?P<an>\d+) avgR=(?P<aR>[-+.\d]+) t=(?P<at>[-+.\d]+) \| hold_B n=\s*(?P<bn>\d+) avgR=(?P<bR>[-+.\d]+) t=(?P<bt>[-+.\d]+)")
sec, i = None, 0
for line in open(R / "lab_report.txt"):
    if line.startswith("== "):
        sec = line.strip("= \n")
    m = pat.match(line.rstrip())
    if m:
        vals = [sec, m["label"].strip()] + [float(m[k]) for k in ("dn", "dR", "dt", "an", "aR", "at", "bn", "bR", "bt")]
        for j, v in enumerate(vals, 1):
            c = wl.cell(5 + i, j, v); c.font = BLUE
            if j in (4, 5, 7, 8, 10, 11): c.number_format = "+0.000;-0.000;0.000"
        i += 1

# ---------------- price action ----------------
wp = wb.create_sheet("Price action")
title(wp, "Round 6 · seven price-action setups vs their noise thresholds",
      "Pick = best DEV t-stat over tf × params × target. Noise columns = 95th percentile of the same search on 40 random-walk universes ('Noise tests'). Source: pa_summary.csv, pa_null.csv")
header(wp, 4, ["Setup", "TF", "Params", "Target R", "DEV trades", "DEV avg R", "DEV t", "Hold-A avg R", "Hold-B avg R", "Holdouts avg R", "Holdouts t",
               "Noise p95 DEV t", "Noise p95 holdout R", "Noise p95 holdout t", "Beats noise on all three?"],
       [24, 6, 12, 9, 10, 10, 9, 11, 11, 12, 10, 12, 13, 13, 14])
nul = pd.read_csv(R / "pa_null.csv").sort_values(["setup", "u"]).reset_index(drop=True)
blocks = {}
for s, grp in nul.groupby("setup", sort=False):
    blocks[s] = (6 + grp.index.min(), 6 + grp.index.max())    # rows on 'Noise tests' PA block (offset below)
PA_OFF = 50
for i, p in psum.iterrows():
    r = 5 + i
    vals = [names[p.setup], p.tf, p.params, float(p.tp), int(p.dev_n), p.dev_R, p.dev_t, p.A_R, p.B_R, p.hold_R, p.hold_t]
    for j, v in enumerate(vals, 1):
        c = wp.cell(r, j, v); c.font = BLUE
        if j >= 6: c.number_format = "+0.000;-0.000;0.000"
    a, b = blocks[p.setup]
    for j, col in ((12, "G"), (13, "N"), (14, "O")):
        c = wp.cell(r, j, f"=PERCENTILE('Noise tests'!${col}${a + PA_OFF}:${col}${b + PA_OFF},0.95)")
        c.number_format = "+0.000;-0.000;0.000"
    wp.cell(r, 15, f"=IF(AND(G{r}>L{r},J{r}>M{r},K{r}>N{r}),\"yes\",\"no\")")
# patch Summary noise refs for Round 6
setup_row = {p.setup: 5 + i for i, p in psum.iterrows()}
for s, srow in null_rows.items():
    ws.cell(srow, 9, f"='Price action'!M{setup_row[s]}").font = Font(name=F, size=10, color="008000")

# ---------------- donchian variants ----------------
wd = wb.create_sheet("Donchian variants")
title(wd, "Round 7 · exit and side variants of the daily Donchian breakout",
      "Choice rule: highest DEV t-stat (picked: fixed 3R, long only). Source: smc_results/donchian_report.txt, donchian_null.csv")
header(wd, 4, ["Exit", "Sides", "DEV trades", "DEV avg R", "DEV t", "Hold-A avg R", "Hold-B avg R", "Holdouts avg R", "Holdouts t", "Picked?"],
       [10, 8, 10, 10, 9, 11, 11, 12, 10, 9])
pat2 = re.compile(r"^(\w+)\s+(\w+)\s+\| dev n=\s*(\d+) R=([-+.\d]+) t=([-+.\d]+) \| hold_A R=([-+.\d]+) \| hold_B R=([-+.\d]+) \| holdouts R=([-+.\d]+) t=([-+.\d]+)")
var = [pat2.match(l).groups() for l in open(R / "donchian_report.txt") if pat2.match(l)]
for i, v in enumerate(var):
    r = 5 + i
    for j, x in enumerate(v, 1):
        c = wd.cell(r, j, x if j <= 2 else float(x)); c.font = BLUE
        if j >= 4: c.number_format = "+0.000;-0.000;0.000"
    wd.cell(r, 10, f"=IF(E{r}=MAX($E$5:$E${4 + len(var)}),\"picked\",\"\")")
r = 5 + len(var) + 1
wd.cell(r, 1, "Noise p95 of the same choice (60 random-walk universes)").font = BOLD
for j, col in ((5, "E"), (8, "H"), (9, "I")):
    c = wd.cell(r, j, f"=PERCENTILE('Noise tests'!{col}$96:{col}$155,0.95)"); c.number_format = "+0.000;-0.000;0.000"
wd.cell(r + 2, 1, "Yearly result of the picked rules (formulas over 'Donchian trades')").font = BOLD
header(wd, r + 3, ["Year", "Trades", "Avg R"])
for k, y in enumerate(range(2020, 2027)):
    rr = r + 4 + k
    wd.cell(rr, 1, str(y))
    wd.cell(rr, 2, f"=COUNTIFS('Donchian trades'!$N$5:$N${n_don},A{rr})")
    c = wd.cell(rr, 3, f"=IFERROR(AVERAGEIFS('Donchian trades'!$J$5:$J${n_don},'Donchian trades'!$N$5:$N${n_don},A{rr}),\"-\")"); c.number_format = "+0.00;-0.00;0.00"

# ---------------- noise tests (raw) ----------------
wn = wb.create_sheet("Noise tests")
title(wn, "Noise tests: the same searches run on random-walk markets (no edge exists)",
      "A real result has to beat these distributions, not zero. Rows 6-45: SMC lab (40 universes); rows 56+: price action (40 × 7); rows 96-155 cols A-J: Donchian variant choice (60). Inputs from smc_results/*null*.csv")
sl = pd.read_csv(R / "null.csv")
header(wn, 5, ["Universe", "Filters picked", "DEV t", "DEV avg R", "Hold-A avg R", "Hold-A t", "Hold-B avg R", "Hold-B t", "Holdouts avg R", "Holdout trades"])
for i, rr in sl.iterrows():
    for j, c in enumerate(["u", "nf", "dev_t", "dev_R", "A_R", "A_t", "B_R", "B_t", "AB_R", "AB_n"], 1):
        wn.cell(6 + i, j, float(rr[c])).font = BLUE
header(wn, 55, ["Setup", "TF", "Params", "Target", "DEV n", "DEV avg R", "DEV t", "A n", "A avg R", "A t", "B n", "B avg R", "B t", "Holdouts avg R", "Holdouts t", "Universe"])
for i, rr in nul.iterrows():
    for j, c in enumerate(["setup", "tf", "params", "tp", "dev_n", "dev_R", "dev_t", "A_n", "A_R", "A_t", "B_n", "B_R", "B_t", "hold_R", "hold_t", "u"], 1):
        v = rr[c]
        wn.cell(56 + i, j, v if isinstance(v, str) else (None if pd.isna(v) else float(v))).font = BLUE
dn = pd.read_csv(R / "donchian_null.csv")
assert 56 + len(nul) <= 400
D0 = 56 + len(nul) + 3
wn.cell(D0 - 1, 1, "Donchian variant choice on random walks").font = BOLD
header(wn, D0, ["Exit", "Sides", "DEV n", "DEV avg R", "DEV t", "Hold-A avg R", "Hold-B avg R", "Holdouts avg R", "Holdouts t", "Universe"])
for i, rr in dn.iterrows():
    for j, c in enumerate(["exit", "sides", "dev_n", "dev_R", "dev_t", "A_R", "B_R", "hold_R", "hold_t", "u"], 1):
        v = rr[c]
        wn.cell(D0 + 1 + i, j, v if isinstance(v, str) else float(v)).font = BLUE
# point Donchian noise formulas at the real block
for j, col in ((5, "E"), (8, "H"), (9, "I")):
    wd.cell(5 + len(var) + 1, j, f"=PERCENTILE('Noise tests'!{col}${D0 + 1}:{col}${D0 + len(dn)},0.95)").number_format = "+0.000;-0.000;0.000"
# price-action blocks start at row 56 (header 55): blocks were computed with base 6, PA_OFF=50
for ws_ in wb.worksheets:                     # one professional font everywhere
    for rr in ws_.iter_rows():
        for c in rr:
            if c.value is not None and c.font.name != F:
                c.font = c.font.copy(name=F, size=c.font.size or 10)
wb.save(OUT)
print("saved", OUT, "summary rows", summary_last)
