adjusted.csv: Adjusted close prices — corrected for stock splits and dividends. This is what you use for return calculations. Rows = days, columns = Date + 915 tickers.
close.csv: Raw (unadjusted) close prices — the actual closing price on that day.
dateline.csv: Trading calendar — list of every valid trading day (2006–2025). Just dates, no header.
dv.csv: Dollar volume (turnover) — how much money (in CNY) was traded for each stock each day. Used for liquidity filtering.
eps.csv: Earnings per share — annual EPS data (small file, summary-level).
high.csv: Daily high prices.
in_univ.csv: Binary matrix (1/0) — was this ticker in the investable universe that year? No header row; columns align with tickers.csv order.
low.csv: Daily low prices.
mktcap.csv
open.csv: Opening prices.

p2b.csv: Price-to-book ratio — a valuation metric (stock price ÷ book value per share).
recm.csvAnalyst recommendation scores — consensus ratings from sell-side analysts.
tickers.csv: Maps each ticker code to its GICS sector code (e.g., 000001,40101010 = Ping An Bank → Banks). No header.
univ_h.csv: Same as in_univ.csv but with a header row (year, 000001, 000002, ...). This is the one your code actually uses.