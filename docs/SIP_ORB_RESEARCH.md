# Stocks-in-Play Opening-Range Breakout: research basis and limits

This file records why this candidate was chosen, what is borrowed from the published
work, what is my own adaptation, and what is NOT established. Nothing here is a claim
that the strategy is profitable on NSE.

## Where the idea comes from
"Opening Range Breakout for Stocks in Play" (Zarattini, Barbon, Aziz), as replicated in
a QuantConnect research post. As summarised there:

- Each day rank stocks by relative volume: first-5-minute volume today divided by the
  average first-5-minute volume of the previous 14 days. Trade only the top ones.
- Direction comes from the opening 5-minute bar (up bar -> buy stop at the range high,
  down bar -> sell stop at the range low). Stop-loss is a multiple of 14-day ATR.
  Exit at the stop or at the close.
- The replication used 1,000 liquid US stocks and reported, for 2016, a Sharpe of 2.396
  versus 0.836 for buy-and-hold SPY.

## Caveats reported alongside it (all from the same source and its discussion)
- The headline result covers one year; other years looked weaker; overfitting was raised.
- No commission or slippage assumptions were stated; fill quality was questioned.
- A commenter reported a low win rate (about 17%) offset by large winners, so most trades
  end at the stop.
- Results were described as regime sensitive.
- The post did not state the ATR stop multiple, and the Python port omitted a filter and
  gave different results.

## What is mine (adaptation to this project, not from the paper)
- Universe: ~50 liquid NSE stocks from `config/watchlist.yaml`, not 1,000 US stocks.
- Defaults `min_rvol = 1.5`, `top_n = 5`, `min_price`, `min_atr_pct`, and the stop grid
  (0.10 / 0.25 / 0.50 x daily ATR) were chosen by me in advance, not fitted to data.
  I have not verified the stop multiple used in the paper.
- A cost screen skips trades whose estimated round-trip costs exceed 35% of the planned
  risk, using this repo's NSE cost model. On a Rs 20,000 account this matters a lot.
- Cash-only sizing (no leverage), positions capped at capital / top_n.
- Pessimistic fills (see `src/orb_backtest.py` docstring).

## Considered and not implemented
"Market intraday momentum" (Gao, Han, Li, Zhou): first-half-hour return predicts the
last-half-hour return. Evidence is on the SPY index (1999-2012), R-squared about 2-4%,
and costs are not addressed. It is an index-level effect that does not translate into a
tradable single-stock cash-segment edge for this account.

## What the validation gate does and does not prove
- It checks for a net-of-cost edge on a chronologically later holdout, with a day-block
  bootstrap and a random-direction placebo, after choosing parameters on earlier data only.
- Verified on synthetic data: pure noise does not pass; a planted edge is detected.
- NOT verified: behaviour on real NSE data (this workspace cannot reach Yahoo or NSE).
- Not checked: survivorship in the stock list, data errors, corporate actions, circuit
  limits, liquidity and slippage beyond the flat model, and regime change.
- Yahoo's 5-minute history is limited to about 60 days, so on Yahoo data the gate will
  normally answer INSUFFICIENT_DATA. A longer, reliable 5-minute history is needed
  (Kite Connect's historical API is the intended source; check its current limits).
- A pass permits forward paper trading only. It is not a forecast or an approval.
