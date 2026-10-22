# PROJECT_PROGRESS — Stocks-in-Play ORB candidate + validation gate

Branch: `feature/sip-orb-validation` (off `main` @ 43807c7). Nothing is pushed or committed yet.
Goal: replace "is the strategy good?" with a measurable answer: a candidate strategy plus a
validation gate that decides whether it earns a forward paper-trading period. It does NOT
claim the strategy is profitable. No order-placement code is added anywhere.

## Checkpoint 1 (2026-10-22, resumed after usage limit)

### Verified so far
- Baseline suite: 123 tests pass (unchanged by this branch).
- `src/strategies/stocks_in_play_orb.py`, `src/orb_backtest.py`, `src/synthetic_data.py`
  import cleanly and run end to end on synthetic data (smoke run: 30 symbols x 120 sessions
  -> 187 trades, 178 plans rejected by the cost screen, 6 never triggered).

### Written but NOT yet covered by tests (do not treat as correct yet)
- `src/strategies/stocks_in_play_orb.py`: features (rvol, daily ATR), plan selection.
- `src/orb_backtest.py`: pessimistic fill simulator, sizing, cost screen, placebo hook (`rng`).
- `src/synthetic_data.py`: synthetic market with optional planted edge.

### Known issues found in smoke run
- `prepare_data` is slow (~16 s for 30 symbols x 120 sessions); fine for now, optimise if it
  becomes a problem on real data.
- Cost screen rejects ~48% of plans at the default size (capital 20,000 / top_n 5 = ~4,000
  per position). That is a real small-capital constraint to report, not a bug (to confirm).

### Not started (in order)
1. DONE (see milestone 1).
2. DONE (see milestone 2).
3. DONE (see milestone 2).
4. DONE (see milestone 3).
5. DONE (see milestone 3).
6. Commit on the branch and push the branch (never `main`) - see milestone 4.

### Hard limits of this environment
- Yahoo and NSE are unreachable from this workspace, so no real-data backtest can be run here.
  The gate is run on synthetic data here and must be run on the user's machine for real data.

## Milestone log
- **M1 (done, verified):** `tests/test_stocks_in_play_orb.py` (9) and `tests/test_orb_backtest.py` (11)
  added, 20/20 pass. No-lookahead tests were mutation-checked: removing either `shift(1)` in
  `compute_features` makes them fail. Two initially vacuous assertions were rewritten so they can fail.
  Files changed: tests only (source files unchanged this milestone).
  Next: task 2, `src/orb_validation.py`.
- **M2 (done, verified):** `src/orb_validation.py` + `tests/test_orb_validation.py` (28 tests).
  Full suite: 123 original + 48 new = 171 passing.
  * Refactor (reason: speed): `compute_features` and `prepare_data` now slice per session with
    numpy instead of per-session pandas calls (16.4 s -> 0.5 s for 30 symbols x 120 sessions).
    Verified identical to the previous implementation on data with a dropped opening bar, a NaN
    volume and a partial day, and the earlier smoke run reproduced exactly (187 trades, mean net -9.3355).
  * Harness self-check at full size (30 symbols x 300 sessions, via CLI): pure noise -> FAIL (holdout
    mean net -7.11/trade, profit factor 0.80); planted edge -> PASS_HISTORICAL_GATE. This validates
    the machinery only, not the strategy on real markets.
  * Leak test is mutation-checked: scoring the dev grid on all sessions makes it fail. A first version
    of that test failed because of MY test bug (swapped 45 sessions, holdout is ~42), not a code leak.
  * Observed: with a small data set the gate returns INSUFFICIENT_DATA instead of a verdict (e.g. an
    edge-world seed with only 41 holdout trades) - intended behaviour.
  Next: task 4, `src/paper_log.py`.
- **M3 (done, verified):** `src/paper_log.py` + `tests/test_paper_log.py` (7), `docs/SIP_ORB_RESEARCH.md`,
  README section, `requirements.txt` (+numpy, pandas, yfinance, pinned to the versions tested).
  * Key test: a plan built from only the 09:15 bar (what a live run sees at 09:20) equals the
    backtest's plan for that day, and settlement reproduces the backtest trades exactly.
  * One test failed first because of a wrong assumption of mine (most days have no stocks in
    play, so "sessions settled" is below the calendar count); fixed the test and documented the
    definition in `summarize_log`.
  * Clean-venv check: `pip install -r requirements.txt` then full suite -> 178 passed
    (123 original + 55 new), ~35 s.
  * Both CLIs start (`--help`, `paper_log summary` on an empty log). `plan`/`settle` and the real-data
    `orb_validation` path were NOT run here: they need Yahoo, which this workspace cannot reach.
