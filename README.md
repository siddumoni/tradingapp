# NSE Advisory-Only Intraday Assistant — Phase 1

A manually run Python assistant for NSE equities using Zerodha Kite Connect. The system is **advisory-only**: it must never place, modify, or cancel orders. Phase 1 implements configuration and authentication only; market-data retrieval and signals come in later phases.

## Requirements

- Windows 11
- Python 3.13 (64-bit recommended)
- VS Code with the Python extension
- A valid Kite Connect app with API key and API secret

## Setup in the VS Code PowerShell terminal

From the directory where you want the project:

```powershell
cd path\to\nse_advisory_assistant
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in VS Code and replace both placeholder values with your Kite Connect credentials. Do not put credentials in source code, screenshots, chat messages, or Git commits.

If PowerShell blocks virtual-environment activation, use this only for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Configure the Kite Connect redirect URL

In the Kite developer console, configure the redirect URL for your app. When login completes, Kite redirects to that URL with a `request_token` query parameter. This project uses a manual redirect flow: you copy only the `request_token` value and paste it into the terminal. The request token is short-lived and single-use; do not reuse an old one.

## Daily login

```powershell
python -m src.main login
```

1. Open the printed login URL in your browser.
2. Sign in to Zerodha and complete any required authentication.
3. From the registered redirect URL, copy the value of `request_token` (not the full URL).
4. Paste it into the terminal prompt.
5. The app exchanges it for an access token and stores the token in `.local\kite_token.json`.

Kite access tokens expire at 06:00 IST the following day (unless invalidated earlier). The local cache cutoff is conservative; the actual API may reject a session sooner. Re-run login if the session check fails.

## Check the cached session

```powershell
python -m src.main check
```

This makes a read-only profile request to verify the cached token. It does not fetch market data or access order endpoints.

## Run unit tests

```powershell
python -m pytest -q
```

The tests use dummy values and local temporary files. They do not require a Zerodha login or a live API call.

## Security notes

- `.env` and `.local\kite_token.json` contain secrets and are excluded by `.gitignore`.
- The token cache is plaintext. Treat it as a password: keep your Windows account protected, do not sync/share the `.local` folder, and delete the cache if you suspect exposure. On Windows, ensure the file is accessible only to your user account using NTFS permissions.
- Never commit `.env` or the token cache. If a secret is accidentally committed, rotate it in the Kite developer console; deleting the file in a later commit is not enough.
- Never paste API secrets or access tokens into ChatGPT, issues, screenshots, or logs.
- This phase contains no order-placement, order-modification, or order-cancellation calls.

## Phase 1 scope

Included: project layout, YAML defaults, credential loading, manual daily login, local token caching, session verification, unit tests.

Not included yet: instruments, quotes, historical candles, indicators, strategy signals, cost calculations, margin estimates, news, backtesting, report generation, or scheduling.

## Default risk configuration

`config/settings.yaml` records the requested starting values: ₹20,000 capital, ₹100 risk per trade, 1:1.5 minimum net R:R, MIS intent, margin verification required, no new entries after 14:45 IST, and intended square-off by 15:15 IST. These are configuration values only; Phase 1 does not create or manage trades.
