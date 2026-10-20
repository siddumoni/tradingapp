"""Command-line entry point for Phase 1 authentication utilities."""

from __future__ import annotations

import argparse
import logging
import sys

from src.auth import AuthConfigurationError, TokenCacheError, check_session, login


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NSE intraday assistant — Phase 1 authentication only; no trading actions."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging (never logs token values)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("login", help="Perform manual Kite login and cache today's token.")
    subparsers.add_parser("check", help="Validate the cached Kite session.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "login":
            login()
            return 0
        if args.command == "check":
            return 0 if check_session() else 1
        return 2
    except (AuthConfigurationError, TokenCacheError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
