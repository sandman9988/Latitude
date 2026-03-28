#!/usr/bin/env python3
"""
cTrader OAuth2 authentication helper.
Opens browser to obtain access token and saves credentials to .env.openapi.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="cTrader OAuth2 auth helper")
    parser.add_argument("--print-only", action="store_true", help="Only print instructions")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env.openapi"

    instructions = """
cTrader Open API credentials
============================

1. Register an application at https://openapi.ctrader.com/
   - Create app, copy Client ID and Client Secret

2. Get Access Token (OAuth2):
   - Use the "Token" tab on openapi.ctrader.com
   - Or: https://openapi.ctrader.com/connect/token
   - Grant access and copy the access_token

3. Get Account ID:
   - cTrader: Tools → Open API → copy ctidTraderAccountId (integer)

4. Create or edit .env.openapi with:

CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_ACCOUNT_ID=12345678
CTRADER_ENVIRONMENT=demo

Environment: "demo" or "live"
"""
    print(instructions)

    if args.print_only:
        return 0

    try:
        webbrowser.open("https://openapi.ctrader.com/")
    except Exception:
        pass

    print("\nOpening https://openapi.ctrader.com/ in browser (if possible)")
    print(f"Save credentials to: {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
