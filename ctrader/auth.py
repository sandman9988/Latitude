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
   - Set Redirect URI to http://127.0.0.1:8787/callback

2. Get Access Token (OAuth2):
   - Use the "Token" tab on openapi.ctrader.com
   - Or: https://openapi.ctrader.com/connect/token
   - Grant access and copy the access_token and refresh_token

3. Get Account ID:
   - cTrader: Tools → Open API → copy ctidTraderAccountId (integer)

4. Create or edit .env.openapi with:

# ============================================================================
# Latitude cTrader Open API — Environment Configuration
# ============================================================================

# --- OAuth2 Application Credentials -----------------------------------------
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_REDIRECT_URI=http://127.0.0.1:8787/callback
CTRADER_SCOPE=trading

# --- Access Token (from OAuth2 flow above) -----------------------------------
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_REFRESH_TOKEN=your_refresh_token   # optional, for token refresh

# --- Account -----------------------------------------------------------------
CTRADER_ACCOUNT_ID=12345678
CTRADER_ENVIRONMENT=demo   # "demo" or "live"

# --- DNS hardening (optional, broker/API agnostic) ---------------------------
LATITUDE_DNS_ALLOW_SUFFIXES=ctraderapi.com,spotware.com
LATITUDE_DNS_MIN_UNIQUE_IPS=1
LATITUDE_DNS_BLOCK_PRIVATE=true
LATITUDE_DNS_FAIL_CLOSED=true
LATITUDE_DNS_RESOLVERS=1.1.1.1,1.0.0.1,8.8.8.8,8.8.4.4
LATITUDE_DNS_QUERY_TIMEOUT_S=2.0
LATITUDE_DNS_QUERY_LIFETIME_S=4.0

# --- Endpoint redundancy -----------------------------------------------------
CTRADER_ALT_ENDPOINTS=15.197.239.248,3.33.208.221,live.ctraderapi.com
CTRADER_INCLUDE_RESOLVED_IP_FALLBACKS=true
CTRADER_ENDPOINT_PROBE_TIMEOUT_S=2.0
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
