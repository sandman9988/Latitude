#!/usr/bin/env python3
"""
Fetch real account balance, equity, and margin from the cTrader Open API
and write the result to data/account_balance.json.

The bot and HUD read this file to display the live Pepperstone balance
instead of relying on the FIX CollateralInquiry (which cTrader ignores).

Install once:
    pip install ctrader-open-api

Credentials (in order of priority):
    CLI flags > environment variables > config/cTraderAppTokens

Required:
    CTRADER_CLIENT_ID       – OAuth2 client ID
    CTRADER_CLIENT_SECRET   – OAuth2 client secret
    CTRADER_ACCESS_TOKEN    – OAuth2 access token
    CTRADER_ACCOUNT_ID      – cTrader numeric account ID

Usage:
    # One-shot fetch (e.g. from cron every 5 min)
    python scripts/fetch_balance.py

    # Continuous polling every 60 s (runs until killed)
    python scripts/fetch_balance.py --loop --interval 60

    # Use live server (default is demo)
    python scripts/fetch_balance.py --live

    # Obtain a fresh access token (opens browser)
    python scripts/fetch_balance.py --auth
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

LOG = logging.getLogger("fetch_balance")

# cTrader Open API endpoints
LIVE_HOST = "live.ctraderapi.com"
DEMO_HOST = "demo.ctraderapi.com"
PORT = 5035

OUTPUT_FILE = "data/account_balance.json"


# ---------------------------------------------------------------------------
# Credential loading  (shared pattern with download_ctrader_history.py)
# ---------------------------------------------------------------------------

def _load_tokens_file(path: Path) -> dict[str, str]:
    """Parse a shell-export key=value file and return a dict."""
    result: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            m = re.match(r'^\s*export\s+(\w+)=["\']?([^"\']+)["\']?\s*$', line)
            if m:
                result[m.group(1)] = m.group(2).strip()
    except OSError:
        pass
    return result


def _get_cred(name: str, tokens_file: dict[str, str], cli_value: str | None) -> str:
    """Return credential: CLI > env > tokens file.  Raises if not found."""
    if cli_value:
        return cli_value
    if name in os.environ:
        return os.environ[name]
    if name in tokens_file:
        return tokens_file[name]
    raise SystemExit(
        f"Missing credential {name!r}.\n"
        f"  Set it via --{name.lower().replace('_', '-')},\n"
        f"  export {name}=..., or add it to config/cTraderAppTokens."
    )


def _detect_account_id(project_root: Path) -> str | None:
    """Try to read the account number from the FIX quote config SenderCompID."""
    cfg = project_root / "config" / "ctrader_quote.cfg"
    try:
        for line in cfg.read_text().splitlines():
            m = re.match(r"SenderCompID\s*=\s*[\w.]+?\.(\d+)\s*$", line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Core: connect → auth → fetch balance → write JSON
# ---------------------------------------------------------------------------

def fetch_and_write(
    *,
    host: str,
    client_id: str,
    client_secret: str,
    access_token: str,
    account_id: int,
    output_path: Path,
) -> dict | None:
    """
    Connect to cTrader Open API, authenticate, request trader info,
    and write balance data to *output_path*.

    Returns the balance dict on success, None on error.
    Uses Twisted reactor in a blocking one-shot pattern.
    """
    try:
        from ctrader_open_api import Client, Protobuf, TcpProtocol  # type: ignore
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # type: ignore
            ProtoOAApplicationAuthReq,
            ProtoOAApplicationAuthRes,
            ProtoOAAccountAuthReq,
            ProtoOAAccountAuthRes,
            ProtoOATraderReq,
            ProtoOATraderRes,
        )
        from twisted.internet import reactor, defer  # type: ignore
    except ImportError:
        LOG.error("ctrader-open-api is not installed.  pip install ctrader-open-api")
        return None

    result: dict = {"error": None, "data": None}

    def on_error(failure):
        result["error"] = str(failure)
        LOG.error("Open API error: %s", failure)
        if reactor.running:
            reactor.stop()

    def run(client):
        # Step 1: Application auth
        app_req = ProtoOAApplicationAuthReq()
        app_req.clientId = client_id
        app_req.clientSecret = client_secret
        app_d: defer.Deferred = defer.Deferred()

        def on_app_auth(client, message):
            if message.payloadType == 2101:  # ProtoOAApplicationAuthRes
                app_d.callback(None)

        client.setMessageReceivedCallback(on_app_auth)
        client.sendProtoMessage(app_req)

        @app_d.addCallback
        def account_auth(_):
            acc_req = ProtoOAAccountAuthReq()
            acc_req.ctidTraderAccountId = account_id
            acc_req.accessToken = access_token
            acc_d: defer.Deferred = defer.Deferred()

            def on_acc_auth(client, message):
                if message.payloadType == 2103:  # ProtoOAAccountAuthRes
                    acc_d.callback(None)

            client.setMessageReceivedCallback(on_acc_auth)
            client.sendProtoMessage(acc_req)
            return acc_d

        @app_d.addCallback
        def request_trader(_):
            trader_req = ProtoOATraderReq()
            trader_req.ctidTraderAccountId = account_id
            trader_d: defer.Deferred = defer.Deferred()

            def on_trader(client, message):
                if message.payloadType == 2122:  # ProtoOATraderRes
                    res = Protobuf.extract(message)
                    trader = res.trader
                    # balance is in cents (integer) — divide by 10^moneyDigits
                    money_digits = trader.moneyDigits if trader.moneyDigits else 2
                    divisor = 10 ** money_digits
                    balance = trader.balance / divisor

                    now = datetime.datetime.now(datetime.timezone.utc)
                    data = {
                        "balance": balance,
                        "money_digits": money_digits,
                        "balance_raw": trader.balance,
                        "leverage_in_cents": trader.leverageInCents,
                        "max_leverage": trader.maxLeverage,
                        "broker_name": trader.brokerName,
                        "trader_login": trader.traderLogin,
                        "swap_free": trader.swapFree,
                        "account_id": account_id,
                        "fetched_at": now.isoformat(),
                        "fetched_at_unix": now.timestamp(),
                    }
                    result["data"] = data
                    trader_d.callback(data)

            client.setMessageReceivedCallback(on_trader)
            client.sendProtoMessage(trader_req)
            return trader_d

        @app_d.addCallback
        def done(data):
            # Write JSON atomically
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = output_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(output_path)
            LOG.info(
                "✓ Balance: %.2f  (login=%s, broker=%s)  → %s",
                data["balance"],
                data.get("trader_login", "?"),
                data.get("broker_name", "?"),
                output_path,
            )
            if reactor.running:
                reactor.stop()

        app_d.addErrback(on_error)

    client = Client(host, PORT, TcpProtocol)
    client.setConnectedCallback(run)
    client.setDisconnectedCallback(lambda c, reason: None)
    client.startService()

    # Timeout: if nothing happens in 20 s, stop
    reactor.callLater(20, lambda: reactor.stop() if reactor.running else None)  # type: ignore
    reactor.run(installSignalHandlers=False)  # type: ignore

    if result["error"]:
        LOG.error("Fetch failed: %s", result["error"])
        return None
    return result["data"]


# ---------------------------------------------------------------------------
# OAuth2 auth flow (reused from download_ctrader_history.py)
# ---------------------------------------------------------------------------

def run_auth_flow(client_id: str, client_secret: str, redirect_uri: str) -> str:
    """Launch the OAuth2 authorisation-code flow, return an access token."""
    import http.server
    import threading
    import urllib.parse
    import urllib.request
    import webbrowser

    auth_code: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            code = params.get("code", [None])[0]
            if code:
                auth_code.append(code)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h2>Authorised! You can close this tab.</h2>")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"<h2>No code received.</h2>")
        def log_message(self, *_):
            pass

    # Parse port from redirect_uri
    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or 8787

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    auth_url = (
        f"https://openapi.ctrader.com/apps/auth"
        f"?client_id={client_id}"
        f"&redirect_uri={urllib.parse.quote(redirect_uri)}"
        f"&scope=trading"
    )
    print(f"Opening browser for authorisation...\n{auth_url}")
    webbrowser.open(auth_url)

    t.join(timeout=120)
    server.server_close()

    if not auth_code:
        raise SystemExit("No authorisation code received within 120 s.")

    # Exchange code for token
    token_url = "https://openapi.ctrader.com/apps/token"
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": auth_code[0],
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }).encode()
    req = urllib.request.Request(token_url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    access_token = body.get("accessToken") or body.get("access_token")
    if not access_token:
        raise SystemExit(f"Token exchange failed: {body}")
    return access_token


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Fetch cTrader account balance via Open API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--auth", action="store_true",
                    help="Run OAuth2 flow to obtain an access token, then exit.")
    ap.add_argument("--loop", action="store_true",
                    help="Run continuously, polling every --interval seconds.")
    ap.add_argument("--interval", type=int, default=300,
                    help="Polling interval in seconds (default: 300 = 5 min).")
    ap.add_argument("--demo", action="store_true", default=True,
                    help="Use demo server (default).")
    ap.add_argument("--live", dest="demo", action="store_false",
                    help="Use live server.")
    ap.add_argument("--output", type=Path, default=None,
                    help=f"Output JSON path (default: {OUTPUT_FILE}).")
    ap.add_argument("--client-id", help="OAuth2 client ID")
    ap.add_argument("--client-secret", help="OAuth2 client secret")
    ap.add_argument("--access-token", help="OAuth2 access token")
    ap.add_argument("--account-id", help="cTrader numeric account ID")
    ap.add_argument("-v", "--verbose", action="store_true")

    args = ap.parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    root = Path(__file__).resolve().parent.parent
    tokens_file = _load_tokens_file(root / "config" / "cTraderAppTokens")

    client_id = _get_cred("CTRADER_CLIENT_ID", tokens_file, args.client_id)
    client_secret = _get_cred("CTRADER_CLIENT_SECRET", tokens_file, args.client_secret)
    redirect_uri = os.environ.get("CTRADER_REDIRECT_URI", "http://127.0.0.1:8787/callback")

    # Auth-only mode
    if args.auth:
        token = run_auth_flow(client_id, client_secret, redirect_uri)
        print(f"\nCTRADER_ACCESS_TOKEN={token}")
        print("\nAdd to config/cTraderAppTokens:")
        print(f'  export CTRADER_ACCESS_TOKEN="{token}"')
        return 0

    access_token = _get_cred("CTRADER_ACCESS_TOKEN", tokens_file, args.access_token)

    raw_account = (
        args.account_id
        or os.environ.get("CTRADER_ACCOUNT_ID")
        or _detect_account_id(root)
    )
    if not raw_account:
        ap.error("Cannot determine account ID.  Set CTRADER_ACCOUNT_ID or pass --account-id.")
    account_id = int(raw_account)

    host = DEMO_HOST if args.demo else LIVE_HOST
    output_path = args.output or (root / OUTPUT_FILE)

    LOG.info("Server: %s  |  Account: %d  |  Output: %s", host, account_id, output_path)

    if not args.loop:
        # One-shot
        data = fetch_and_write(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            account_id=account_id,
            output_path=output_path,
        )
        return 0 if data else 1

    # Loop mode — poll every N seconds
    # Twisted reactor can only run once, so for loop mode we use subprocess
    LOG.info("Loop mode: polling every %d s  (Ctrl+C to stop)", args.interval)

    stop = False

    def _sigterm(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    import subprocess
    cmd_base = [
        sys.executable, __file__,
        "--client-id", client_id,
        "--client-secret", client_secret,
        "--access-token", access_token,
        "--account-id", str(account_id),
        "--output", str(output_path),
    ]
    if not args.demo:
        cmd_base.append("--live")

    while not stop:
        try:
            subprocess.run(cmd_base, timeout=30, check=False)
        except subprocess.TimeoutExpired:
            LOG.warning("Fetch timed out — will retry in %d s", args.interval)
        except Exception as e:
            LOG.error("Fetch error: %s — will retry in %d s", e, args.interval)

        # Sleep in small increments so SIGTERM is responsive
        for _ in range(args.interval):
            if stop:
                break
            time.sleep(1)

    LOG.info("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
