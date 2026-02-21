#!/usr/bin/env python3
"""
Download historical OHLCV bars from the cTrader Open API and save them as CSV
files compatible with src/training/historical_loader.load_csv().

Install the Spotware library once:
    pip install ctrader-open-api

Credentials are read from (in order of priority):
    CLI flags > environment variables > config/cTraderAppTokens

Required environment variables (or CLI equivalents):
    CTRADER_CLIENT_ID       – OAuth2 client ID
    CTRADER_CLIENT_SECRET   – OAuth2 client secret
    CTRADER_ACCESS_TOKEN    – OAuth2 access token for the account
    CTRADER_ACCOUNT_ID      – cTrader account ID (numeric)

Usage examples:
    # Download EURUSD M1 for all of 2024 (live account)
    python scripts/download_ctrader_history.py \\
        --symbol EURUSD --timeframe 1 \\
        --from 2024-01-01 --to 2025-01-01

    # Download multiple symbols + timeframes in one call
    python scripts/download_ctrader_history.py \\
        --symbol EURUSD GBPUSD USDJPY \\
        --timeframe 1 5 15 60 \\
        --from 2023-01-01 --to 2025-01-01 \\
        --demo

    # Obtain a fresh access token (opens browser)
    python scripts/download_ctrader_history.py --auth

Output CSV location:
    data/history/<SYMBOL>_M<TF>.csv

CSV format (cTrader layout, detected automatically by historical_loader):
    Date & Time,Open,High,Low,Close,Volume
    2024-01-02 00:00:00,1.10423,1.10455,1.10391,1.10432,1234

Max bars per API call: 4 096.  Requests are chunked automatically.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cTrader Open API endpoints
# ---------------------------------------------------------------------------
LIVE_HOST = "live.ctraderapi.com"
DEMO_HOST = "demo.ctraderapi.com"
PORT = 5035

# OATrendbarPeriod enum values (from Open API proto)
_TF_MINUTES_TO_PERIOD: dict[int, int] = {
    1: 1,      # M1
    2: 2,      # M2
    3: 3,      # M3
    4: 4,      # M4
    5: 5,      # M5
    10: 6,     # M10
    15: 7,     # M15
    30: 8,     # M30
    60: 9,     # H1
    240: 10,   # H4
    720: 11,   # H12
    1440: 12,  # D1
    10080: 13, # W1
    43200: 14, # MN1
}
MAX_BARS_PER_REQUEST = 4_096
MS = 1_000  # timestamp multiplier (API uses milliseconds)

# ---------------------------------------------------------------------------
# Credential loading
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


def _detect_account_id_from_fix_cfg(project_root: Path) -> str | None:
    """Try to read the account number from the FIX quote config."""
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
# OAuth2 auth-code flow helper (opens a local HTTP server + browser)
# ---------------------------------------------------------------------------

def run_auth_flow(client_id: str, client_secret: str, redirect_uri: str) -> str:
    """
    Launch the OAuth2 authorisation-code flow.
    Opens a browser, listens on redirect_uri, returns an access token.
    """
    import http.server
    import threading
    import urllib.parse
    import urllib.request
    import webbrowser
    import json

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
        def log_message(self, *_):  # suppress access logs
            pass

    # Parse port from redirect_uri
    parsed_redirect = urllib.parse.urlparse(redirect_uri)
    port = parsed_redirect.port or 8787

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    auth_url = (
        "https://connect.spotware.com/apps/auth"
        f"?client_id={urllib.parse.quote(client_id)}"
        f"&redirect_uri={urllib.parse.quote(redirect_uri)}"
        "&response_type=code"
        "&scope=trading"
    )
    print(f"Opening browser for OAuth2 authorisation...\n{auth_url}")
    webbrowser.open(auth_url)

    # Wait up to 120 seconds for the callback
    deadline = time.time() + 120
    while not auth_code and time.time() < deadline:
        time.sleep(0.3)
    server.shutdown()

    if not auth_code:
        raise SystemExit("Timed out waiting for OAuth2 callback.")

    # Exchange code for token
    token_data = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": auth_code[0],
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(
        "https://connect.spotware.com/apps/token",
        data=token_data,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read())

    access_token = payload.get("access_token")
    if not access_token:
        raise SystemExit(f"Token exchange failed: {payload}")

    print(f"\nAccess token obtained.  Set it permanently with:\n"
          f"  export CTRADER_ACCESS_TOKEN={access_token!r}")
    return access_token


# ---------------------------------------------------------------------------
# Twisted / ctrader-open-api download engine
# ---------------------------------------------------------------------------

def _check_library() -> None:
    """Raise a clear error if ctrader-open-api is not installed."""
    try:
        import ctrader_open_api  # noqa: F401
    except ImportError:
        raise SystemExit(
            "ctrader-open-api is not installed.\n"
            "  pip install ctrader-open-api\n"
            "Then re-run this script."
        )


def _dt_to_ms(dt: datetime.datetime) -> int:
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * MS)


def _ms_to_dt(ms: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ms / MS, tz=datetime.timezone.utc).replace(tzinfo=None)


def download_symbol(
    *,
    host: str,
    client_id: str,
    client_secret: str,
    access_token: str,
    account_id: int,
    symbol_name: str,
    timeframe_minutes: int,
    from_dt: datetime.datetime,
    to_dt: datetime.datetime,
    output_dir: Path,
) -> Path | None:
    """
    Download all bars for one (symbol, timeframe) window.
    Returns the path of the written CSV, or None on error.
    Runs synchronously using twisted.internet.reactor in a deferred chain.
    """
    _check_library()

    from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol  # type: ignore
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (  # type: ignore
        ProtoMessage,
        ProtoPayloadType,
    )
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # type: ignore
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOAGetSymbolsListReq,
        ProtoOAGetTrendbarsReq,
    )
    from twisted.internet import reactor, defer  # type: ignore
    from twisted.internet.protocol import ReconnectingClientFactory  # type: ignore

    period = _TF_MINUTES_TO_PERIOD.get(timeframe_minutes)
    if period is None:
        LOG.error("Unsupported timeframe: %d minutes", timeframe_minutes)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol_name.upper()}_M{timeframe_minutes}.csv"
    out_path = output_dir / filename

    # Shared mutable state for the deferred chain
    state: dict = {
        "symbol_id": None,
        "all_bars": [],
        "error": None,
    }

    def on_error(failure):
        state["error"] = str(failure)
        LOG.error("Download error for %s M%d: %s", symbol_name, timeframe_minutes, failure)
        if reactor.running:
            reactor.stop()

    def fetch_bars_chunk(client, symbol_id, chunk_from, chunk_to):
        """Send one GetTrendbarsReq and collect the response."""
        result_d: defer.Deferred = defer.Deferred()

        def on_message(client, message):
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # type: ignore
                ProtoOAGetTrendbarsRes,
            )
            if message.payloadType == Protobuf.get(ProtoOAGetTrendbarsRes).payloadType:
                res = Protobuf.extract(message)
                bars = list(res.trendbar)
                result_d.callback(bars)

        client.setMessageReceivedCallback(on_message)
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = chunk_from
        req.toTimestamp = chunk_to
        client.sendProtoMessage(req)
        return result_d

    def run(client):
        # Step 1: App auth
        app_req = ProtoOAApplicationAuthReq()
        app_req.clientId = client_id
        app_req.clientSecret = client_secret
        app_d: defer.Deferred = defer.Deferred()

        def on_app_auth(client, message):
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # type: ignore
                ProtoOAApplicationAuthRes,
            )
            if message.payloadType == Protobuf.get(ProtoOAApplicationAuthRes).payloadType:
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
                from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # type: ignore
                    ProtoOAAccountAuthRes,
                )
                if message.payloadType == Protobuf.get(ProtoOAAccountAuthRes).payloadType:
                    acc_d.callback(None)

            client.setMessageReceivedCallback(on_acc_auth)
            client.sendProtoMessage(acc_req)
            return acc_d

        @app_d.addCallback
        def fetch_symbols(_):
            sym_req = ProtoOAGetSymbolsListReq()
            sym_req.ctidTraderAccountId = account_id
            sym_req.includeArchivedSymbols = False
            sym_d: defer.Deferred = defer.Deferred()

            def on_symbols(client, message):
                from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # type: ignore
                    ProtoOAGetSymbolsListRes,
                )
                if message.payloadType == Protobuf.get(ProtoOAGetSymbolsListRes).payloadType:
                    res = Protobuf.extract(message)
                    for s in res.symbol:
                        if s.symbolName.upper() == symbol_name.upper():
                            state["symbol_id"] = s.symbolId
                            break
                    sym_d.callback(None)

            client.setMessageReceivedCallback(on_symbols)
            client.sendProtoMessage(sym_req)
            return sym_d

        @app_d.addCallback
        def download_loop(_):
            if state["symbol_id"] is None:
                raise ValueError(f"Symbol {symbol_name!r} not found on this account.")

            symbol_id = state["symbol_id"]
            chunk_minutes = MAX_BARS_PER_REQUEST * timeframe_minutes
            chunk_td = datetime.timedelta(minutes=chunk_minutes)

            all_bars = state["all_bars"]
            chunks = []
            cursor = from_dt
            while cursor < to_dt:
                end = min(cursor + chunk_td, to_dt)
                chunks.append((_dt_to_ms(cursor), _dt_to_ms(end)))
                cursor = end

            d = defer.succeed(None)
            for chunk_from_ms, chunk_to_ms in chunks:
                def _fetch(_, cfm=chunk_from_ms, ctm=chunk_to_ms):
                    return fetch_bars_chunk(client, symbol_id, cfm, ctm)

                def _collect(bars):
                    all_bars.extend(bars)
                    LOG.debug("  Fetched %d bars (total so far: %d)", len(bars), len(all_bars))

                d.addCallback(_fetch)
                d.addCallback(_collect)
            return d

        @app_d.addCallback
        def write_csv(_):
            bars = state["all_bars"]
            LOG.info("Writing %d bars → %s", len(bars), out_path)
            # Sort by timestamp ascending
            bars.sort(key=lambda b: b.utcTimestampInMinutes)
            with out_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Date & Time", "Open", "High", "Low", "Close", "Volume"])
                for b in bars:
                    ts = _ms_to_dt(b.utcTimestampInMinutes * 60 * MS)
                    dt_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                    # cTrader API returns prices in relative ticks; digits depend on symbol
                    # We store as-received integers — caller must apply pip factor if needed
                    writer.writerow([dt_str, b.open, b.high, b.low, b.close, b.volume])

        @app_d.addCallback
        def done(_):
            if reactor.running:
                reactor.stop()

        app_d.addErrback(on_error)

    endpoint = EndPoints.PROTOBUF_DEMO_HOST if "demo" in host else EndPoints.PROTOBUF_LIVE_HOST
    client = Client(endpoint, PORT, TcpProtocol)
    client.setConnectedCallback(run)
    client.setDisconnectedCallback(lambda *_: None)

    reactor.connectTCP(endpoint, PORT, client.factory)
    reactor.run()

    if state["error"]:
        return None
    if not out_path.exists():
        LOG.error("No output file created for %s M%d", symbol_name, timeframe_minutes)
        return None
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime.datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Cannot parse date {s!r}  (expected YYYY-MM-DD)")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Download cTrader historical bars to data/history/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--auth", action="store_true",
                    help="Run OAuth2 auth flow to obtain an access token, then exit.")
    ap.add_argument("--symbol", nargs="+", metavar="SYM",
                    help="One or more symbol names, e.g. EURUSD GBPUSD")
    ap.add_argument("--timeframe", nargs="+", type=int, metavar="MIN",
                    help="One or more timeframe values in minutes, e.g. 1 5 60")
    ap.add_argument("--from", dest="date_from", type=_parse_date, metavar="YYYY-MM-DD",
                    help="Start date (inclusive, UTC)")
    ap.add_argument("--to", dest="date_to", type=_parse_date, metavar="YYYY-MM-DD",
                    default=datetime.datetime.utcnow(),
                    help="End date (exclusive, UTC).  Defaults to today.")
    ap.add_argument("--demo", action="store_true", default=True,
                    help="Use demo server (default).  Pass --live to override.")
    ap.add_argument("--live", dest="demo", action="store_false",
                    help="Use live server instead of demo.")
    ap.add_argument("--output-dir", type=Path, default=Path("data/history"),
                    help="Output directory (default: data/history/)")
    ap.add_argument("--client-id", help="OAuth2 client ID (overrides env/tokens file)")
    ap.add_argument("--client-secret", help="OAuth2 client secret (overrides env/tokens file)")
    ap.add_argument("--access-token", help="OAuth2 access token (overrides env/tokens file)")
    ap.add_argument("--account-id", help="cTrader numeric account ID (overrides env/config)")
    ap.add_argument("-v", "--verbose", action="store_true")

    args = ap.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Locate project root
    root = Path(__file__).resolve().parent.parent
    tokens_file = _load_tokens_file(root / "config" / "cTraderAppTokens")

    client_id = _get_cred("CTRADER_CLIENT_ID", tokens_file, args.client_id)
    client_secret = _get_cred("CTRADER_CLIENT_SECRET", tokens_file, args.client_secret)
    redirect_uri = os.environ.get("CTRADER_REDIRECT_URI", "http://127.0.0.1:8787/callback")

    # Auth-only mode
    if args.auth:
        token = run_auth_flow(client_id, client_secret, redirect_uri)
        print(f"CTRADER_ACCESS_TOKEN={token}")
        return 0

    # Validate required args for download mode
    if not args.symbol:
        ap.error("--symbol is required for download mode (or use --auth first)")
    if not args.timeframe:
        ap.error("--timeframe is required")
    if not args.date_from:
        ap.error("--from is required")

    # Check library before touching credentials
    _check_library()

    access_token = _get_cred("CTRADER_ACCESS_TOKEN", tokens_file, args.access_token)

    # Account ID: CLI > env > FIX config
    raw_account = (
        args.account_id
        or os.environ.get("CTRADER_ACCOUNT_ID")
        or _detect_account_id_from_fix_cfg(root)
    )
    if not raw_account:
        ap.error(
            "Cannot determine account ID.  Set CTRADER_ACCOUNT_ID or pass --account-id."
        )
    account_id = int(raw_account)

    host = DEMO_HOST if args.demo else LIVE_HOST
    LOG.info("Server: %s  |  Account: %d", host, account_id)

    jobs = [(sym, tf) for sym in args.symbol for tf in args.timeframe]
    LOG.info("Jobs: %d  (%s × %s)", len(jobs), args.symbol, args.timeframe)

    successes = 0
    for symbol, tf in jobs:
        LOG.info("▶  %s M%d  %s → %s",
                 symbol, tf,
                 args.date_from.strftime("%Y-%m-%d"),
                 args.date_to.strftime("%Y-%m-%d"))
        out = download_symbol(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            account_id=account_id,
            symbol_name=symbol,
            timeframe_minutes=tf,
            from_dt=args.date_from,
            to_dt=args.date_to,
            output_dir=args.output_dir,
        )
        if out:
            LOG.info("   ✓  %s", out)
            successes += 1
        else:
            LOG.error("   ✗  %s M%d failed", symbol, tf)

    LOG.info("Done: %d/%d succeeded.", successes, len(jobs))
    return 0 if successes == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
