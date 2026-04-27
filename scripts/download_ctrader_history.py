#!/usr/bin/env python3
r"""Download historical OHLCV bars from the cTrader Open API and save them as CSV
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
from collections.abc import Generator
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
    1: 1,  # M1
    2: 2,  # M2
    3: 3,  # M3
    4: 4,  # M4
    5: 5,  # M5
    10: 6,  # M10
    15: 7,  # M15
    30: 8,  # M30
    60: 9,  # H1
    240: 10,  # H4
    720: 11,  # H12
    1440: 12,  # D1
    10080: 13,  # W1
    43200: 14,  # MN1
}
MAX_BARS_PER_REQUEST = 4_096
MS = 1_000  # timestamp multiplier (API uses milliseconds)
_WIRE_SCALE = 100_000  # cTrader SpotEvent wire format: 10^5 precision

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
    msg = (
        f"Missing credential {name!r}.\n"
        f"  Set it via --{name.lower().replace('_', '-')},\n"
        f"  export {name}=..., or add it to config/cTraderAppTokens."
    )
    raise SystemExit(
        msg
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
    """Launch the OAuth2 authorisation-code flow.
    Opens a browser, listens on redirect_uri, returns an access token.
    """
    import http.server
    import json
    import threading
    import urllib.parse
    import urllib.request
    import webbrowser

    auth_code: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
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

        def log_message(self, *_) -> None:  # suppress access logs
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
        msg = "Timed out waiting for OAuth2 callback."
        raise SystemExit(msg)

    # Exchange code for token
    token_data = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": auth_code[0],
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode()
    req = urllib.request.Request(
        "https://connect.spotware.com/apps/token",
        data=token_data,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read())

    access_token = payload.get("access_token")
    if not access_token:
        msg = f"Token exchange failed: {payload}"
        raise SystemExit(msg)

    print(f"\nAccess token obtained.  Set it permanently with:\n  export CTRADER_ACCESS_TOKEN={access_token!r}")
    return access_token


# ---------------------------------------------------------------------------
# Twisted / ctrader-open-api download engine
# ---------------------------------------------------------------------------


def _check_library() -> None:
    """Raise a clear error if ctrader-open-api is not installed."""
    try:
        import ctrader_open_api  # noqa: F401
    except ImportError as exc:
        msg = "ctrader-open-api is not installed.\n  pip install ctrader-open-api\nThen re-run this script."
        raise SystemExit(
            msg
        ) from exc


def _dt_to_ms(dt: datetime.datetime) -> int:
    return int(dt.replace(tzinfo=datetime.UTC).timestamp() * MS)


def _ms_to_dt(ms: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ms / MS, tz=datetime.UTC).replace(tzinfo=None)


try:
    from twisted.internet.defer import inlineCallbacks as _inline_callbacks  # type: ignore
except ImportError:
    def _inline_callbacks(f):  # type: ignore[misc]  # noqa: N802
        return f


def _find_symbol_id(sym_res, symbol_name: str) -> int | None:
    for s in sym_res.symbol:
        if s.symbolName.upper() == symbol_name.upper():
            return s.symbolId
    return None


def _write_bars_csv(out_path: Path, all_bars: list) -> None:
    all_bars.sort(key=lambda b: b.utcTimestampInMinutes)
    LOG.info("Writing %d bars → %s", len(all_bars), out_path)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date & Time", "Open", "High", "Low", "Close", "Volume"])
        for b in all_bars:
            ts = _ms_to_dt(b.utcTimestampInMinutes * 60 * MS)
            low_raw = b.low
            open_px = (low_raw + b.deltaOpen) / _WIRE_SCALE
            high_px = (low_raw + b.deltaHigh) / _WIRE_SCALE
            low_px = low_raw / _WIRE_SCALE
            close_px = (low_raw + b.deltaClose) / _WIRE_SCALE
            writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), open_px, high_px, low_px, close_px, b.volume])


@_inline_callbacks  # type: ignore[misc]
def _fetch_all_bars(client, account_id: int, symbol_id: int, period: int,
                    timeframe_minutes: int, from_dt: datetime.datetime,
                    to_dt: datetime.datetime, label: str) -> Generator:
    from ctrader_open_api import Protobuf  # type: ignore
    from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq  # type: ignore
    from twisted.internet import defer  # type: ignore  # noqa: F401

    all_bars: list = []
    chunk_td = datetime.timedelta(minutes=MAX_BARS_PER_REQUEST * timeframe_minutes)
    cursor = from_dt
    while cursor < to_dt:
        end = min(cursor + chunk_td, to_dt)
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = _dt_to_ms(cursor)
        req.toTimestamp = _dt_to_ms(end)
        msg = yield client.send(req, responseTimeoutInSeconds=60)
        res = Protobuf.extract(msg)
        chunk = list(res.trendbar)
        all_bars.extend(chunk)
        LOG.debug("  %s chunk %s→%s: %d bars", label, cursor.date(), end.date(), len(chunk))
        cursor = end
    defer.returnValue(all_bars)


@_inline_callbacks  # type: ignore[misc]
def _download_one(
    *,
    endpoint: str,
    client_id: str,
    client_secret: str,
    access_token: str,
    account_id: int,
    symbol_name: str,
    timeframe_minutes: int,
    from_dt: datetime.datetime,
    to_dt: datetime.datetime,
    output_dir: Path,
) -> Generator:
    """Download bars for one (symbol, TF) pair.  Returns Path or None.
    Must be called from within a running Twisted reactor.
    """
    from ctrader_open_api import Client, Protobuf, TcpProtocol  # type: ignore
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # type: ignore
        ProtoOAAccountAuthReq,
        ProtoOAApplicationAuthReq,
        ProtoOASymbolsListReq,
    )
    from twisted.internet import defer  # type: ignore  # noqa: F401

    period = _TF_MINUTES_TO_PERIOD.get(timeframe_minutes)
    if period is None:
        LOG.error("Unsupported timeframe: %d minutes", timeframe_minutes)
        defer.returnValue(None)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol_name.upper()}_M{timeframe_minutes}.csv"

    client = Client(endpoint, PORT, TcpProtocol)
    client.setDisconnectedCallback(lambda *_: None)
    client.startService()
    yield client.whenConnected(failAfterFailures=1)

    app_req = ProtoOAApplicationAuthReq()
    app_req.clientId = client_id
    app_req.clientSecret = client_secret
    yield client.send(app_req, responseTimeoutInSeconds=15)

    acc_req = ProtoOAAccountAuthReq()
    acc_req.ctidTraderAccountId = account_id
    acc_req.accessToken = access_token
    yield client.send(acc_req, responseTimeoutInSeconds=15)

    sym_req = ProtoOASymbolsListReq()
    sym_req.ctidTraderAccountId = account_id
    sym_req.includeArchivedSymbols = False
    sym_msg = yield client.send(sym_req, responseTimeoutInSeconds=30)
    symbol_id = _find_symbol_id(Protobuf.extract(sym_msg), symbol_name)
    if symbol_id is None:
        LOG.error("Symbol %r not found on this account", symbol_name)
        try:
            client.stopService()
        except Exception:
            pass
        defer.returnValue(None)

    label = f"{symbol_name} M{timeframe_minutes}"
    all_bars = yield _fetch_all_bars(client, account_id, symbol_id, period,
                                     timeframe_minutes, from_dt, to_dt, label)
    try:
        client.stopService()
    except Exception:
        pass
    if not all_bars:
        LOG.error("No bars returned for %s M%d — check account ID and date range", symbol_name, timeframe_minutes)
        defer.returnValue(None)
    _write_bars_csv(out_path, all_bars)
    defer.returnValue(out_path)


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
    """Synchronous wrapper: download one (symbol, TF) pair, return Path or None."""
    _check_library()

    from ctrader_open_api import EndPoints  # type: ignore
    from twisted.internet import defer, reactor  # type: ignore

    endpoint = EndPoints.PROTOBUF_DEMO_HOST if "demo" in host else EndPoints.PROTOBUF_LIVE_HOST
    result: list[Path | None] = [None]

    @defer.inlineCallbacks
    def run_and_stop() -> Generator:
        try:
            out = yield _download_one(
                endpoint=endpoint,
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                account_id=account_id,
                symbol_name=symbol_name,
                timeframe_minutes=timeframe_minutes,
                from_dt=from_dt,
                to_dt=to_dt,
                output_dir=output_dir,
            )
            result[0] = out
        except Exception as exc:
            LOG.error("Download error for %s M%d: %s", symbol_name, timeframe_minutes, exc)
        finally:
            if reactor.running:
                reactor.stop()

    reactor.callWhenRunning(run_and_stop)
    reactor.run()
    return result[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> datetime.datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    msg = f"Cannot parse date {s!r}  (expected YYYY-MM-DD)"
    raise argparse.ArgumentTypeError(msg)


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
    ap.add_argument("--auth", action="store_true", help="Run OAuth2 auth flow to obtain an access token, then exit.")
    ap.add_argument("--symbol", nargs="+", metavar="SYM", help="One or more symbol names, e.g. EURUSD GBPUSD")
    ap.add_argument(
        "--timeframe", nargs="+", type=int, metavar="MIN", help="One or more timeframe values in minutes, e.g. 1 5 60"
    )
    ap.add_argument(
        "--from", dest="date_from", type=_parse_date, metavar="YYYY-MM-DD", help="Start date (inclusive, UTC)"
    )
    ap.add_argument(
        "--to",
        dest="date_to",
        type=_parse_date,
        metavar="YYYY-MM-DD",
        default=datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
        help="End date (exclusive, UTC).  Defaults to today.",
    )
    ap.add_argument(
        "--demo", action="store_true", default=True, help="Use demo server (default).  Pass --live to override."
    )
    ap.add_argument("--live", dest="demo", action="store_false", help="Use live server instead of demo.")
    ap.add_argument(
        "--output-dir", type=Path, default=Path("data/history"), help="Output directory (default: data/history/)"
    )
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
    raw_account = args.account_id or os.environ.get("CTRADER_ACCOUNT_ID") or _detect_account_id_from_fix_cfg(root)
    if not raw_account:
        ap.error("Cannot determine account ID.  Set CTRADER_ACCOUNT_ID or pass --account-id.")
    account_id = int(raw_account)

    host = DEMO_HOST if args.demo else LIVE_HOST
    LOG.info("Server: %s  |  Account: %d", host, account_id)

    jobs = [(sym, tf) for sym in args.symbol for tf in args.timeframe]
    LOG.info("Jobs: %d  (%s × %s)", len(jobs), args.symbol, args.timeframe)

    from ctrader_open_api import EndPoints  # type: ignore
    from twisted.internet import defer, reactor  # type: ignore

    endpoint = EndPoints.PROTOBUF_DEMO_HOST if args.demo else EndPoints.PROTOBUF_LIVE_HOST
    results: list[int] = []

    @defer.inlineCallbacks
    def run_all() -> Generator:
        for symbol, tf in jobs:
            LOG.info(
                "▶  %s M%d  %s → %s",
                symbol, tf, args.date_from.strftime("%Y-%m-%d"), args.date_to.strftime("%Y-%m-%d"),
            )
            try:
                out = yield _download_one(
                    endpoint=endpoint,
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
                    results.append(1)
                else:
                    LOG.error("   ✗  %s M%d failed", symbol, tf)
                    results.append(0)
            except Exception as exc:
                LOG.error("   ✗  %s M%d error: %s", symbol, tf, exc)
                results.append(0)
        reactor.stop()

    reactor.callWhenRunning(run_all)
    reactor.run()

    successes = sum(results)
    LOG.info("Done: %d/%d succeeded.", successes, len(jobs))
    return 0 if successes == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
