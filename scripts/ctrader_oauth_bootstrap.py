#!/usr/bin/env python3
"""One-shot OAuth bootstrap for cTrader Open API.

Starts a local callback server, opens the cTrader authorisation URL in the
browser, waits for the redirect, exchanges the code for tokens, then writes
CTRADER_ACCESS_TOKEN and CTRADER_REFRESH_TOKEN directly into the .env.openapi
credentials file so the hub picks them up immediately.

Usage:
    cd /home/renierdejager/Projects/ctrader_trading_bot
    python3 scripts/ctrader_oauth_bootstrap.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ctrader_open_api import Auth

HOST = "127.0.0.1"
PORT = 8787

_CRED_FILE = Path(__file__).resolve().parent.parent / ".env.openapi"


def _load_env_file(path: Path) -> None:
    """Parse key=value lines from an env file into os.environ (skips already-set vars)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip().removeprefix("export").strip().strip('"')
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# Auto-load from .env.openapi so the script works without manual sourcing
_load_env_file(_CRED_FILE)


def _must_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        print(f"ERROR: {name} is not set in environment or in {_CRED_FILE}", file=sys.stderr)
        raise SystemExit(1)
    return v


class _CallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None
    error: str | None = None

    def do_GET(self) -> None:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _CallbackHandler.code = (qs.get("code") or [None])[0]
        _CallbackHandler.error = (qs.get("error") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        if _CallbackHandler.code:
            self.wfile.write(b"<h2>Authorised</h2><p>Token received. You can close this tab.</p>")
        else:
            self.wfile.write(
                f"<h2>Failed</h2><p>error={_CallbackHandler.error}</p>".encode()
            )

    def log_message(self, *_) -> None:
        # suppress default stderr request logging
        pass


def _wait_for_callback(httpd: HTTPServer) -> None:
    while _CallbackHandler.code is None and _CallbackHandler.error is None:
        httpd.handle_request()


def _write_tokens_to_env(access_token: str, refresh_token: str) -> None:
    """Update CTRADER_ACCESS_TOKEN and CTRADER_REFRESH_TOKEN in .env.openapi."""
    if not _CRED_FILE.exists():
        print(f"WARNING: credentials file not found at {_CRED_FILE}")
        print(f"  access_token  = {access_token}")
        print(f"  refresh_token = {refresh_token}")
        return

    lines = _CRED_FILE.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    found_access = found_refresh = False
    for line in lines:
        key = line.partition("=")[0].strip().removeprefix("export").strip().strip('"')
        if key == "CTRADER_ACCESS_TOKEN":
            updated.append(f'CTRADER_ACCESS_TOKEN="{access_token}"')
            found_access = True
        elif key == "CTRADER_REFRESH_TOKEN":
            updated.append(f'CTRADER_REFRESH_TOKEN="{refresh_token}"')
            found_refresh = True
        else:
            updated.append(line)

    if not found_access:
        updated.append(f'CTRADER_ACCESS_TOKEN="{access_token}"')
    if not found_refresh:
        updated.append(f'CTRADER_REFRESH_TOKEN="{refresh_token}"')

    _CRED_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"\nTokens written to {_CRED_FILE}")


def main() -> None:
    client_id     = _must_env("CTRADER_CLIENT_ID")
    client_secret = _must_env("CTRADER_CLIENT_SECRET")
    redirect_uri  = os.environ.get("CTRADER_REDIRECT_URI", f"http://{HOST}:{PORT}/callback").strip()
    scope         = os.environ.get("CTRADER_SCOPE", "trading").strip()

    auth = Auth(client_id, client_secret, redirect_uri)
    auth_url = auth.getAuthUri(scope=scope)

    print("\n" + "=" * 60)
    print("cTrader OAuth — open this URL in your browser:")
    print()
    print(f"  {auth_url}")
    print()
    print(f"Waiting for callback on http://{HOST}:{PORT}/callback ...")
    print("=" * 60 + "\n")

    # Try to open the browser automatically
    try:
        subprocess.Popen(
            ["xdg-open", auth_url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # browser open is best-effort

    httpd = HTTPServer((HOST, PORT), _CallbackHandler)
    httpd.timeout = 1
    t = threading.Thread(target=_wait_for_callback, args=(httpd,), daemon=True)
    t.start()
    t.join()

    if _CallbackHandler.error:
        print(f"\nERROR: OAuth failed — {_CallbackHandler.error}", file=sys.stderr)
        raise SystemExit(1)
    if not _CallbackHandler.code:
        print("\nERROR: No authorization code received.", file=sys.stderr)
        raise SystemExit(1)

    print("Authorization code received. Exchanging for tokens ...")
    token = auth.getToken(_CallbackHandler.code)

    if token.get("errorCode"):
        print(f"\nERROR: Token exchange failed: {token.get('errorCode')} — {token.get('description')}", file=sys.stderr)
        raise SystemExit(1)

    access_token  = token.get("accessToken", "")
    refresh_token = token.get("refreshToken", "")
    expires_in    = token.get("expiresIn", "?")

    if not access_token:
        print(f"\nERROR: No accessToken in response: {token}", file=sys.stderr)
        raise SystemExit(1)

    print(f"  access_token  = {access_token[:12]}...  (expires in {expires_in} s, ~{int(expires_in or 0) // 86400} days)")
    print(f"  refresh_token = {refresh_token[:12]}...  (no expiry)")

    _write_tokens_to_env(access_token, refresh_token)
    print("\nDone. Restart the hub to pick up the new tokens:")
    print("  bash run.sh")


if __name__ == "__main__":
    main()
