#!/usr/bin/env python3
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from ctrader_open_api import Auth

HOST = "127.0.0.1"
PORT = 8787


def must_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


class CodeHandler(BaseHTTPRequestHandler):
    server_version = "cTraderOAuth/1.0"
    code = None
    error = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        CodeHandler.code = qs.get("code", [None])[0]
        CodeHandler.error = qs.get("error", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        if CodeHandler.code:
            self.wfile.write(b"<h2>OK</h2><p>Authorization code received. You can close this tab.</p>")
        else:
            msg = f"<h2>Failed</h2><p>No code received. error={CodeHandler.error}</p>"
            self.wfile.write(msg.encode("utf-8"))

    def log_message(self, fmt, *args):
        # quiet
        return


def run_server():
    httpd = HTTPServer((HOST, PORT), CodeHandler)
    httpd.timeout = 1
    while CodeHandler.code is None and CodeHandler.error is None:
        httpd.handle_request()


def main():
    client_id = must_env("CTRADER_CLIENT_ID")
    client_secret = must_env("CTRADER_CLIENT_SECRET")

    # Your redirect URI must be registered in cTrader Open API portal
    redirect_uri = os.environ.get("CTRADER_REDIRECT_URI", f"http://{HOST}:{PORT}/callback").strip()

    # scope: "trading" or "accounts"
    scope = os.environ.get("CTRADER_SCOPE", "trading").strip()

    auth = Auth(client_id, client_secret, redirect_uri)
    auth_uri = auth.getAuthUri(scope=scope)

    print("\n1) Open this URL in a normal browser and login/allow access:\n")
    print(auth_uri)
    print(f"\n2) After allowing access, your browser will redirect to:\n   {redirect_uri}\n")
    print("Waiting for redirect...\n")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    t.join()

    if CodeHandler.error:
        raise SystemExit(f"OAuth failed: error={CodeHandler.error}")
    if not CodeHandler.code:
        raise SystemExit("OAuth failed: no authorization code received")

    code = CodeHandler.code
    print(f"Received code: {code[:8]}... (redacted)\n")

    token = auth.getToken(code)
    if token.get("errorCode"):
        raise SystemExit(f"Token error: {token.get('errorCode')} {token.get('description')}")

    # Save token safely
    out_path = os.environ.get("CTRADER_TOKEN_FILE", "ctrader_token.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(token, f, indent=2)

    print(f"Saved token to: {out_path}")
    print("Keys: accessToken, refreshToken, expiresIn, tokenType\n")


if __name__ == "__main__":
    main()
