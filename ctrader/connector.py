"""
cTrader Open API Connector — authentication and transport.
Based on Kinetra kinetra/connectors/ctrader_connector.py.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

try:
    from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
    from ctrader_open_api.messages import OpenApiCommonMessages_pb2 as _common_msgs
    from ctrader_open_api.messages import OpenApiMessages_pb2 as _api_msgs
    from twisted.internet import reactor as _reactor

    _CTRADER_AVAILABLE = True
except ImportError:
    _CTRADER_AVAILABLE = False
    Client = None
    EndPoints = None
    Protobuf = None
    TcpProtocol = None
    _common_msgs = None
    _api_msgs = None
    _reactor = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CTRADER_M1_PERIOD: int = 1
DEFAULT_REQUEST_TIMEOUT_S: float = 15.0
DEFAULT_CONNECT_TIMEOUT_S: float = 30.0

DEFAULT_SYMBOL_ALIASES: Dict[str, str] = {
    "JP225": "JPN225",
    "USOIL": "SpotCrude",
    "UKOIL": "SpotBrent",
    "NGAS": "NatGas",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _tcp_probe_latency_ms(host: str, port: int, timeout_s: float) -> Optional[float]:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return (time.perf_counter() - start) * 1000.0
    except OSError:
        return None


def _expand_endpoint_candidates(hosts: List[str]) -> List[str]:
    return [h for h in hosts if h and h.strip()]


def _rank_reachable_endpoints(
    candidates: List[str],
    port: int,
    timeout_s: float = 2.0,
) -> List[tuple[str, float]]:
    ranked: List[tuple[str, float]] = []
    for c in candidates:
        lat = _tcp_probe_latency_ms(c, port=port, timeout_s=timeout_s)
        if lat is not None:
            ranked.append((c, lat))
    ranked.sort(key=lambda x: x[1])
    return ranked


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass
class CTraderCredentials:
    """cTrader Open API OAuth2 credentials."""

    client_id: str
    client_secret: str
    access_token: str
    account_id: int
    environment: str = "demo"

    def __repr__(self) -> str:
        return (
            f"CTraderCredentials(client_id={self.client_id!r}, "
            f"client_secret='***', access_token='***', "
            f"account_id={self.account_id!r}, environment={self.environment!r})"
        )

    @classmethod
    def from_env(
        cls,
        env_file: Optional[str] = None,
        openapi_env_file: Optional[str] = None,
    ) -> "CTraderCredentials":
        root = _project_root()
        loaded: Dict[str, str] = {}
        for p in [root / ".env", root / ".env.openapi"]:
            if p.exists():
                loaded.update(_parse_env_file(p))
        if env_file and Path(env_file).exists():
            loaded.update(_parse_env_file(Path(env_file)))
        if openapi_env_file and Path(openapi_env_file).exists():
            loaded.update(_parse_env_file(Path(openapi_env_file)))
        # Fallback: look for Kinetra .env.openapi in sibling directory
        kinetra_env = root.parent / "Kinetra" / ".env.openapi"
        if kinetra_env.exists() and not all(
            loaded.get(k) for k in ("CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID")
        ):
            loaded.update(_parse_env_file(kinetra_env))

        def _get(k: str) -> str:
            return (os.environ.get(k) or loaded.get(k) or "").strip()

        client_id = _get("CTRADER_CLIENT_ID")
        client_secret = _get("CTRADER_CLIENT_SECRET")
        access_token = _get("CTRADER_ACCESS_TOKEN")
        raw_account = _get("CTRADER_ACCOUNT_ID")
        environment = _get("CTRADER_ENVIRONMENT") or "demo"

        missing = [
            k for k, v in [
                ("CTRADER_CLIENT_ID", client_id),
                ("CTRADER_CLIENT_SECRET", client_secret),
                ("CTRADER_ACCESS_TOKEN", access_token),
                ("CTRADER_ACCOUNT_ID", raw_account),
            ]
            if not v
        ]
        if missing:
            raise ValueError(
                f"Missing cTrader credentials: {', '.join(missing)}. "
                "Set in .env.openapi or run: python -m ctrader.auth"
            )

        try:
            account_id = int(raw_account)
        except ValueError:
            raise ValueError(
                f"CTRADER_ACCOUNT_ID must be an integer, got: {raw_account!r}"
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            account_id=account_id,
            environment=environment,
        )


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class CTraderConnector:
    """cTrader Open API connector — Twisted transport + auth."""

    def __init__(
        self,
        credentials: CTraderCredentials,
        symbol_aliases: Optional[Dict[str, str]] = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        if not _CTRADER_AVAILABLE:
            raise ImportError(
                "ctrader_open_api and twisted are required. "
                "Install with: pip install ctrader-open-api"
            )
        self.credentials = credentials
        self.request_timeout_s = request_timeout_s
        self._symbol_aliases = dict(symbol_aliases or DEFAULT_SYMBOL_ALIASES)

        self._client: Any = None
        self._reactor_thread: Optional[threading.Thread] = None
        self._connected_event = threading.Event()
        self._app_auth_done = threading.Event()
        self._app_auth_ok = False
        self._acct_auth_done = threading.Event()
        self._acct_auth_ok = False
        self._responses: Dict[str, Any] = {}
        self._response_events: Dict[str, threading.Event] = {}
        self._cancelled_ids: set = set()
        self._msg_counter = 0
        self._lock = threading.Lock()
        self._shutting_down = False
        self._symbol_map: Dict[str, int] = {}
        self._symbol_names: Dict[int, str] = {}
        self._symbol_digits: Dict[int, int] = {}
        self._symbol_cache_lock = threading.Lock()
        self.broker_title = ""
        self.account_login = 0
        self._event_queue: Optional[queue.Queue] = None

    def set_event_queue(self, q: Optional[queue.Queue] = None) -> None:
        self._event_queue = q

    def start(self, timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> bool:
        host = (
            EndPoints.PROTOBUF_DEMO_HOST
            if self.credentials.environment == "demo"
            else EndPoints.PROTOBUF_LIVE_HOST
        )
        port = EndPoints.PROTOBUF_PORT
        alt_raw = os.environ.get("CTRADER_ALT_ENDPOINTS", "")
        alt_hosts = [h.strip() for h in alt_raw.split(",") if h.strip()]
        candidates = _expand_endpoint_candidates([host] + alt_hosts)

        ranked = _rank_reachable_endpoints(candidates, port=port, timeout_s=2.0)
        selected = ranked[0][0] if ranked else candidates[0]
        if ranked:
            logger.info("[cTrader] Endpoint %s (%.1fms)", ranked[0][0], ranked[0][1])
        else:
            logger.warning("[cTrader] No probe success, using %s", selected)

        self._client = Client(selected, port, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        self._reactor_thread = threading.Thread(
            target=self._run_reactor,
            daemon=True,
            name="ctrader-reactor",
        )
        self._reactor_thread.start()

        if not self._connected_event.wait(timeout=timeout_s):
            logger.error("[cTrader] Connection timeout")
            return False
        if not self._app_auth_done.wait(timeout=timeout_s):
            logger.error("[cTrader] App auth timeout")
            return False
        if not self._app_auth_ok:
            logger.error("[cTrader] App auth failed")
            return False

        self._do_account_auth(timeout_s=timeout_s)
        if not self._acct_auth_done.wait(timeout=timeout_s):
            logger.error("[cTrader] Account auth timeout")
            return False
        if not self._acct_auth_ok:
            logger.error("[cTrader] Account auth failed")
            return False

        logger.info("[cTrader] Connected: account=%d env=%s", self.credentials.account_id, self.credentials.environment)
        return True

    def stop(self) -> None:
        self._shutting_down = True
        try:
            if self._client:
                _reactor.callFromThread(self._client.stopService)
            time.sleep(0.15)
            if _reactor.running:
                _reactor.callFromThread(_reactor.stop)
        except Exception:
            pass

    def is_connected(self) -> bool:
        return (
            self._connected_event.is_set()
            and self._app_auth_ok
            and self._acct_auth_ok
            and not self._shutting_down
        )

    def send_and_wait(self, message: Any, timeout_s: Optional[float] = None) -> Optional[Any]:
        if timeout_s is None:
            timeout_s = self.request_timeout_s
        if self._client is None or self._shutting_down or not self._connected_event.is_set():
            return None

        msg_id = self._next_msg_id()
        event = threading.Event()
        with self._lock:
            self._response_events[msg_id] = event
        _reactor.callFromThread(self._do_send, message, msg_id)

        if not event.wait(timeout=timeout_s):
            with self._lock:
                self._response_events.pop(msg_id, None)
                self._cancelled_ids.add(msg_id)
            return None
        with self._lock:
            self._response_events.pop(msg_id, None)
            return self._responses.pop(msg_id, None)

    def find_symbol_id(self, name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> Optional[int]:
        with self._symbol_cache_lock:
            if not self._symbol_map:
                self._populate_symbol_cache(timeout_s=timeout_s)
        alias = self._symbol_aliases.get(name)
        if alias and alias in self._symbol_map:
            return self._symbol_map[alias]
        stripped = name
        if stripped.endswith("+"):
            stripped = stripped[:-1]
        if stripped.endswith("-C"):
            stripped = stripped[:-2]
        if stripped in self._symbol_map:
            return self._symbol_map[stripped]
        if name in self._symbol_map:
            return self._symbol_map[name]
        name_upper = name.upper()
        for sym_name, sym_id in self._symbol_map.items():
            if sym_name.upper().startswith(name_upper):
                return sym_id
        return None

    def symbol_name_for_id(self, symbol_id: int) -> str:
        return self._symbol_names.get(symbol_id, f"<{symbol_id}>")

    def get_digits(self, symbol_id: int, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> int:
        with self._symbol_cache_lock:
            if symbol_id in self._symbol_digits:
                return self._symbol_digits[symbol_id]
        digits = self._fetch_symbol_digits(symbol_id, timeout_s=timeout_s)
        if digits is not None:
            with self._symbol_cache_lock:
                self._symbol_digits[symbol_id] = digits
            return digits
        return 5

    def _run_reactor(self) -> None:
        self._client.startService()
        try:
            _reactor.run(installSignalHandlers=False)
        except Exception:
            pass

    def _next_msg_id(self) -> str:
        with self._lock:
            self._msg_counter += 1
            return f"msg_{self._msg_counter}"

    def _do_send(self, message: Any, msg_id: str) -> None:
        if self._shutting_down:
            with self._lock:
                self._responses[msg_id] = None
                ev = self._response_events.get(msg_id)
            if ev:
                ev.set()
            return
        d = self._client.send(
            message,
            clientMsgId=msg_id,
            responseTimeoutInSeconds=int(self.request_timeout_s),
        )
        d.addCallback(lambda resp: self._on_response(msg_id, resp))
        d.addErrback(lambda f: self._on_response(msg_id, None))
        d.addErrback(lambda _: None)

    def _on_response(self, msg_id: str, raw: Any) -> None:
        try:
            payload = Protobuf.extract(raw) if raw else None
        except Exception:
            payload = raw
        with self._lock:
            if msg_id in self._cancelled_ids:
                self._cancelled_ids.discard(msg_id)
                return
            self._responses[msg_id] = payload
            ev = self._response_events.get(msg_id)
        if ev:
            ev.set()

    def _on_connected(self, client: Any) -> None:
        self._app_auth_done.clear()
        self._acct_auth_done.clear()
        self._app_auth_ok = False
        self._acct_auth_ok = False
        self._connected_event.set()
        self._send_app_auth(client)

    def _send_app_auth(self, client: Any) -> None:
        req = _api_msgs.ProtoOAApplicationAuthReq()
        req.clientId = self.credentials.client_id
        req.clientSecret = self.credentials.client_secret
        d = client.send(req, responseTimeoutInSeconds=15)
        d.addCallback(self._on_app_auth_response)
        d.addErrback(lambda f: (setattr(self, "_app_auth_ok", False), self._app_auth_done.set()))
        d.addErrback(lambda _: None)

    def _on_app_auth_response(self, raw: Any) -> None:
        try:
            payload = Protobuf.extract(raw)
            name = getattr(getattr(payload, "DESCRIPTOR", None), "name", "") or ""
            self._app_auth_ok = name != "ProtoOAErrorRes"
        except Exception:
            self._app_auth_ok = False
        self._app_auth_done.set()

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        self._connected_event.clear()
        self._app_auth_done.clear()
        self._acct_auth_done.clear()
        self._app_auth_ok = False
        self._acct_auth_ok = False
        if not self._shutting_down:
            logger.warning("[cTrader] Disconnected: %s", reason)

    def _on_message(self, client: Any, raw_message: Any) -> None:
        try:
            payload = Protobuf.extract(raw_message)
        except Exception:
            return
        if getattr(payload, "DESCRIPTOR", None) and payload.DESCRIPTOR.name == "ProtoHeartbeatEvent":
            d = client.send(_common_msgs.ProtoHeartbeatEvent())
            d.addErrback(lambda _: None)
            return
        if self._event_queue is not None and getattr(payload, "DESCRIPTOR", None):
            try:
                self._event_queue.put_nowait((payload.DESCRIPTOR.name, payload))
            except queue.Full:
                pass

    def _do_account_auth(self, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
        req = _api_msgs.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.credentials.account_id
        req.accessToken = self.credentials.access_token
        resp = self.send_and_wait(req, timeout_s=timeout_s)
        if resp is None or hasattr(resp, "errorCode"):
            self._acct_auth_ok = False
        else:
            self._acct_auth_ok = True
            trader = self._fetch_trader(timeout_s=timeout_s)
            if trader:
                self.broker_title = str(getattr(trader, "brokerName", "") or "")
                self.account_login = int(getattr(trader, "traderLogin", 0) or 0)
        self._acct_auth_done.set()

    def _fetch_trader(self, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> Optional[Any]:
        req = _api_msgs.ProtoOATraderReq()
        req.ctidTraderAccountId = self.credentials.account_id
        r = self.send_and_wait(req, timeout_s=timeout_s)
        if r is None or hasattr(r, "errorCode") or not hasattr(r, "trader"):
            return None
        return r.trader

    def _populate_symbol_cache(self, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
        req = _api_msgs.ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.credentials.account_id
        resp = self.send_and_wait(req, timeout_s=timeout_s)
        if resp is None or hasattr(resp, "errorCode"):
            return
        for sym in getattr(resp, "symbol", []):
            sid = getattr(sym, "symbolId", None)
            sname = getattr(sym, "symbolName", None)
            if sid and sname:
                self._symbol_map[sname] = int(sid)
                self._symbol_names[int(sid)] = sname

    def _fetch_symbol_digits(self, symbol_id: int, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> Optional[int]:
        req = _api_msgs.ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = self.credentials.account_id
        req.symbolId.append(symbol_id)
        resp = self.send_and_wait(req, timeout_s=timeout_s)
        if resp is None or hasattr(resp, "errorCode") or not getattr(resp, "symbol", []):
            return None
        sym = resp.symbol[0]
        digits = getattr(sym, "digits", None)
        if digits is not None:
            return int(digits)
        pip = getattr(sym, "pipPosition", None)
        if pip is not None:
            return int(pip) + 1
        return None
