                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        #!/usr/bin/env python3
"""
Startup Self-Test
=================
Runs before FIX sessions are created.  Every check is isolated: one failure
never crashes the others.

Severity levels
---------------
CRITICAL  – bot cannot function safely; startup is aborted.
WARNING   – degraded / cold-start mode; bot continues with a logged caveat.
INFO      – diagnostic-only; no action required.

Usage
-----
    from src.core.self_test import run_self_test
    run_self_test()          # raises SystemExit(1) on any CRITICAL failure
"""

import json
import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

# ── ANSI colours (disabled when not a tty) ────────────────────────────────────
_IS_TTY = os.isatty(1)
_G  = "\033[92m"  if _IS_TTY else ""   # green
_Y  = "\033[93m"  if _IS_TTY else ""   # yellow
_R  = "\033[91m"  if _IS_TTY else ""   # red
_B  = "\033[94m"  if _IS_TTY else ""   # blue/cyan
_W  = "\033[0m"   if _IS_TTY else ""   # reset
_BD = "\033[1m"   if _IS_TTY else ""   # bold

# ── check constants ────────────────────────────────────────────────────────────
_MAX_QTY_LOTS        = 100    # position size (lots) considered suspiciously large
_MAX_CORRUPT_LINES   = 5      # max corrupt trade-log lines before WARNING
_STALE_POSITION_SECS = 3600   # 1 h — stale position data threshold
_STALE_DAY_SECS      = 86400  # 24 h — generic one-day staleness threshold
_MAX_PLATT_PARAM     = 50     # abs(a) or abs(b) above this is considered extreme


class Sev(IntEnum):
    """Test result severity."""
    PASS     = 0   # ✓ green  — all good
    INFO     = 1   # ℹ blue   — diagnostic only
    WARNING  = 2   # ⚠ yellow — degraded, but safe to continue
    CRITICAL = 3   # ✗ red    — must abort


_SEV_ICON  = {Sev.PASS: "✓", Sev.INFO: "ℹ", Sev.WARNING: "⚠", Sev.CRITICAL: "✗"}
_SEV_COLOR = {Sev.PASS: _G, Sev.INFO: _B, Sev.WARNING: _Y, Sev.CRITICAL: _R}


@dataclass
class TestResult:
    name: str
    sev: Sev
    detail: str = ""

    def __str__(self) -> str:
        col  = _SEV_COLOR[self.sev]
        icon = _SEV_ICON[self.sev]
        tail = f"  {_W}{self.detail}" if self.detail else ""
        return f"  {col}{icon} {self.name}{_W}{tail}"


@dataclass
class SelfTestReport:
    results: list[TestResult] = field(default_factory=list)

    def add(self, name: str, sev: Sev, detail: str = "") -> TestResult:
        r = TestResult(name, sev, detail)
        self.results.append(r)
        return r

    @property
    def critical_failures(self) -> list[TestResult]:
        return [r for r in self.results if r.sev == Sev.CRITICAL]

    @property
    def warnings(self) -> list[TestResult]:
        return [r for r in self.results if r.sev == Sev.WARNING]

    def print_banner(self) -> None:
        n       = len(self.results)
        n_pass  = sum(1 for r in self.results if r.sev == Sev.PASS)
        n_info  = sum(1 for r in self.results if r.sev == Sev.INFO)
        n_warn  = len(self.warnings)
        n_crit  = len(self.critical_failures)

        print(f"\n{_BD}{'─' * 60}{_W}")
        print(f"{_BD}  🔍 STARTUP SELF-TEST{_W}")
        print(f"{_BD}{'─' * 60}{_W}")
        for r in self.results:
            print(str(r))

        print(f"{_BD}{'─' * 60}{_W}")
        if n_crit:
            status_str = f"{_R}{_BD}FAILED ({n_crit} critical){_W}"
        elif n_warn:
            status_str = f"{_Y}DEGRADED ({n_warn} warnings){_W}"
        else:
            status_str = f"{_G}{_BD}ALL CLEAR{_W}"
        print(
            f"  Checks: {n}  "
            f"{_G}✓{n_pass}{_W}  "
            f"{_B}ℹ{n_info}{_W}  "
            f"{_Y}⚠{n_warn}{_W}  "
            f"{_R}✗{n_crit}{_W}  "
            f"→ {status_str}"
        )
        print(f"{_BD}{'─' * 60}{_W}\n")


# ── individual check helpers ──────────────────────────────────────────────────

def _check(
    report: SelfTestReport,
    name: str,
    fn: Callable[[], tuple[Sev, str]],
) -> None:
    """Run one isolated check; catch all exceptions as CRITICAL."""
    try:
        sev, detail = fn()
        report.add(name, sev, detail)
    except Exception as exc:  # noqa: BLE001
        report.add(name, Sev.CRITICAL, f"unhandled exception: {exc}")


# ── individual checks ─────────────────────────────────────────────────────────

def _chk_env_vars() -> tuple[Sev, str]:
    required = ("CTRADER_USERNAME", "CTRADER_PASSWORD_QUOTE", "CTRADER_PASSWORD_TRADE")
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        return Sev.CRITICAL, f"missing env vars: {', '.join(missing)}"
    return Sev.PASS, ""


def _chk_fix_configs() -> tuple[Sev, str]:
    cfg_q = os.environ.get("CTRADER_CFG_QUOTE", "ctrader_quote.cfg")
    cfg_t = os.environ.get("CTRADER_CFG_TRADE", "ctrader_trade.cfg")
    missing = [p for p in (cfg_q, cfg_t) if not Path(p).exists()]
    if missing:
        return Sev.CRITICAL, f"not found: {missing}"
    # Make sure they are non-empty and contain [DEFAULT] or [SESSION]
    for p in (cfg_q, cfg_t):
        txt = Path(p).read_text()
        if not any(tok in txt for tok in ("[DEFAULT]", "[SESSION]", "BeginString")):
            return Sev.CRITICAL, f"{p} looks corrupt (no FIX session markers)"
    return Sev.PASS, f"{cfg_q}, {cfg_t}"


def _chk_data_dir() -> tuple[Sev, str]:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    probe = data_dir / ".write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        return Sev.CRITICAL, f"data/ not writable: {e}"
    return Sev.PASS, str(data_dir.resolve())


def _chk_log_dir() -> tuple[Sev, str]:
    for d in ("logs", "logs/fix", "logs/audit", "logs/ctrader"):
        Path(d).mkdir(parents=True, exist_ok=True)
    return Sev.PASS, "logs/ (fix/, audit/, ctrader/)"


def _chk_qty() -> tuple[Sev, str]:
    try:
        qty = float(os.environ.get("CTRADER_QTY", "0.01"))
    except ValueError:
        return Sev.CRITICAL, "CTRADER_QTY is not a valid float"
    if qty <= 0:
        return Sev.CRITICAL, f"qty={qty} must be > 0"
    if qty > _MAX_QTY_LOTS:
        return Sev.CRITICAL, f"qty={qty} suspiciously large (>{_MAX_QTY_LOTS} lots)"
    if qty > 1.0:
        return Sev.WARNING, f"qty={qty} — confirm this is intentional (>1 lot)"
    return Sev.PASS, f"qty={qty}"


def _chk_learned_params() -> tuple[Sev, str]:
    p = Path("data/learned_parameters.json")
    if not p.exists():
        return Sev.WARNING, "not found — cold start, defaults will be used"
    try:
        data = json.loads(p.read_text())
        n = len(data) if isinstance(data, dict) else 0
        return Sev.PASS, f"{n} params loaded"
    except json.JSONDecodeError as e:
        return Sev.WARNING, f"corrupt ({e}) — cold start"


def _chk_bars_cache() -> tuple[Sev, str]:
    p = Path("data/bars_cache.json")
    if not p.exists():
        return Sev.WARNING, "not found — vol/regime will be blind until first bars close"
    try:
        age = time.time() - p.stat().st_mtime
        data = json.loads(p.read_text())
        n = len(data.get("bars", [])) if isinstance(data, dict) else 0
        age_str = f"{age/3600:.1f}h old"
        if age > 86400 * 2:
            return Sev.WARNING, f"{n} bars cached but {age_str} — stale"
        return Sev.PASS, f"{n} bars cached ({age_str})"
    except Exception as e:
        return Sev.WARNING, f"corrupt ({e}) — will rebuild from live data"


def _chk_trade_log() -> tuple[Sev, str]:
    p = Path("data/trade_log.jsonl")
    if not p.exists():
        return Sev.INFO, "no trade history yet"
    try:
        lines = p.read_text().splitlines()
        valid = 0
        corrupt = 0
        for line in lines[-50:]:   # spot-check last 50 lines only
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
                valid += 1
            except json.JSONDecodeError:
                corrupt += 1
        total = len(lines)
        if corrupt > _MAX_CORRUPT_LINES:
            return Sev.WARNING, f"{total} entries, {corrupt} corrupt in last 50 — check trade_log.jsonl"
        return Sev.PASS, f"{total} entries ({valid} of last 50 valid)"
    except Exception as e:
        return Sev.WARNING, f"unreadable: {e}"


def _chk_circuit_breakers() -> tuple[Sev, str]:
    p = Path("data/circuit_breakers.json")
    if not p.exists():
        return Sev.INFO, "no state file — all breakers start open"
    try:
        data = json.loads(p.read_text())
        # Schema uses "is_tripped" (matches CircuitBreakers.save_state())
        # Guard against both old schema ("tripped") and missing keys gracefully.
        tripped = [
            k for k, v in data.items()
            if isinstance(v, dict) and (v.get("is_tripped") or v.get("tripped"))
        ]
        if tripped:
            return Sev.WARNING, f"tripped from previous session: {tripped}"
        # Verify schema version matches what save_state() writes
        expected_keys = {"sortino", "kurtosis", "drawdown", "consecutive_losses"}
        present_keys = {k for k in data if k != "timestamp"}
        if present_keys and not present_keys.issubset(expected_keys | {"timestamp"}):
            return Sev.WARNING, f"unexpected CB schema keys {present_keys - expected_keys} — schema drift?"
        return Sev.PASS, "none tripped"
    except Exception as e:
        return Sev.WARNING, f"unreadable ({e}) — will reset"


def _chk_current_position() -> tuple[Sev, str]:
    p = Path("data/current_position.json")
    if not p.exists():
        return Sev.INFO, "no persisted position (flat start)"
    try:
        data = json.loads(p.read_text())
        pos  = data.get("position", 0)
        price = data.get("entry_price", 0)
        age   = time.time() - (data.get("timestamp") or 0)
        if pos != 0 and age > _STALE_POSITION_SECS:
            return Sev.WARNING, (
                f"stale position data ({age/3600:.1f}h old): "
                f"pos={pos} entry={price} — verify via FIX"
            )
        if pos != 0:
            side = "LONG" if pos > 0 else "SHORT"
            return Sev.INFO, f"recovering {side} qty={abs(pos)} entry={price}"
        return Sev.PASS, "flat"
    except Exception as e:
        return Sev.WARNING, f"corrupt ({e}) — will recover from FIX"


def _chk_per_buffer() -> tuple[Sev, str]:
    found = []
    for name in ("trigger", "harvester"):
        # checkpoint dir is data/checkpoints/
        cand = list(Path("data/checkpoints").glob(f"{name}_buffer*.npz")) if Path("data/checkpoints").exists() else []
        if cand:
            latest = max(cand, key=lambda f: f.stat().st_mtime)
            age = time.time() - latest.stat().st_mtime
            found.append(f"{name}({age/3600:.1f}h)")
    if not found:
        return Sev.WARNING, "no checkpoints — buffers start empty (training delayed)"
    return Sev.PASS, f"found: {', '.join(found)}"


def _chk_model_weights() -> tuple[Sev, str]:
    trigger_path = os.environ.get("DDQN_TRIGGER_MODEL", "").strip()
    harvester_path = os.environ.get("DDQN_HARVESTER_MODEL", "").strip()

    results = []
    for label, path in (("trigger", trigger_path), ("harvester", harvester_path)):
        if not path:
            results.append(f"{label}=not configured (online init)")
            continue
        p = Path(path)
        if not p.exists():
            return Sev.WARNING, f"{label} model path configured but not found: {path}"
        kb = p.stat().st_size // 1024
        # GAP-3: Verify torch is installed and the file is actually loadable.
        # Without this check the bot silently falls back to the MA/VPIN heuristic
        # while the operator believes the learned model is active.
        try:
            import torch as _torch  # noqa: PLC0415
            _torch.load(p, map_location="cpu", weights_only=True)
        except ImportError:
            return Sev.WARNING, (
                f"{label}: file exists ({kb}KB) but torch not installed "
                "— falling back to DDQN numpy / heuristic"
            )
        except Exception as e:
            return Sev.WARNING, f"{label}: file exists ({kb}KB) but torch.load failed: {e}"
        results.append(f"{label}={p.name} ({kb}KB, loadable)")
    return Sev.PASS, ", ".join(results)


def _chk_risk_metrics() -> tuple[Sev, str]:
    p = Path("data/risk_metrics.json")
    if not p.exists():
        return Sev.INFO, "not found — will be written after first bar"
    try:
        data = json.loads(p.read_text())
        var  = data.get("var_95", 0)
        vol  = data.get("realized_vol", 0)
        age  = time.time() - (data.get("timestamp") or 0)
        return Sev.PASS if age < _STALE_DAY_SECS else Sev.WARNING, (
            f"VaR={var:.4f} vol={vol:.4f} age={age/3600:.1f}h"
            + (" (stale)" if age >= _STALE_DAY_SECS else "")
        )
    except Exception as e:
        return Sev.WARNING, f"corrupt ({e})"


def _chk_bot_config() -> tuple[Sev, str]:
    p = Path("data/bot_config.json")
    if not p.exists():
        return Sev.INFO, "not yet written (created on first bar)"
    try:
        data = json.loads(p.read_text())
        qty = data.get("qty", 0)
        sym = data.get("symbol", "?")
        return Sev.PASS, f"symbol={sym} qty={qty}"
    except Exception as e:
        return Sev.WARNING, f"corrupt ({e})"


def _chk_platt_sanity() -> tuple[Sev, str]:
    """Check Platt calibration params from learned_parameters; warn if degenerate."""
    p = Path("data/learned_parameters.json")
    if not p.exists():
        return Sev.INFO, "no learned params — Platt at defaults (a=1.0 b=0.0)"
    try:
        data = json.loads(p.read_text())
        a = float(data.get("platt_a", 1.0))
        b = float(data.get("platt_b", 0.0))
        if not (math.isfinite(a) and math.isfinite(b)):
            return Sev.WARNING, f"non-finite Platt params a={a} b={b} — will reset"
        if abs(a) > _MAX_PLATT_PARAM or abs(b) > _MAX_PLATT_PARAM:
            return Sev.WARNING, f"extreme Platt params a={a:.2f} b={b:.2f} — may over/undergate"
        if math.isclose(a, 1.0) and math.isclose(b, 0.0):
            return Sev.INFO, "Platt at defaults (not yet adapted)"
        return Sev.PASS, f"a={a:.4f} b={b:+.4f}"
    except Exception as e:
        return Sev.WARNING, f"could not read: {e}"


def _chk_quickfix_importable() -> tuple[Sev, str]:
    """Verify QuickFIX Python bindings are importable.

    QuickFIX is not on PyPI and must be installed manually from the
    vendor source tree.  If import fails the bot cannot create FIX
    sessions at all, so this is CRITICAL.
    """
    try:
        import quickfix  # noqa: PLC0415, F401
        version = getattr(quickfix, "__version__", "unknown")
        return Sev.PASS, f"quickfix {version}"
    except ImportError as e:
        return Sev.CRITICAL, (
            f"quickfix not importable: {e} — run: "
            "cd ../quickfix && pip install -e ."
        )


def _chk_numpy_sanity() -> tuple[Sev, str]:
    """Quick sanity: numpy basic ops work and give finite results."""
    x = np.array([1.0, 2.0, 3.0])
    if not math.isfinite(float(np.mean(x))):
        return Sev.CRITICAL, "numpy mean returned non-finite value"
    softmax = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    if not all(math.isfinite(v) for v in softmax):
        return Sev.CRITICAL, "numpy softmax returned non-finite values"
    return Sev.PASS, f"numpy {np.__version__}"


def _chk_symbol_specs() -> tuple[Sev, str]:
    p = Path("config/symbol_specs.json")
    if not p.exists():
        return Sev.WARNING, "config/symbol_specs.json not found — using defaults"
    try:
        data = json.loads(p.read_text())
        sym  = os.environ.get("CTRADER_SYMBOL", "XAUUSD")
        if sym not in data:
            return Sev.WARNING, f"{sym} not in symbol_specs.json — broker defaults used"
        spec = data[sym]
        return Sev.PASS, f"{sym}: contract={spec.get('contract_size')} pip={spec.get('pip_value')}"
    except Exception as e:
        return Sev.WARNING, f"corrupt ({e})"


# ── orchestrator ──────────────────────────────────────────────────────────────

# Ordered list of (name, check_fn, is_critical_check)
# is_critical_check=True means a CRITICAL result aborts startup.
# is_critical_check=False means even CRITICAL is downgraded to WARNING.
_CHECKS: list[tuple[str, Callable[[], tuple[Sev, str]], bool]] = [
    # ── Hard requirements ─────────────────────────────────────────────────
    ("Environment variables",     _chk_env_vars,             True),
    ("QuickFIX importable",        _chk_quickfix_importable,  True),
    ("FIX config files",           _chk_fix_configs,          True),
    ("Data directory writable",    _chk_data_dir,             True),
    ("Log directories",            _chk_log_dir,              True),
    ("Position size (qty)",        _chk_qty,                  True),
    ("NumPy arithmetic",           _chk_numpy_sanity,         True),
    # ── Soft requirements (warn, don't abort) ─────────────────────────────
    ("Learned parameters",        _chk_learned_params,    False),
    ("Bars cache",                _chk_bars_cache,        False),
    ("Trade log",                 _chk_trade_log,         False),
    ("Circuit breaker state",     _chk_circuit_breakers,  False),
    ("Persisted position",        _chk_current_position,  False),
    ("PER buffer checkpoints",    _chk_per_buffer,        False),
    ("DDQN model weights",        _chk_model_weights,     False),
    # ── Informational ─────────────────────────────────────────────────────
    ("Risk metrics cache",        _chk_risk_metrics,      False),
    ("Bot config cache",          _chk_bot_config,        False),
    ("Symbol specs",              _chk_symbol_specs,      False),
    ("Platt calibration params",  _chk_platt_sanity,      False),
]


def _persist_results(report: SelfTestReport) -> None:
    """Write data/self_test.json for the HUD; silently swallowed on failure."""
    try:
        export = {
            "timestamp": time.time(),
            "results": [
                {"name": r.name, "sev": r.sev.name, "detail": r.detail}
                for r in report.results
            ],
            "summary": {
                "pass":     sum(1 for r in report.results if r.sev == Sev.PASS),
                "info":     sum(1 for r in report.results if r.sev == Sev.INFO),
                "warnings": len(report.warnings),
                "critical": len(report.critical_failures),
            },
        }
        Path("data").mkdir(exist_ok=True)
        Path("data/self_test.json").write_text(json.dumps(export, indent=2))
    except Exception:  # noqa: BLE001
        pass  # never let JSON export block startup


def _log_results(report: SelfTestReport) -> None:
    """Emit per-result log lines at the appropriate level."""
    for r in report.results:
        if r.sev == Sev.CRITICAL:
            LOG.error("[SELF_TEST] CRITICAL: %s — %s", r.name, r.detail)
        elif r.sev == Sev.WARNING:
            LOG.warning("[SELF_TEST] WARNING: %s — %s", r.name, r.detail)
        else:
            LOG.info("[SELF_TEST] %s: %s — %s", r.sev.name, r.name, r.detail)


def run_self_test(*, abort_on_critical: bool = True) -> SelfTestReport:
    """
    Run all startup checks.

    Parameters
    ----------
    abort_on_critical : bool
        If True (default) raise SystemExit(1) when any CRITICAL check fails.
        Set False in unit tests.

    Returns
    -------
    SelfTestReport – full results (useful for tests / programmatic inspection)
    """
    report = SelfTestReport()

    for name, fn, can_be_critical in _CHECKS:
        try:
            sev, detail = fn()
            # If the check isn't designated critical, cap severity at WARNING
            if not can_be_critical and sev == Sev.CRITICAL:
                sev = Sev.WARNING
            report.add(name, sev, detail)
        except Exception as exc:  # noqa: BLE001
            sev = Sev.CRITICAL if can_be_critical else Sev.WARNING
            report.add(name, sev, f"unhandled exception: {exc}")

    report.print_banner()
    _persist_results(report)
    _log_results(report)

    if abort_on_critical and report.critical_failures:
        names = ", ".join(r.name for r in report.critical_failures)
        LOG.critical("[SELF_TEST] Aborting startup: critical failures in [%s]", names)
        raise SystemExit(1)

    return report
