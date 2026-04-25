#!/usr/bin/env python3
"""
HUD Data Validation Script
==========================
Runs comprehensive checks on HUD data sources for inconsistencies, staleness, and incompleteness.

Usage:
    python3 validate_hud_data.py
    python3 validate_hud_data.py --export audit_results.json
    python3 validate_hud_data.py --fix (apply recommended fixes)

This script is the audit enforcement mechanism for HUD data integrity.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ============================================================================
# CONSTANTS & THRESHOLDS
# ============================================================================
DATA_DIR = Path("data")

# Staleness thresholds (seconds)
STALE_THRESHOLD_CRITICAL = 300  # 5 minutes
STALE_THRESHOLD_WARNING = 60    # 1 minute

# Data quality thresholds
MIN_ENTRY_TIME_COVERAGE = 0.95  # At least 95% of trades should have entry_time
MIN_QUANTITY_COVERAGE = 1.00    # All trades must have quantity
MIN_PNL_CONSISTENCY = 0.05      # PnL across sources should match within 5%
DEFAULT_CONTRACT_SIZE_BY_SYMBOL = {
    "XAUUSD": 100.0,
}
FRESHNESS_PREFIXES = (
    "training_stats",
    "risk_metrics",
    "performance_snapshot",
    "production_metrics",
    "paper_stats",
    "bot_config",
    "current_position",
)

# File configuration (source of truth hierarchy)
FILES_AUTHORITATIVE = {
    "trade_log.jsonl": ("trades", "CRITICAL"),
    "learned_parameters.json": ("params", "HIGH"),
    "training_stats_XAUUSD_M5.json": ("training", "HIGH"),
}

FILES_SECONDARY = {
    "production_metrics.json": ("metrics", "MEDIUM"),
    "performance_snapshot.json": ("snapshot", "LOW"),
    "bot_config.json": ("config", "MEDIUM"),
}

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

class HUDDataValidator:
    """Validates HUD data integrity and reports inconsistencies."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.issues = []
        self.warnings = []
        self.infos = []
        self.trade_log = []
        self.load_trade_log()

    def load_trade_log(self):
        """Load trade_log.jsonl into memory."""
        trade_file = self.data_dir / "trade_log.jsonl"
        if not trade_file.exists():
            self.issues.append("CRITICAL: trade_log.jsonl not found")
            return

        try:
            with open(trade_file) as f:
                for line in f:
                    if line.strip():
                        self.trade_log.append(json.loads(line.strip()))
        except Exception as e:
            self.issues.append(f"CRITICAL: Failed to parse trade_log.jsonl: {e}")

    def check_entry_time_coverage(self) -> tuple[float, int]:
        """Check what % of trades have entry_time field."""
        if not self.trade_log:
            return 0.0, 0

        null_count = sum(1 for t in self.trade_log if t.get("entry_time") is None)
        coverage = 1.0 - (null_count / len(self.trade_log))

        if coverage < MIN_ENTRY_TIME_COVERAGE:
            self.issues.append(
                f"⚠️  CRITICAL: Only {coverage:.1%} trades have entry_time "
                f"({null_count} missing, affects avg_trade_duration calculation)"
            )
        elif null_count > 0:
            self.warnings.append(
                f"⚠️  {null_count} trades missing entry_time (will be excluded from duration calc)"
            )

        return coverage, null_count

    def check_quantity_coverage(self) -> tuple[float, int]:
        """Check what % of trades have quantity field."""
        if not self.trade_log:
            return 0.0, 0

        null_count = sum(1 for t in self.trade_log if "quantity" not in t or t.get("quantity") is None)
        coverage = 1.0 - (null_count / len(self.trade_log))

        if coverage < MIN_QUANTITY_COVERAGE:
            self.issues.append(
                f"❌ CRITICAL: {null_count}/{len(self.trade_log)} trades missing 'quantity' field. "
                f"HUD cannot display: qty_usage_ratio, position_sizing, risk_per_trade"
            )

        return coverage, null_count

    def check_pnl_consistency(self) -> dict[str, float]:
        """Check PnL across different sources."""
        if not self.trade_log:
            return {}

        current_pnl = sum(t.get("pnl", 0) for t in self.trade_log)
        original_pnl = sum(t.get("pnl_original", 0) for t in self.trade_log if "pnl_original" in t)
        recalc_count = sum(1 for t in self.trade_log if t.get("pnl_recalculated"))

        if recalc_count > 0:
            variance = abs(current_pnl - original_pnl)
            variance_pct = (variance / abs(original_pnl) * 100) if original_pnl != 0 else 0
            reconciled, mismatches = self._reconciled_recalculated_pnl()
            if reconciled:
                self.infos.append(
                    f"✓ PnL recalculation reconciled for {recalc_count} trades. "
                    "Current pnl is account-currency; pnl_original is retained point-scale history."
                )
            else:
                self.issues.append(
                    f"⚠️  CRITICAL: {recalc_count} trades ({recalc_count/len(self.trade_log):.1%}) have unreconciled recalculated PnL. "
                    f"Current total: ${current_pnl:.2f}, Original: ${original_pnl:.2f}, "
                    f"Variance: ${variance:.2f} ({variance_pct:.1f}%). "
                    f"Mismatches against expected account-currency scale: {mismatches}"
                )

        result = {
            "current_pnl": current_pnl,
            "original_pnl": original_pnl,
            "recalc_count": recalc_count,
            "variance_usd": abs(current_pnl - original_pnl),
            "variance_pct": (abs(current_pnl - original_pnl) / abs(original_pnl) * 100) if original_pnl != 0 else 0,
        }
        return result

    def _reconciled_recalculated_pnl(self) -> tuple[bool, int]:
        """Return True when recalculated PnL matches point PnL * qty * contract size."""
        recalc = [t for t in self.trade_log if t.get("pnl_recalculated") and "pnl_original" in t]
        if not recalc:
            return True, 0
        mismatches = 0
        for trade in recalc:
            try:
                symbol = str(trade.get("symbol", "") or "").upper()
                qty = abs(float(trade.get("quantity", trade.get("qty", 0.0)) or 0.0))
                contract_size = float(
                    trade.get("scale_contract_size")
                    or trade.get("contract_size")
                    or DEFAULT_CONTRACT_SIZE_BY_SYMBOL.get(symbol, 1.0)
                )
                expected_from_original = float(trade.get("pnl_original", 0.0) or 0.0) * qty * contract_size
                actual = float(trade.get("pnl", 0.0) or 0.0)
            except (TypeError, ValueError):
                mismatches += 1
                continue
            expected_values = [expected_from_original]
            try:
                entry = float(trade.get("entry_price", 0.0) or 0.0)
                exit_price = float(trade.get("exit_price", 0.0) or 0.0)
                direction = str(trade.get("direction", "") or "").upper()
                direction_sign = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 0.0
                if entry and exit_price and direction_sign:
                    expected_values.append((exit_price - entry) * direction_sign * qty * contract_size)
            except (TypeError, ValueError):
                pass
            if not any(abs(actual - expected) <= max(0.05, abs(expected) * 0.001) for expected in expected_values):
                mismatches += 1
        return mismatches == 0, mismatches

    def check_file_staleness(self) -> dict[str, dict[str, Any]]:
        """Check age of critical data files."""
        now = datetime.now(UTC)
        staleness = {}

        for filename in self.data_dir.glob("*.json"):
            if not self._should_check_staleness(filename):
                continue
            mtime = filename.stat().st_mtime
            file_dt = datetime.fromtimestamp(mtime, UTC)
            age = now - file_dt
            age_secs = age.total_seconds()

            status = "OK"
            if age_secs > STALE_THRESHOLD_CRITICAL:
                status = "CRITICAL"
                self.issues.append(
                    f"⚠️  STALE: {filename.name} is {age_secs/60:.1f}min old"
                )
            elif age_secs > STALE_THRESHOLD_WARNING:
                status = "WARNING"
                self.warnings.append(
                    f"ℹ️  AGING: {filename.name} is {age_secs:.0f}s old"
                )

            staleness[filename.name] = {
                "age_seconds": age_secs,
                "last_modified": file_dt.isoformat(),
                "status": status,
            }

        return staleness

    def _should_check_staleness(self, path: Path) -> bool:
        """Only check freshness for runtime HUD telemetry, not static registries/history."""
        if "archive" in path.parts:
            return False
        return path.name.startswith(FRESHNESS_PREFIXES)

    def _load_json_file(self, path: Path) -> dict:
        """Load a JSON file for validation, returning an empty dict on failure."""
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
        except Exception as e:
            self.issues.append(f"CRITICAL: Failed to parse {self._display_path(path)}: {e}")
            return {}

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.data_dir))
        except ValueError:
            return str(path)

    def _expected_scope_from_path(self, path: Path) -> tuple[str | None, int | None]:
        """Infer expected (symbol, timeframe_minutes) from scoped HUD paths."""
        for parent in (path.parent, *path.parents):
            name = parent.name
            if name.startswith(("paper_", "live_")) and "_M" in name:
                body = name.split("_", 1)[1]
                symbol_part, tf_part = body.rsplit("_M", 1)
                if tf_part.isdigit():
                    return symbol_part.upper(), int(tf_part)

        stem = path.stem
        if "_M" not in stem:
            return None, None
        prefix, tf_part = stem.rsplit("_M", 1)
        if not tf_part.isdigit():
            return None, None
        for metric_prefix in (
            "training_stats_",
            "risk_metrics_",
            "performance_snapshot_",
            "production_metrics_",
        ):
            if prefix.startswith(metric_prefix):
                symbol = prefix[len(metric_prefix):]
                if symbol:
                    return symbol.upper(), int(tf_part)
        return None, None

    def _payload_scope(self, data: dict) -> dict:
        """Return the dict that should carry symbol/timeframe scope."""
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            return metrics
        return data

    def _check_scoped_payload_file(self, path: Path) -> bool:
        """Validate that a scoped HUD file carries matching symbol/timeframe metadata."""
        expected_symbol, expected_tf = self._expected_scope_from_path(path)
        if expected_symbol is None or expected_tf is None:
            return True

        data = self._load_json_file(path)
        scope = self._payload_scope(data)
        actual_symbol = str(scope.get("symbol", "") or "").upper()
        try:
            actual_tf = int(scope.get("timeframe_minutes", 0) or 0)
        except (TypeError, ValueError):
            actual_tf = 0

        ok = True
        if actual_symbol and actual_symbol != expected_symbol:
            self.issues.append(
                f"CRITICAL: {self._display_path(path)} symbol scope {actual_symbol} does not match expected {expected_symbol}"
            )
            ok = False
        elif not actual_symbol:
            self.warnings.append(f"⚠️  SCOPE MISSING: {self._display_path(path)} has no symbol field")
            ok = False

        if actual_tf and actual_tf != expected_tf:
            self.issues.append(
                f"CRITICAL: {self._display_path(path)} timeframe scope M{actual_tf} does not match expected M{expected_tf}"
            )
            ok = False
        elif not actual_tf:
            self.warnings.append(f"⚠️  SCOPE MISSING: {self._display_path(path)} has no timeframe_minutes field")
            ok = False

        return ok

    def check_multi_bot_sync(self) -> dict[str, bool]:
        """Validate per-bot HUD payload scope.

        Per-bot values are expected to differ.  The integrity check is that
        each scoped file declares the same symbol/timeframe as its filename or
        runtime directory, not that it matches the legacy root default file.
        """
        sync_status = {}
        patterns = (
            "training_stats_*_M*.json",
            "risk_metrics_*_M*.json",
            "performance_snapshot_*_M*.json",
            "production_metrics_*_M*.json",
            "paper_*_M*/training_stats.json",
            "paper_*_M*/risk_metrics.json",
            "paper_*_M*/performance_snapshot.json",
            "paper_*_M*/production_metrics.json",
            "live_*_M*/training_stats.json",
            "live_*_M*/risk_metrics.json",
            "live_*_M*/performance_snapshot.json",
            "live_*_M*/production_metrics.json",
        )
        for pattern in patterns:
            for scoped_path in self.data_dir.glob(pattern):
                if not scoped_path.is_file():
                    continue
                key = str(scoped_path.relative_to(self.data_dir))
                sync_status[key] = self._check_scoped_payload_file(scoped_path)
        return sync_status

    def check_backup_proliferation(self) -> dict[str, list[str]]:
        """Check for excessive .bak and .backup files."""
        backups = {}

        for backup_file in self.data_dir.glob("*.bak"):
            key = backup_file.name.split(".")[0]
            if key not in backups:
                backups[key] = []
            backups[key].append(backup_file.name)

        for backup_file in self.data_dir.glob("*.backup*"):
            key = backup_file.name.split(".")[0]
            if key not in backups:
                backups[key] = []
            backups[key].append(backup_file.name)

        if backups:
            total_backups = sum(len(v) for v in backups.values())
            self.warnings.append(
                f"⚠️  {total_backups} backup files detected across {len(backups)} data items. "
                f"Multiple versions in play. Recommend archival strategy."
            )

        return backups

    def check_position_consistency(self) -> dict[str, Any]:
        """Check position file integrity."""
        position = {}

        # Load active position
        pos_files = sorted(
            self.data_dir.glob("current_position*.json"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )

        if pos_files:
            pos_data = json.loads(pos_files[0].read_text())
            direction = str(pos_data.get("direction", "") or "").upper()
            position = {
                "file": pos_files[0].name,
                "direction": pos_data.get("direction"),
                "symbol": pos_data.get("symbol"),
                "has_quantity": "quantity" in pos_data or "qty" in pos_data,
                "entry_price": pos_data.get("entry_price"),
            }

            if direction != "FLAT" and not position["has_quantity"]:
                self.warnings.append(
                    f"⚠️  Position file {pos_files[0].name} missing quantity field"
                )

        return position

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive report."""
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "data_directory": str(self.data_dir),
            "trade_log_count": len(self.trade_log),
            "checks": {
                "entry_time_coverage": self.check_entry_time_coverage(),
                "quantity_coverage": self.check_quantity_coverage(),
                "pnl_consistency": self.check_pnl_consistency(),
                "file_staleness": self.check_file_staleness(),
                "multi_bot_sync": self.check_multi_bot_sync(),
                "backup_proliferation": self.check_backup_proliferation(),
                "position_consistency": self.check_position_consistency(),
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "infos": self.infos,
            "health_score": self._calculate_health_score(),
        }
        return report

    def _calculate_health_score(self) -> float:
        """Calculate overall HUD data health (0-100)."""
        base_score = 100.0

        # Deduct for critical issues
        base_score -= len(self.issues) * 15

        # Deduct for warnings
        base_score -= len(self.warnings) * 5

        # Bonus for no issues
        if not self.issues:
            base_score += 10

        return max(0, min(100, base_score))

    def print_report(self):
        """Print formatted report."""
        report = self.generate_report()

        print("\n" + "=" * 100)
        print("📋 HUD DATA VALIDATION REPORT")
        print("=" * 100)

        print(f"\nDate: {report['timestamp']}")
        print(f"Trade Log Entries: {report['trade_log_count']}")
        print(f"Health Score: {report['health_score']:.0f}/100")

        if report["issues"]:
            print(f"\n❌ CRITICAL ISSUES ({len(report['issues'])}):")
            for issue in report["issues"]:
                print(f"   {issue}")

        if report["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
            for warning in report["warnings"]:
                print(f"   {warning}")

        print("\n" + "=" * 100)

        return report

    def export_json(self, filepath: str):
        """Export report to JSON."""
        report = self.generate_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report exported to {filepath}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    validator = HUDDataValidator()
    report = validator.print_report()

    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        if idx + 1 < len(sys.argv):
            export_file = sys.argv[idx + 1]
            validator.export_json(export_file)

    # Exit with error code if issues found
    sys.exit(1 if validator.issues else 0)
