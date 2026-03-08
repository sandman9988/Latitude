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
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
    
    def check_entry_time_coverage(self) -> Tuple[float, int]:
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
    
    def check_quantity_coverage(self) -> Tuple[float, int]:
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
    
    def check_pnl_consistency(self) -> Dict[str, float]:
        """Check PnL across different sources."""
        if not self.trade_log:
            return {}
        
        current_pnl = sum(t.get("pnl", 0) for t in self.trade_log)
        original_pnl = sum(t.get("pnl_original", 0) for t in self.trade_log if "pnl_original" in t)
        recalc_count = sum(1 for t in self.trade_log if t.get("pnl_recalculated"))
        
        if recalc_count > 0:
            variance = abs(current_pnl - original_pnl)
            variance_pct = (variance / abs(original_pnl) * 100) if original_pnl != 0 else 0
            
            self.issues.append(
                f"⚠️  CRITICAL: {recalc_count} trades ({recalc_count/len(self.trade_log):.1%}) have recalculated PnL. "
                f"Current total: ${current_pnl:.2f}, Original: ${original_pnl:.2f}, "
                f"Variance: ${variance:.2f} ({variance_pct:.1f}%). Which is correct?"
            )
        
        result = {
            "current_pnl": current_pnl,
            "original_pnl": original_pnl,
            "recalc_count": recalc_count,
            "variance_usd": abs(current_pnl - original_pnl),
            "variance_pct": (abs(current_pnl - original_pnl) / abs(original_pnl) * 100) if original_pnl != 0 else 0,
        }
        return result
    
    def check_file_staleness(self) -> Dict[str, Dict[str, Any]]:
        """Check age of critical data files."""
        now = datetime.now(UTC)
        staleness = {}
        
        for filename in self.data_dir.glob("*.json"):
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
    
    def check_multi_bot_sync(self) -> Dict[str, bool]:
        """Check if per-bot files are in sync with defaults."""
        sync_status = {}
        
        # Compare training_stats.json with training_stats_*.json files
        default_ts = self.data_dir / "training_stats.json"
        if default_ts.exists():
            default_data = json.loads(default_ts.read_text())
            
            for per_bot_ts in self.data_dir.glob("training_stats_*_M*.json"):
                per_bot_data = json.loads(per_bot_ts.read_text())
                
                # Check if values match
                match = (
                    default_data.get("trigger_training_steps") == per_bot_data.get("trigger_training_steps")
                    and default_data.get("harvester_training_steps") == per_bot_data.get("harvester_training_steps")
                )
                
                sync_status[per_bot_ts.name] = match
                
                if not match:
                    self.warnings.append(
                        f"⚠️  MISMATCH: {per_bot_ts.name} differs from training_stats.json"
                    )
        
        # Compare risk_metrics.json with risk_metrics_*.json files
        default_rm = self.data_dir / "risk_metrics.json"
        if default_rm.exists():
            default_data = json.loads(default_rm.read_text())
            
            for per_bot_rm in self.data_dir.glob("risk_metrics_*_M*.json"):
                per_bot_data = json.loads(per_bot_rm.read_text())
                
                # Check if values match
                match = (
                    default_data.get("spread") == per_bot_data.get("spread")
                    and default_data.get("vpin") == per_bot_data.get("vpin")
                )
                
                sync_status[per_bot_rm.name] = match
                
                if not match:
                    self.warnings.append(
                        f"⚠️  MISMATCH: {per_bot_rm.name} differs from risk_metrics.json"
                    )
        
        return sync_status
    
    def check_backup_proliferation(self) -> Dict[str, List[str]]:
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
    
    def check_position_consistency(self) -> Dict[str, Any]:
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
            position = {
                "file": pos_files[0].name,
                "direction": pos_data.get("direction"),
                "symbol": pos_data.get("symbol"),
                "has_quantity": "quantity" in pos_data or "qty" in pos_data,
                "entry_price": pos_data.get("entry_price"),
            }
            
            if not position["has_quantity"]:
                self.warnings.append(
                    f"⚠️  Position file {pos_files[0].name} missing quantity field"
                )
        
        return position
    
    def generate_report(self) -> Dict[str, Any]:
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
