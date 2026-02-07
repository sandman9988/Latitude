"""
RiskManager - Central Risk Coordinator & Portfolio Controller

The "risk brain" of the trading system. Coordinates all risk management:
- Capital allocation via VaR-based position sizing
- Circuit breaker updates and control
- Comprehensive risk assessment and reporting
- Adaptive risk budget management
- Portfolio-level exposure coordination

Design Philosophy:
- Single point of control: All risk decisions flow through here
- Event-driven updates: Trades update circuit breakers automatically
- Adaptive allocation: Risk budget adjusts based on performance
- Portfolio awareness: Ready for multi-asset expansion
- Regime-adaptive: All limits adjust to market conditions
"""

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.risk.circuit_breakers import CircuitBreakerManager
from src.utils.safe_math import SafeMath
from src.risk.var_estimator import RegimeType, VaREstimator

LOG = logging.getLogger(__name__)


@dataclass
class EntryValidation:
    """Result of entry order validation"""

    approved: bool
    qty: float
    reason: str
    var_used: float = 0.0
    risk_budget_used: float = 0.0


@dataclass
class ExitValidation:
    """Result of exit order validation"""

    approved: bool
    volume: int
    urgency: str  # "NORMAL" | "EMERGENCY"
    reason: str


@dataclass
class ProbabilityCalibration:
    """Tracks prediction accuracy for self-calibration (per-agent)"""

    agent_id: str  # "trigger", "harvester", or "composite"
    confidence_bucket: float  # e.g., 0.7 for 70-79% confidence
    predicted_success_rate: float  # Agent's claimed probability
    actual_success_rate: float  # Observed win rate
    sample_size: int  # Number of trades in this bucket
    calibration_error: float  # |predicted - actual|
    is_well_calibrated: bool  # calibration_error < 0.1


@dataclass
class CorrelationBreakdown:
    """Flash crash / correlation breakdown detector"""

    timestamp: float
    avg_correlation: float  # Average pairwise correlation
    max_correlation: float  # Highest correlation observed
    breakdown_detected: bool  # True if all correlations → 1.0
    flash_crash_risk: str  # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    recommended_action: str  # "MONITOR" | "REDUCE_EXPOSURE" | "CLOSE_ALL"


@dataclass
class CompositeProbabilityPredictor:
    """Composite probability prediction combining all agents - MAIN PREDICTION TOOL"""

    trigger_calibration: Dict[float, ProbabilityCalibration]  # TriggerAgent predictions
    harvester_calibration: Dict[float, ProbabilityCalibration]  # HarvesterAgent predictions
    composite_calibration: Dict[float, ProbabilityCalibration]  # Combined predictions
    trigger_overall_accuracy: float  # Overall win rate for trigger
    harvester_overall_accuracy: float  # Overall win rate for harvester
    best_calibrated_agent: str  # "trigger" or "harvester"
    recommendation: str  # Which agent to trust more


@dataclass
class RiskAssessment:
    """Comprehensive portfolio risk assessment"""

    total_exposure_usd: float
    total_var_usd: float
    risk_utilization_pct: float  # % of risk budget used
    circuit_breaker_status: dict
    portfolio_health: str  # "HEALTHY" | "CAUTION" | "CRITICAL"
    position_concentration: float  # 0-1, higher = more concentrated
    regime_risk_multiplier: float  # Current regime adjustment
    recommendations: list[str]
    # RL & Calibration extensions
    probability_calibration: Optional[Dict[float, ProbabilityCalibration]] = None
    composite_predictor: Optional[CompositeProbabilityPredictor] = None  # Main prediction tool
    correlation_status: Optional[CorrelationBreakdown] = None
    rl_recommended_thresholds: Optional[Dict[str, float]] = None


class RiskManager:
    """
    Central Risk Coordinator - The "Risk Brain" of the Trading System

    Primary Responsibilities:
    1. Capital Allocation: VaR-based position sizing across portfolio
    2. Risk Assessment: Comprehensive portfolio health monitoring
    3. Circuit Breaker Control: Update breakers on trades, enforce stops
    4. Adaptive Risk Management: Adjust budgets based on performance
    5. Entry/Exit Validation: Gate all orders before execution
    6. Portfolio Coordination: Multi-asset exposure management

    Control Flow:
    - Agents -> validate_entry/exit() -> Orders (validation gate)
    - Trades -> on_trade_complete() -> Circuit breakers (updates)
    - Continuous -> assess_risk() -> Risk metrics (monitoring)
    - Performance -> adapt_risk_budget() -> Budget adjustments

    This is the SINGLE POINT OF CONTROL for all risk decisions.
    """

    def __init__(
        self,
        circuit_breakers: CircuitBreakerManager,
        var_estimator: VaREstimator,
        risk_budget_usd: float = 100.0,
        max_position_size: float = 1.0,
        min_confidence_entry: float = 0.6,
        min_confidence_exit: float = 0.5,
        symbol: str = "BTCUSD",
        timeframe: str = "M15",
        broker: str = "default",
    ):
        """
        Initialize RiskManager.

        Args:
            circuit_breakers: Circuit breaker manager instance
            var_estimator: VaR estimator instance
            risk_budget_usd: Maximum USD risk per position
            max_position_size: Maximum position size (lots)
            min_confidence_entry: Minimum confidence for entry
            min_confidence_exit: Minimum confidence for exit
            symbol: Trading symbol (for multi-asset coordination)
            timeframe: Timeframe (for context)
            broker: Broker name (for context)
        """
        self.circuit_breakers = circuit_breakers
        self.var_estimator = var_estimator
        self.risk_budget_usd = risk_budget_usd
        self.max_position_size = max_position_size
        self.min_confidence_entry = min_confidence_entry
        self.min_confidence_exit = min_confidence_exit
        self.symbol = symbol
        self.timeframe = timeframe
        self.broker = broker

        # Portfolio tracking (for multi-asset expansion)
        self.total_exposure_usd = 0.0
        self.active_positions = {}  # {symbol: position_size}

        # Statistics
        self.entries_approved = 0
        self.entries_rejected = 0
        self.exits_approved = 0
        self.exits_rejected = 0

        # Performance tracking for adaptive risk
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_equity = 10000.0  # Will be updated
        self.initial_risk_budget = risk_budget_usd

        # Risk assessment cache
        self._last_assessment: Optional[RiskAssessment] = None

        # === RL LEARNING COMPONENTS ===
        # Q-table: state -> action -> Q-value
        self.q_table: Dict[Tuple, Dict[str, float]] = {}
        self.learning_rate: float = 0.1
        self.discount_factor: float = 0.95
        self.exploration_rate: float = 0.15  # epsilon-greedy
        self.rl_enabled: bool = True
        self.rl_state_history: deque = deque(maxlen=1000)

        # === PROBABILITY CALIBRATION (PER-AGENT) ===
        # Separate tracking for TriggerAgent, HarvesterAgent, and Composite
        self.calibration_buckets_trigger: Dict[float, List[Tuple[float, bool]]] = {
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
            1.0: [],
        }
        self.calibration_buckets_harvester: Dict[float, List[Tuple[float, bool]]] = {
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
            1.0: [],
        }
        self.calibration_buckets_composite: Dict[float, List[Tuple[float, bool]]] = {
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
            1.0: [],
        }
        self.calibration_window: int = 100  # trades per agent

        # === CORRELATION MONITORING ===
        # Multi-symbol returns for correlation calculation
        self.returns_history: Dict[str, deque] = {}  # symbol -> returns
        self.correlation_window: int = 50  # bars
        self.correlation_matrix: Optional[np.ndarray] = None
        self.last_correlation_check: float = 0.0
        self.flash_crash_threshold: float = 0.85  # avg correlation > 0.85 = warning

        LOG.info(
            "[RISK] Initialized RiskManager (Central Risk Coordinator): budget=$%.2f max_size=%.4f symbol=%s",
            risk_budget_usd,
            max_position_size,
            symbol,
        )

    def validate_entry(
        self,
        action: int,
        confidence: float,
        current_position: float = 0.0,
        regime: RegimeType = RegimeType.CRITICAL,
        vpin_z: float = 0.0,
        current_vol: Optional[float] = None,
        account_balance: float = 10000.0,
        max_leverage: float = 100.0,
    ) -> EntryValidation:
        """
        Validate entry order per SYSTEM_FLOW.md specification.

        Validation Steps:
        1. Check circuit breakers → REJECT if tripped
        2. Calculate VaR-based position size
        3. Validate confidence threshold
        4. Check maximum position limits
        5. Account for current exposure

        Args:
            action: 0=NO_ENTRY, 1=LONG, 2=SHORT
            confidence: Agent confidence [0, 1]
            current_position: Current position size (signed)
            regime: Current market regime
            vpin_z: VPIN z-score for VaR adjustment
            current_vol: Current volatility (optional)
            account_balance: Account balance for position sizing
            max_leverage: Maximum allowed leverage

        Returns:
            EntryValidation with approval status, quantity, reason
        """
        # Step 1: Check circuit breakers
        if self.circuit_breakers.is_any_tripped():
            self.entries_rejected += 1
            breakers_status = self.circuit_breakers.get_status()
            # Extract tripped breaker names from status dict
            tripped = []
            for key in ["sortino", "kurtosis", "drawdown", "consecutive_losses"]:
                if key in breakers_status and breakers_status[key].get("tripped", False):
                    tripped.append(key)
            reason = f"Circuit breakers tripped: {', '.join(tripped)}"
            LOG.warning("[RISK] Entry REJECTED: %s", reason)
            return EntryValidation(approved=False, qty=0.0, reason=reason)

        # Step 2: Validate action
        if action == 0:  # NO_ENTRY
            return EntryValidation(approved=False, qty=0.0, reason="Agent decided NO_ENTRY")

        # Step 3: Validate confidence threshold
        if confidence < self.min_confidence_entry:
            self.entries_rejected += 1
            reason = f"Confidence {confidence:.3f} < threshold {self.min_confidence_entry:.3f}"
            LOG.info("[RISK] Entry rejected: %s", reason)
            return EntryValidation(approved=False, qty=0.0, reason=reason)

        # Step 4: Calculate VaR-based position size
        var_value = self.var_estimator.estimate_var(regime=regime, vpin_z=vpin_z, current_vol=current_vol)

        if var_value <= 0:
            self.entries_rejected += 1
            reason = "VaR calculation failed or zero"
            LOG.error("[RISK] Entry REJECTED: %s", reason)
            return EntryValidation(approved=False, qty=0.0, reason=reason)

        # Calculate position size from VaR
        # risk_budget_usd = var_value * position_size * contract_value
        # For simplicity: position_size = risk_budget_usd / (var_value * price)
        # Assuming BTCUSD, contract_size = 1 BTC
        from src.risk.var_estimator import position_size_from_var

        qty = position_size_from_var(
            var=var_value,  # Correct parameter name
            risk_budget_usd=self.risk_budget_usd,
            account_equity=account_balance,  # Correct parameter name
            contract_size=1.0,  # Add contract_size
            max_leverage=max(max_leverage, 1.0),
        )

        if qty <= 0:
            self.entries_rejected += 1
            reason = "Position size calculation resulted in zero/negative"
            LOG.error("[RISK] Entry REJECTED: %s", reason)
            return EntryValidation(approved=False, qty=0.0, reason=reason)

        # Step 5: Check maximum position limits
        if qty > self.max_position_size:
            qty_before = qty
            qty = self.max_position_size
            LOG.warning(
                "[RISK] Position size capped: %.4f → %.4f (max_position_size)",
                qty_before,
                qty,
            )

        # Step 6: Account for current exposure (for multi-asset)
        # If already in position, validate total exposure
        new_exposure = abs(current_position) + qty
        if new_exposure > self.max_position_size:
            self.entries_rejected += 1
            reason = f"Total exposure would exceed limit: current={abs(current_position):.4f} + new={qty:.4f} > max={self.max_position_size:.4f}"
            LOG.warning("[RISK] Entry REJECTED: %s", reason)
            return EntryValidation(approved=False, qty=0.0, reason=reason)

        # Step 7: APPROVED
        self.entries_approved += 1
        risk_used = var_value * qty
        LOG.info(
            "[RISK] Entry APPROVED: action=%d conf=%.3f qty=%.4f VaR=%.4f risk=$%.2f",
            action,
            confidence,
            qty,
            var_value,
            risk_used,
        )

        # Store decision metadata for RL feedback
        self._last_decision_type = "entry"
        self._last_decision_confidence = confidence

        return EntryValidation(
            approved=True,
            qty=qty,
            reason="Passed all validation checks",
            var_used=var_value,
            risk_budget_used=risk_used,
        )

    def validate_exit(
        self,
        action: int,
        exit_type: str = "FULL",
        current_position: float = 0.0,
        fraction: float = 1.0,
        min_position_size: float = 0.01,
    ) -> ExitValidation:
        """
        Validate exit order per SYSTEM_FLOW.md specification.

        Validation Steps:
        1. Check circuit breakers → If emergency, override to FULL close
        2. Validate partial close fraction
        3. Check minimum position size after partial
        4. Determine urgency level

        Args:
            action: 0=HOLD, 1=CLOSE
            exit_type: "FULL" | "PARTIAL" | "TRAILING"
            current_position: Current position size (signed)
            fraction: Fraction to close (for PARTIAL)
            min_position_size: Minimum allowed position size

        Returns:
            ExitValidation with approval status, volume, urgency
        """
        # Step 1: Check action
        if action == 0:  # HOLD
            self.exits_rejected += 1  # Count as rejection
            return ExitValidation(
                approved=False,
                volume=0,
                urgency="NORMAL",
                reason="Agent decided HOLD",
            )

        if abs(current_position) < min_position_size:
            self.exits_rejected += 1  # Count as rejection
            return ExitValidation(
                approved=False,
                volume=0,
                urgency="NORMAL",
                reason="No position to close",
            )

        # Step 2: Check circuit breakers for emergency override
        urgency = "NORMAL"
        if self.circuit_breakers.is_any_tripped():
            # Emergency: override to FULL close regardless of agent decision
            exit_type = "FULL"
            urgency = "EMERGENCY"
            breakers_status = self.circuit_breakers.get_status()
            # Extract tripped breaker names
            tripped = []
            for key in ["sortino", "kurtosis", "drawdown", "consecutive_losses"]:
                if key in breakers_status and breakers_status[key].get("tripped", False):
                    tripped.append(key)
            LOG.warning(
                "[RISK] Circuit breakers tripped: %s → EMERGENCY FULL CLOSE",
                ", ".join(tripped),
            )

        # Step 3: Calculate exit volume
        if exit_type == "FULL":
            volume = abs(current_position)
        elif exit_type == "PARTIAL":
            # Validate fraction
            if not (0.0 < fraction <= 1.0):
                self.exits_rejected += 1
                reason = f"Invalid partial fraction: {fraction:.3f}"
                LOG.error("[RISK] Exit REJECTED: %s", reason)
                return ExitValidation(
                    approved=False,
                    volume=0,
                    urgency=urgency,
                    reason=reason,
                )

            volume = abs(current_position) * fraction

            # Check minimum position size after partial
            remaining = abs(current_position) - volume
            if 0 < remaining < min_position_size:
                # Would leave dust position → force FULL close
                volume = abs(current_position)
                exit_type = "FULL"
                LOG.info(
                    "[RISK] Partial would leave dust (%.4f < %.4f) → upgrading to FULL",
                    remaining,
                    min_position_size,
                )
        elif exit_type == "TRAILING":
            # Trailing stop: full close
            volume = abs(current_position)
        else:
            self.exits_rejected += 1
            reason = f"Unknown exit_type: {exit_type}"
            LOG.error("[RISK] Exit REJECTED: %s", reason)
            return ExitValidation(
                approved=False,
                volume=0,
                urgency=urgency,
                reason=reason,
            )

        # Step 4: Validate volume
        if volume <= 0:
            self.exits_rejected += 1
            reason = "Calculated volume is zero/negative"
            LOG.error("[RISK] Exit REJECTED: %s", reason)
            return ExitValidation(
                approved=False,
                volume=0,
                urgency=urgency,
                reason=reason,
            )

        # Step 5: APPROVED
        self.exits_approved += 1

        # Convert to FIX lots (integer centi-lots)
        volume_lots = int(volume * 100)

        # Guard: tiny positions can round to 0 after int() conversion
        if volume_lots <= 0:
            self.exits_approved -= 1
            self.exits_rejected += 1
            reason = f"Volume rounds to zero after lot conversion (raw={volume:.6f})"
            LOG.error("[RISK] Exit REJECTED: %s", reason)
            return ExitValidation(
                approved=False,
                volume=0,
                urgency=urgency,
                reason=reason,
            )

        LOG.info(
            "[RISK] Exit APPROVED: type=%s volume=%.4f urgency=%s",
            exit_type,
            volume,
            urgency,
        )

        return ExitValidation(
            approved=True,
            volume=volume_lots,
            urgency=urgency,
            reason=f"{exit_type} exit approved",
        )

    def update_exposure(self, symbol: str, position_size: float):
        """
        Update portfolio exposure tracking (for multi-asset coordination).

        Args:
            symbol: Trading symbol
            position_size: Current position size (0 if flat)
        """
        if position_size == 0:
            self.active_positions.pop(symbol, None)
        else:
            self.active_positions[symbol] = position_size

        # Recalculate total exposure
        # For single-symbol, this is simple
        # For multi-asset, would aggregate across all symbols
        self.total_exposure_usd = sum(abs(pos) for pos in self.active_positions.values())

    def get_status(self) -> dict:
        """
        Get RiskManager status for monitoring/logging.

        Returns:
            Dict with current risk state
        """
        return {
            "risk_budget_usd": self.risk_budget_usd,
            "max_position_size": self.max_position_size,
            "total_exposure_usd": self.total_exposure_usd,
            "active_positions": len(self.active_positions),
            "entries_approved": self.entries_approved,
            "entries_rejected": self.entries_rejected,
            "exits_approved": self.exits_approved,
            "exits_rejected": self.exits_rejected,
            "circuit_breakers": self.circuit_breakers.get_status(),
        }

    def update_risk_budget(self, new_budget: float):
        """
        Update risk budget (for adaptive risk management).

        Args:
            new_budget: New risk budget in USD
        """
        if new_budget > 0:
            old_budget = self.risk_budget_usd
            self.risk_budget_usd = new_budget
            LOG.info(
                "[RISK] Risk budget updated: $%.2f → $%.2f",
                old_budget,
                new_budget,
            )
        else:
            LOG.error("[RISK] Invalid risk budget: %.2f (must be positive)", new_budget)

    def update_confidence_thresholds(self, entry: Optional[float] = None, exit: Optional[float] = None):
        """
        Update confidence thresholds (for adaptive tuning).

        Args:
            entry: New entry confidence threshold
            exit: New exit confidence threshold
        """
        if entry is not None and 0.0 <= entry <= 1.0:
            old = self.min_confidence_entry
            self.min_confidence_entry = entry
            LOG.info("[RISK] Entry confidence threshold: %.3f -> %.3f", old, entry)

        if exit is not None and 0.0 <= exit <= 1.0:
            old = self.min_confidence_exit
            self.min_confidence_exit = exit
            LOG.info("[RISK] Exit confidence threshold: %.3f -> %.3f", old, exit)

    # =========================================================================
    # CIRCUIT BREAKER CONTROL - Update breakers on trade completion
    # =========================================================================

    def on_trade_complete(self, pnl: float, equity: float, is_win: Optional[bool] = None):
        """
        Update circuit breakers and performance tracking on trade completion.

        This is the CENTRAL POINT where all trade results flow through.
        Updates circuit breakers, tracks performance, triggers adaptive adjustments.

        Args:
            pnl: Trade P&L (positive for profit, negative for loss)
            equity: Current account equity
            is_win: Optional explicit win/loss (if None, inferred from pnl)
        """
        # Update circuit breakers
        self.circuit_breakers.update_trade(pnl=pnl, equity=equity)
        self.circuit_breakers.check_all()  # Trigger breach detection

        # Track performance
        self.total_trades += 1
        self.total_pnl += pnl
        if is_win is None:
            is_win = pnl > 0
        if is_win:
            self.winning_trades += 1

        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Log trade impact
        win_rate = self.winning_trades / max(self.total_trades, 1)
        avg_pnl = self.total_pnl / max(self.total_trades, 1)

        LOG.info(
            "[RISK] Trade complete: PnL=%.2f, Equity=%.2f, WinRate=%.1f%%, AvgPnL=%.2f",
            pnl,
            equity,
            win_rate * 100,
            avg_pnl,
        )

        # === RL FEEDBACK LOOP ===
        # Feed outcome back for learning (if confidence/decision data available)
        # Note: Full integration requires storing decision metadata with trade
        # For now, we update based on win/loss at average confidence
        if hasattr(self, "_last_decision_confidence"):
            self.update_decision_outcome(
                decision_type=self._last_decision_type,
                confidence=self._last_decision_confidence,
                approved=True,  # Trade happened, so it was approved
                actual_outcome=is_win,
            )

        # Check if adaptive adjustment needed (every 10 trades)
        if self.total_trades % 10 == 0:
            self._consider_risk_adaptation(equity, win_rate)

    def _consider_risk_adaptation(self, equity: float, win_rate: float):
        """
        Consider adaptive risk budget adjustment based on performance.

        Logic:
        - Good performance (win rate > 55%, equity growing) → Increase budget
        - Poor performance (win rate < 45%, equity declining) → Decrease budget
        - Circuit breakers tripped → Reduce budget immediately
        """
        if self.circuit_breakers.is_any_tripped():
            # Emergency: reduce risk budget
            new_budget = self.risk_budget_usd * 0.75
            LOG.warning(
                "[RISK] Circuit breakers active → REDUCING risk budget: $%.2f → $%.2f",
                self.risk_budget_usd,
                new_budget,
            )
            self.update_risk_budget(new_budget)
            return

        # Calculate equity change
        equity_change = (equity - self.peak_equity) / max(self.peak_equity, 1.0)

        # Good performance: increase budget (max 1.5x initial)
        if win_rate > 0.55 and equity_change > 0.05:
            new_budget = min(self.risk_budget_usd * 1.1, self.initial_risk_budget * 1.5)
            if new_budget > self.risk_budget_usd:
                LOG.info(
                    "[RISK] Strong performance → INCREASING risk budget: $%.2f → $%.2f",
                    self.risk_budget_usd,
                    new_budget,
                )
                self.update_risk_budget(new_budget)

        # Poor performance: decrease budget (min 0.5x initial)
        elif win_rate < 0.45 or equity_change < -0.10:
            new_budget = max(self.risk_budget_usd * 0.9, self.initial_risk_budget * 0.5)
            if new_budget < self.risk_budget_usd:
                LOG.warning(
                    "[RISK] Weak performance → REDUCING risk budget: $%.2f → $%.2f",
                    self.risk_budget_usd,
                    new_budget,
                )
                self.update_risk_budget(new_budget)

    # =========================================================================
    # RISK ASSESSMENT - Comprehensive portfolio health monitoring
    # =========================================================================

    def assess_risk(
        self,
        current_regime: RegimeType = RegimeType.CRITICAL,
        current_vol: Optional[float] = None,
    ) -> RiskAssessment:
        """
        Provide comprehensive portfolio risk assessment.

        This is the main risk monitoring interface - call periodically
        to get full portfolio health status.

        Args:
            current_regime: Current market regime
            current_vol: Current volatility estimate

        Returns:
            RiskAssessment with comprehensive metrics and recommendations
        """
        # Calculate total VaR
        total_var = 0.0
        for symbol, position_size in self.active_positions.items():
            if position_size != 0:
                var_value = self.var_estimator.estimate_var(
                    regime=current_regime,
                    vpin_z=0.0,  # Would need per-symbol VPIN
                    current_vol=current_vol,
                )
                total_var += abs(position_size) * var_value

        # Risk utilization
        risk_utilization = SafeMath.safe_div(total_var * 100.0, self.risk_budget_usd, default=0.0)

        # Position concentration (0 = diversified, 1 = single position)
        if len(self.active_positions) == 0:
            concentration = 0.0
        elif len(self.active_positions) == 1:
            concentration = 1.0
        else:
            # Herfindahl index
            total_exposure = sum(abs(p) for p in self.active_positions.values())
            concentration = sum((abs(p) / max(total_exposure, 1e-8)) ** 2 for p in self.active_positions.values())

        # Regime risk multiplier
        regime_multipliers = {
            RegimeType.OVERDAMPED: 1.0,
            RegimeType.CRITICAL: 1.5,
            RegimeType.UNDERDAMPED: 2.0,
        }
        regime_mult = regime_multipliers.get(current_regime, 1.5)

        # Determine portfolio health
        cb_status = self.circuit_breakers.get_status()
        breakers_tripped = cb_status.get("any_tripped", False)

        if breakers_tripped:
            health = "CRITICAL"
        elif risk_utilization > 80 or concentration > 0.8:
            health = "CAUTION"
        else:
            health = "HEALTHY"

        # Generate recommendations
        recommendations = []
        if breakers_tripped:
            recommendations.append("STOP TRADING: Circuit breakers active")
        if risk_utilization > 90:
            recommendations.append("Risk budget nearly exhausted - avoid new positions")
        if concentration > 0.9:
            recommendations.append("High concentration - consider diversification")
        if current_regime == RegimeType.UNDERDAMPED:
            recommendations.append("High volatility regime - reduce position sizes")
        if len(recommendations) == 0:
            recommendations.append("Portfolio health normal")

        # === EXTENDED RISK METRICS ===
        # Composite Probability Predictor - MAIN PREDICTION TOOL
        try:
            composite_predictor = self.get_composite_probability_predictor()

            # Add per-agent calibration warnings
            if composite_predictor.trigger_calibration:
                for bucket, calib in composite_predictor.trigger_calibration.items():
                    if not calib.is_well_calibrated and calib.sample_size > 10:
                        recommendations.append(
                            f"[TRIGGER] Miscalibrated at {bucket:.0%}: "
                            f"{calib.predicted_success_rate:.0%} vs {calib.actual_success_rate:.0%}"
                        )

            if composite_predictor.harvester_calibration:
                for bucket, calib in composite_predictor.harvester_calibration.items():
                    if not calib.is_well_calibrated and calib.sample_size > 10:
                        recommendations.append(
                            f"[HARVESTER] Miscalibrated at {bucket:.0%}: "
                            f"{calib.predicted_success_rate:.0%} vs {calib.actual_success_rate:.0%}"
                        )

            # Add composite recommendation
            if composite_predictor.recommendation and "Insufficient" not in composite_predictor.recommendation:
                recommendations.append(f"📊 {composite_predictor.recommendation}")
        except Exception as e:
            LOG.debug(f"[RISK] Could not generate composite predictor: {e}")
            composite_predictor = None

        # Legacy calibration data (for backward compatibility)
        calibration_data = self.get_probability_calibration("composite")

        # Correlation breakdown check
        import time

        correlation_status = self.check_correlation_breakdown(current_time=time.time())
        if correlation_status and correlation_status.breakdown_detected:
            recommendations.insert(0, f"⚠️  CORRELATION BREAKDOWN: {correlation_status.recommended_action}")

        # RL threshold recommendations
        rl_thresholds = self.get_rl_recommended_thresholds()
        if rl_thresholds.get("confidence", 0) > 0.7:
            recommendations.append(
                f"RL suggests: entry={rl_thresholds['entry_threshold']:.2f} "
                f"exit={rl_thresholds['exit_threshold']:.2f}"
            )

        assessment = RiskAssessment(
            total_exposure_usd=self.total_exposure_usd,
            total_var_usd=total_var,
            risk_utilization_pct=risk_utilization,
            circuit_breaker_status=cb_status,
            portfolio_health=health,
            position_concentration=concentration,
            regime_risk_multiplier=regime_mult,
            recommendations=recommendations,
            # RL & Calibration extensions
            probability_calibration=calibration_data if calibration_data else None,
            composite_predictor=composite_predictor,  # Main prediction tool
            correlation_status=correlation_status,
            rl_recommended_thresholds=rl_thresholds,
        )

        self._last_assessment = assessment

        LOG.debug(
            "[RISK ASSESSMENT] Health=%s, Utilization=%.1f%%, Concentration=%.2f, Regime=%.1fx",
            health,
            risk_utilization,
            concentration,
            regime_mult,
        )

        return assessment

    def get_risk_summary(self) -> str:
        """Get human-readable risk summary"""
        if self._last_assessment is None:
            return "No risk assessment available"

        a = self._last_assessment
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║           RISK MANAGER - PORTFOLIO STATUS                ║
╠══════════════════════════════════════════════════════════╣
║ Portfolio Health:     {a.portfolio_health:>30} ║
║ Total Exposure:       ${a.total_exposure_usd:>28.2f} ║
║ Total VaR:            ${a.total_var_usd:>28.2f} ║
║ Risk Utilization:     {a.risk_utilization_pct:>27.1f}% ║
║ Concentration:        {a.position_concentration:>30.2f} ║
║ Regime Multiplier:    {a.regime_risk_multiplier:>29.1f}x ║
╠══════════════════════════════════════════════════════════╣
║ Statistics:                                              ║
║   Total Trades:       {self.total_trades:>30} ║
║   Win Rate:           {(self.winning_trades/max(self.total_trades,1)*100):>27.1f}% ║
║   Total P&L:          ${self.total_pnl:>28.2f} ║
║   Entries Approved:   {self.entries_approved:>30} ║
║   Entries Rejected:   {self.entries_rejected:>30} ║
║   Exits Approved:     {self.exits_approved:>30} ║
║   Exits Rejected:     {self.exits_rejected:>30} ║
╠══════════════════════════════════════════════════════════╣
║ Recommendations:                                         ║
"""
        for rec in a.recommendations:
            summary += f"║   • {rec:<53} ║\n"
        summary += "╚══════════════════════════════════════════════════════════╝"

        return summary

    # ============================================================================
    # RL LEARNING & SELF-IMPROVEMENT
    # ============================================================================

    def update_decision_outcome(
        self,
        decision_type: str,  # "entry" | "exit"
        confidence: float,
        approved: bool,
        actual_outcome: Optional[bool] = None,  # True = profitable, False = loss
        agent_id: str = "composite",  # "trigger", "harvester", or "composite"
    ) -> None:
        """
        Feed decision outcomes back into RL and probability calibration.

        This creates the self-learning feedback loop:
        1. Track prediction accuracy by confidence level (PER AGENT)
        2. Update Q-table for threshold optimization
        3. Calibrate probability estimates separately for each agent

        Args:
            decision_type: "entry" or "exit"
            confidence: Agent's confidence [0-1]
            approved: Whether RiskManager approved it
            actual_outcome: Trade result (True=win, False=loss, None=pending)
            agent_id: Which agent made the prediction ("trigger"/"harvester"/"composite")
        """
        # Update probability calibration (PER AGENT)
        if actual_outcome is not None:
            bucket = self._get_confidence_bucket(confidence)

            # Select appropriate bucket based on agent
            if agent_id == "trigger":
                target_buckets = self.calibration_buckets_trigger
            elif agent_id == "harvester":
                target_buckets = self.calibration_buckets_harvester
            else:
                target_buckets = self.calibration_buckets_composite

            target_buckets[bucket].append((confidence, actual_outcome))

            # Keep only recent history
            if len(target_buckets[bucket]) > self.calibration_window:
                target_buckets[bucket].pop(0)

            LOG.debug(
                "[RL FEEDBACK] agent=%s %s confidence=%.2f approved=%s outcome=%s bucket=%.1f",
                agent_id,
                decision_type.upper(),
                confidence,
                approved,
                actual_outcome,
                bucket,
            )

        # RL state update (run after enough data collected)
        if self.rl_enabled and actual_outcome is not None:
            self._update_q_learning(decision_type, confidence, approved, actual_outcome)

    def _get_confidence_bucket(self, confidence: float) -> float:
        """Map confidence to calibration bucket (0.5, 0.6, ..., 1.0)"""
        if confidence < 0.55:
            return 0.5
        elif confidence < 0.65:
            return 0.6
        elif confidence < 0.75:
            return 0.7
        elif confidence < 0.85:
            return 0.8
        elif confidence < 0.95:
            return 0.9
        else:
            return 1.0

    def get_probability_calibration(self, agent_id: str = "composite") -> Dict[float, ProbabilityCalibration]:
        """
        Analyze how well-calibrated agent probabilities are.

        Args:
            agent_id: "trigger", "harvester", or "composite"

        Returns:
            Dict of confidence_bucket -> ProbabilityCalibration

        Well-calibrated: A model that predicts 70% confidence should win ~70% of the time
        """
        # Select appropriate buckets
        if agent_id == "trigger":
            buckets = self.calibration_buckets_trigger
        elif agent_id == "harvester":
            buckets = self.calibration_buckets_harvester
        else:
            buckets = self.calibration_buckets_composite

        calibration_report = {}

        for bucket, outcomes in buckets.items():
            if len(outcomes) < 5:  # Need minimum samples
                continue

            confidences, results = zip(*outcomes)
            predicted_rate = np.mean(confidences)
            actual_rate = np.mean([1.0 if r else 0.0 for r in results])
            calibration_error = abs(predicted_rate - actual_rate)

            calibration_report[bucket] = ProbabilityCalibration(
                agent_id=agent_id,
                confidence_bucket=bucket,
                predicted_success_rate=predicted_rate,
                actual_success_rate=actual_rate,
                sample_size=len(outcomes),
                calibration_error=calibration_error,
                is_well_calibrated=calibration_error < 0.1,  # Within 10%
            )

        return calibration_report

    def get_composite_probability_predictor(self) -> CompositeProbabilityPredictor:
        """
        Get composite probability prediction combining all agents.

        This is the MAIN RISK MANAGEMENT TOOL for probability predictions.
        It shows:
        - TriggerAgent's prediction accuracy
        - HarvesterAgent's prediction accuracy
        - Combined/composite predictions
        - Which agent to trust more

        Returns:
            CompositeProbabilityPredictor with full analysis
        """
        # Get calibration for each agent
        trigger_calib = self.get_probability_calibration("trigger")
        harvester_calib = self.get_probability_calibration("harvester")
        composite_calib = self.get_probability_calibration("composite")

        # Calculate overall accuracy for each agent
        trigger_total = sum(len(outcomes) for outcomes in self.calibration_buckets_trigger.values())
        trigger_wins = sum(
            sum(1 for _, outcome in outcomes if outcome) for outcomes in self.calibration_buckets_trigger.values()
        )
        trigger_accuracy = trigger_wins / max(trigger_total, 1)

        harvester_total = sum(len(outcomes) for outcomes in self.calibration_buckets_harvester.values())
        harvester_wins = sum(
            sum(1 for _, outcome in outcomes if outcome) for outcomes in self.calibration_buckets_harvester.values()
        )
        harvester_accuracy = harvester_wins / max(harvester_total, 1)

        # Determine best calibrated agent
        trigger_avg_error = np.mean([c.calibration_error for c in trigger_calib.values()]) if trigger_calib else 1.0
        harvester_avg_error = (
            np.mean([c.calibration_error for c in harvester_calib.values()]) if harvester_calib else 1.0
        )

        best_calibrated = "trigger" if trigger_avg_error < harvester_avg_error else "harvester"

        # Generate recommendation
        if trigger_total < 10 and harvester_total < 10:
            recommendation = "Insufficient data - need more trades"
        elif trigger_avg_error < 0.1 and harvester_avg_error < 0.1:
            recommendation = "Both agents well-calibrated - trust both equally"
        elif trigger_avg_error < harvester_avg_error * 0.7:
            recommendation = f"Trust TriggerAgent more (error: {trigger_avg_error:.1%} vs {harvester_avg_error:.1%})"
        elif harvester_avg_error < trigger_avg_error * 0.7:
            recommendation = f"Trust HarvesterAgent more (error: {harvester_avg_error:.1%} vs {trigger_avg_error:.1%})"
        else:
            recommendation = "Both agents similarly calibrated - weight equally"

        LOG.info(
            "[COMPOSITE PREDICTOR] Trigger: %.1f%% acc (error=%.1f%%) | Harvester: %.1f%% acc (error=%.1f%%) | Best: %s",
            trigger_accuracy * 100,
            trigger_avg_error * 100,
            harvester_accuracy * 100,
            harvester_avg_error * 100,
            best_calibrated,
        )

        return CompositeProbabilityPredictor(
            trigger_calibration=trigger_calib,
            harvester_calibration=harvester_calib,
            composite_calibration=composite_calib,
            trigger_overall_accuracy=trigger_accuracy,
            harvester_overall_accuracy=harvester_accuracy,
            best_calibrated_agent=best_calibrated,
            recommendation=recommendation,
        )

    def _update_q_learning(self, decision_type: str, confidence: float, approved: bool, outcome: bool) -> None:
        """
        Update Q-table for threshold optimization.

        State: (drawdown_level, win_rate_bucket, confidence_bucket)
        Action: (threshold_adjustment: -0.1, 0.0, +0.1)
        Reward: +1 for correct decision, -1 for incorrect
        """
        # Current state
        drawdown_pct = ((self.peak_equity - (self.peak_equity + self.total_pnl)) / self.peak_equity) * 100
        drawdown_level = int(drawdown_pct / 5)  # 0-5% = 0, 5-10% = 1, etc.
        win_rate = self.winning_trades / max(self.total_trades, 1)
        win_bucket = int(win_rate * 10)  # 0-10% = 0, 10-20% = 1, etc.
        conf_bucket = self._get_confidence_bucket(confidence)

        state = (drawdown_level, win_bucket, conf_bucket)

        # Reward: Did we make the right approval decision?
        if approved and outcome:  # Approved and won
            reward = 1.0
        elif not approved and not outcome:  # Rejected and would have lost
            reward = 0.5
        elif approved and not outcome:  # Approved and lost
            reward = -1.0
        else:  # not approved and outcome (missed opportunity)
            reward = -0.5

        # Initialize Q-values for this state if needed
        if state not in self.q_table:
            self.q_table[state] = {
                "lower_threshold": 0.0,  # -0.1
                "keep_threshold": 0.0,  # 0.0
                "raise_threshold": 0.0,  # +0.1
            }

        # Determine which action was implicitly taken
        # (This is simplified - full implementation would track explicit actions)
        action = "keep_threshold"

        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        old_q = self.q_table[state][action]
        # Simplified: no next state (episodic)
        new_q = old_q + self.learning_rate * (reward - old_q)
        self.q_table[state][action] = new_q

        LOG.debug(
            "[RL Q-UPDATE] state=%s action=%s reward=%.1f Q: %.3f → %.3f",
            state,
            action,
            reward,
            old_q,
            new_q,
        )

        # Store state for analysis
        self.rl_state_history.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "q_value": new_q,
                "confidence": confidence,
                "approved": approved,
                "outcome": outcome,
            }
        )

    def get_rl_recommended_thresholds(self) -> Dict[str, float]:
        """
        Get RL-recommended threshold adjustments based on learned Q-values.

        Returns:
            {
                "entry_threshold": 0.65,  # Recommended entry threshold
                "exit_threshold": 0.55,   # Recommended exit threshold
                "confidence": 0.8         # How confident in recommendation (0-1)
            }
        """
        if len(self.q_table) < 10:  # Need sufficient learning
            return {
                "entry_threshold": self.min_confidence_entry,
                "exit_threshold": self.min_confidence_exit,
                "confidence": 0.0,
                "reason": "Insufficient learning data",
            }

        # Aggregate Q-values across all states to find best actions
        action_scores = {"lower_threshold": [], "keep_threshold": [], "raise_threshold": []}

        for state, actions in self.q_table.items():
            for action, q_val in actions.items():
                action_scores[action].append(q_val)

        # Average Q-value for each action
        avg_q = {action: np.mean(scores) if scores else 0.0 for action, scores in action_scores.items()}

        # Best action
        best_action = max(avg_q, key=avg_q.get)
        confidence = abs(avg_q[best_action]) / (sum(abs(v) for v in avg_q.values()) + 1e-6)

        # Apply recommendation
        adjustment = 0.0
        if best_action == "lower_threshold":
            adjustment = -0.05
        elif best_action == "raise_threshold":
            adjustment = 0.05

        return {
            "entry_threshold": max(0.5, min(0.9, self.min_confidence_entry + adjustment)),
            "exit_threshold": max(0.4, min(0.8, self.min_confidence_exit + adjustment)),
            "confidence": confidence,
            "reason": f"RL recommends: {best_action} (Q={avg_q[best_action]:.3f})",
        }

    # ============================================================================
    # CORRELATION MONITORING & FLASH CRASH DETECTION
    # ============================================================================

    def update_returns(self, symbol: str, price_return: float) -> None:
        """
        Update returns history for correlation calculation.

        Args:
            symbol: Trading symbol
            price_return: Price return for this bar (e.g., 0.01 for +1%)
        """
        if symbol not in self.returns_history:
            self.returns_history[symbol] = deque(maxlen=self.correlation_window)

        self.returns_history[symbol].append(price_return)

        LOG.debug("[CORRELATION] Updated %s return: %.4f", symbol, price_return)

    def check_correlation_breakdown(self, current_time: float) -> Optional[CorrelationBreakdown]:
        """
        Detect correlation breakdowns (flash crash indicator).

        Flash crashes occur when:
        1. All asset correlations suddenly approach 1.0
        2. Diversification benefits disappear
        3. Everything moves together (panic selling)

        Returns:
            CorrelationBreakdown if multi-symbol data available, None otherwise
        """
        # Need at least 2 symbols and minimum history
        if len(self.returns_history) < 2:
            return None

        symbols_with_data = [
            sym for sym, returns in self.returns_history.items() if len(returns) >= min(20, self.correlation_window)
        ]

        if len(symbols_with_data) < 2:
            return None

        # Build returns matrix
        returns_matrix = []
        for sym in symbols_with_data:
            returns_matrix.append(list(self.returns_history[sym]))

        # Ensure equal length (use minimum available)
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = [r[-min_length:] for r in returns_matrix]

        # Calculate correlation matrix
        returns_array = np.array(returns_matrix)
        self.correlation_matrix = np.corrcoef(returns_array)

        # Get off-diagonal correlations (exclude self-correlation)
        n = len(symbols_with_data)
        off_diag_correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag_correlations.append(self.correlation_matrix[i, j])

        avg_correlation = np.mean(np.abs(off_diag_correlations))
        max_correlation = np.max(np.abs(off_diag_correlations))

        # Flash crash detection
        breakdown_detected = avg_correlation > self.flash_crash_threshold

        # Risk level
        if avg_correlation > 0.95:
            risk = "CRITICAL"
            action = "CLOSE_ALL"
        elif avg_correlation > 0.90:
            risk = "HIGH"
            action = "REDUCE_EXPOSURE"
        elif avg_correlation > 0.85:
            risk = "MODERATE"
            action = "REDUCE_EXPOSURE"
        else:
            risk = "LOW"
            action = "MONITOR"

        self.last_correlation_check = current_time

        breakdown = CorrelationBreakdown(
            timestamp=current_time,
            avg_correlation=avg_correlation,
            max_correlation=max_correlation,
            breakdown_detected=breakdown_detected,
            flash_crash_risk=risk,
            recommended_action=action,
        )

        if breakdown_detected:
            LOG.warning(
                "🚨 [CORRELATION BREAKDOWN] Avg=%.3f Max=%.3f Risk=%s Action=%s",
                avg_correlation,
                max_correlation,
                risk,
                action,
            )

        return breakdown

    def allocate_capital_by_correlation(self, symbols: List[str], total_capital: float) -> Dict[str, float]:
        """
        Allocate capital using negative correlation for diversification.

        Strategy:
        - Prefer negatively correlated assets (hedge each other)
        - Penalize highly correlated assets (concentration risk)
        - Allocate more capital to uncorrelated/negatively correlated pairs

        Args:
            symbols: List of symbols to allocate across
            total_capital: Total capital to allocate (USD)

        Returns:
            {symbol: allocated_capital_usd}
        """
        if len(symbols) < 2 or self.correlation_matrix is None:
            # Equal allocation fallback
            equal_alloc = total_capital / len(symbols)
            return {sym: equal_alloc for sym in symbols}

        # Calculate diversification score for each symbol
        # Higher score = better diversification (more negative/low correlations)
        symbols_with_data = [
            sym for sym in symbols if sym in self.returns_history and len(self.returns_history[sym]) >= 20
        ]

        if len(symbols_with_data) < 2:
            equal_alloc = total_capital / len(symbols)
            return {sym: equal_alloc for sym in symbols}

        # Diversification score: average of (1 - |correlation|) with other assets
        # Higher score = less correlated = better diversification
        div_scores = {}
        for i, sym in enumerate(symbols_with_data):
            correlations_with_others = []
            for j, other_sym in enumerate(symbols_with_data):
                if i != j:
                    corr = self.correlation_matrix[i, j]
                    # Reward negative correlation, penalize positive correlation
                    div_score = 1.0 - corr  # Range: [0, 2], higher = better
                    correlations_with_others.append(div_score)

            div_scores[sym] = np.mean(correlations_with_others)

        # Normalize scores to sum to 1.0
        total_score = sum(div_scores.values())
        normalized_weights = {sym: score / total_score for sym, score in div_scores.items()}

        # Allocate capital
        allocation = {sym: total_capital * weight for sym, weight in normalized_weights.items()}

        # Add remaining symbols with equal allocation (if any)
        remaining_symbols = set(symbols) - set(symbols_with_data)
        if remaining_symbols:
            remaining_capital = total_capital * 0.1  # Reserve 10% for uncorrelated
            equal_remaining = remaining_capital / len(remaining_symbols)
            for sym in remaining_symbols:
                allocation[sym] = equal_remaining

            # Reduce primary allocations proportionally
            reduction_factor = (total_capital - remaining_capital) / total_capital
            for sym in symbols_with_data:
                allocation[sym] *= reduction_factor

        LOG.info("[CAPITAL ALLOCATION] Correlation-based allocation:")
        for sym, amount in sorted(allocation.items(), key=lambda x: -x[1]):
            div_score = div_scores.get(sym, 0.0)
            LOG.info("  %s: $%.2f (div_score=%.3f)", sym, amount, div_score)

        return allocation
