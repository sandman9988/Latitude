#!/usr/bin/env python3
"""
Reward Shaper - Asymmetric component-based reward calculation
Python port of MASTER_HANDBOOK.md Section 4.6 - Reward Shaping

Implements three reward components:
1. Capture Efficiency: Rewards capturing high % of MFE
2. Winner-to-Loser Penalty: Punishes giving back profits
3. Opportunity Cost: Penalizes missing potential profits
4. Activity Bonus: Rewards action when stagnant (NEW)
5. Counterfactual Adjustment: Penalty for early exits (NEW)
6. Ensemble Disagreement: Rewards exploration in uncertain states (NEW)

All weights are adaptive per instrument (NO MAGIC NUMBERS principle).
Uses LearnedParametersManager for DRY compliance.
"""

import math
from typing import Any

from src.monitoring.activity_monitor import ActivityMonitor, CounterfactualAnalyzer
from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.safe_math import SafeMath

TARGET_CAPTURE_RATIO: float = 0.7
WTL_THRESHOLD: float = 1.0  # Relative: penalize WTL on any scale
# BASELINE_MFE and OPPORTUNITY_THRESHOLD were formerly hardcoded (100.0, 50.0).
# They are now learned per-instrument via LearnedParametersManager so the
# reward shaper is instrument-agnostic (works on XAUUSD, EURUSD, BTC, etc.).
# The values below are used only during the very first trades before live data
# has been observed. After ~20 trades they will have fully self-calibrated.
BASELINE_MFE_SEED: float = 10.0  # Neutral seed — updated by first real trades
OPPORTUNITY_THRESHOLD_SEED: float = 15.0  # Neutral seed
OPPORTUNITY_SIGNAL_MIN: float = 0.5
OPPORTUNITY_SCALE: float = 0.3
WEIGHT_CAPTURE: float = 1.0
WEIGHT_WTL: float = 1.0
WEIGHT_OPPORTUNITY: float = 0.5
WEIGHT_ACTIVITY: float = 0.8
WEIGHT_COUNTERFACTUAL: float = 0.6
WEIGHT_ENSEMBLE: float = 0.4
RUNWAY_MULT_DEFAULT: float = 2.0
RUNWAY_PENALTY_INVALID: float = -2.0
RUNWAY_LOG_PENALTY: float = -5.0
RUNWAY_CLAMP_ABS: float = 3.0
OPPORTUNITY_BONUS_DEFAULT: float = 0.0
RUNWAY_EXCELLENT_MIN: float = 0.8
RUNWAY_GOOD_MIN: float = 0.6
RUNWAY_FAIR_MIN: float = 0.4
WTL_MULT_DEFAULT: float = 3.0
CAPTURE_MULT_FALLBACK: float = 2.0
TIMING_PENALTY_SCALE: float = -1.5  # Increased from -0.5 for stronger late-exit penalty
RUNWAY_EXPECTED_GAIN_MULT: float = 2.0
RUNWAY_EXPECTED_LOSS_MULT: float = 1.0
FRICTION_COST_MULT: float = 0.1

# Undeveloped-MFE penalty: rewards based on how much of MFE was realised
# (timeframe-agnostic — no bar counts).  Fires when MFE existed but most
# of the move was surrendered (high MAE relative to MFE).
UNDEVELOPED_MFE_PENALTY_SCALE: float = -1.0  # max penalty at full giveback
ZERO_MFE_PENALTY: float = -0.3  # flat penalty when MFE ≤ 0

# Session quality multiplier: MFE during high-liquidity sessions is "worth
# more" because the signal is cleaner and slippage lower.  Pure results
# weighting — no bar counting.
SESSION_BONUS_OVERLAP: float = 1.3   # London/NY overlap
SESSION_BONUS_LONDON: float = 1.15   # London session
SESSION_BONUS_NY: float = 1.15       # New York session
SESSION_BONUS_OFFPEAK: float = 0.85  # Asian/overnight

# Runway quality thresholds (for trigger reward calculation)
RUNWAY_EXCELLENT_MAX: float = 1.2
RUNWAY_GOOD_MAX: float = 1.5
RUNWAY_OVERPREDICT_THRESHOLD: float = 0.5

# Capture quality thresholds (for harvester reward quality assessment)
CAPTURE_QUALITY_EXCELLENT: float = 0.8
CAPTURE_QUALITY_GOOD: float = 0.6
CAPTURE_QUALITY_FAIR: float = 0.4


class RewardShaper:
    """
    Asymmetric reward shaper for DDQN training.
    Implements component-based rewards with adaptive weights.

    Now uses LearnedParametersManager (DRY - single source of truth)
    Includes activity monitoring and counterfactual analysis
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",  # Instrument-agnostic: default for tests/demos
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: LearnedParametersManager | None = None,
        activity_monitor: ActivityMonitor | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.broker = broker

        # Use shared parameter manager (DRY)
        if param_manager is None:
            self.param_manager = LearnedParametersManager()
            self.param_manager.load()  # Load existing parameters if available
        else:
            self.param_manager = param_manager

        self._param_cache = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "broker": self.broker,
        }

        # Activity monitoring (prevent learned helplessness)
        self.activity_monitor = activity_monitor or ActivityMonitor()

        # Counterfactual analysis (optimal vs actual exit)
        self.counterfactual = CounterfactualAnalyzer()

        # Adaptive component weights: stored in LearnedParametersManager so
        # they persist and self-calibrate.  Bounded to [0.2, 2.0] so no
        # component can be zeroed out or dominate entirely.
        self._weight_names = [
            ("reward_weight_capture", WEIGHT_CAPTURE),
            ("reward_weight_wtl", WEIGHT_WTL),
            ("reward_weight_opportunity", WEIGHT_OPPORTUNITY),
            ("reward_weight_activity", WEIGHT_ACTIVITY),
            ("reward_weight_counterfactual", WEIGHT_COUNTERFACTUAL),
            ("reward_weight_ensemble", WEIGHT_ENSEMBLE),
        ]

        # Session quality engine (optional — used to weight MFE value by session)
        self._event_engine: Any | None = None

        # Statistics for monitoring
        self.total_rewards_calculated = 0
        self.component_stats = {
            "capture": {"sum": 0.0, "count": 0},
            "wtl": {"sum": 0.0, "count": 0},
            "opportunity": {"sum": 0.0, "count": 0},
            "activity": {"sum": 0.0, "count": 0},
            "counterfactual": {"sum": 0.0, "count": 0},
            "ensemble": {"sum": 0.0, "count": 0},  # NEW: Ensemble disagreement bonus
        }

    def _get_weight(self, param_name: str, default: float) -> float:
        """Get adaptive weight, bounded to [0.2, 2.0]."""
        raw = self._get_param(param_name, default)
        if raw is None:
            return default
        return max(0.2, min(2.0, float(raw)))

    def _get_param(self, name: str, default: float | None = None) -> float:
        return self.param_manager.get(self.symbol, name, timeframe=self.timeframe, broker=self.broker, default=default)

    def set_event_engine(self, engine: Any) -> None:
        """Inject an EventTimeFeatureEngine for session-aware MFE weighting."""
        self._event_engine = engine

    def _get_session_quality(self, exit_time: str) -> float:
        """Return a multiplier reflecting session liquidity quality.

        London/NY overlap → highest quality (cleaner MFE, lower slippage).
        London or NY solo → above-average.
        Off-peak (Asian/overnight) → below-average.

        Returns 1.0 when no event engine is available or timestamp is empty.
        """
        if not exit_time or self._event_engine is None:
            return 1.0
        try:
            from datetime import datetime, timezone

            if isinstance(exit_time, str):
                dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
            else:
                dt = exit_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            feats = self._event_engine.compute(dt)
            # feats is a dict with keys like london_active, ny_active, london_ny_overlap...
            if feats.get("london_ny_overlap", 0.0) > 0.5:
                return SESSION_BONUS_OVERLAP
            if feats.get("london_active", 0.0) > 0.5:
                return SESSION_BONUS_LONDON
            if feats.get("ny_active", 0.0) > 0.5:
                return SESSION_BONUS_NY
            return SESSION_BONUS_OFFPEAK
        except Exception:
            return 1.0

    def calculate_capture_efficiency_reward(self, exit_pnl: float, mfe: float) -> float:
        """
        Reward based on how much of MFE was captured at exit.

        Formula from handbook:
        capture_ratio = exit_pnl / mfe
        r_capture = (capture_ratio - target_capture) * multiplier

        Args:
            exit_pnl: Final P&L at trade exit
            mfe: Maximum favorable excursion during trade

        Returns:
            Capture efficiency reward (positive if above target, negative if below)
        """
        if mfe <= 0:
            return 0.0

        # Get adaptive parameters
        capture_mult = self._get_param("capture_multiplier")
        target_capture = TARGET_CAPTURE_RATIO  # Principled default from handbook: aim for 70% MFE capture

        # Clamp ratio to prevent explosion when mfe is tiny or pnl/mfe have different scales.
        # A ratio outside [-5, 5] carries no additional discriminative signal for the DDQN
        # (both -5 and -355 mean "catastrophic loss relative to MFE").
        raw_ratio = exit_pnl / mfe
        capture_ratio = max(-5.0, min(5.0, raw_ratio))

        # Reward = difference from target × multiplier, bounded for DDQN stability
        reward = max(-3.0, min(3.0, (capture_ratio - target_capture) * capture_mult))

        # Track statistics
        self.component_stats["capture"]["sum"] += reward
        self.component_stats["capture"]["count"] += 1

        return reward

    def calculate_wtl_penalty(
        self, was_wtl: bool, mfe: float, exit_pnl: float, bars_from_mfe_to_exit: int = 0
    ) -> float:
        """
        Penalty for Winner-to-Loser trades (had profit, ended in loss).

        Formula from handbook:
        if was_winner_to_loser AND mfe > threshold:
            mfe_normalized = mfe / baseline_mfe
            giveback_ratio = (mfe - exit_pnl) / mfe
            time_penalty = 1 + (bars_from_mfe_to_exit / 10)
            r_wtl = -mfe_normalized * giveback_ratio * penalty_mult * time_penalty

        Args:
            was_wtl: Boolean flag from MFEMAETracker
            mfe: Maximum favorable excursion
            exit_pnl: Final P&L (negative for WTL)
            bars_from_mfe_to_exit: Time elapsed from MFE peak to exit

        Returns:
            Penalty (negative reward) for WTL, 0 otherwise
        """
        # Get adaptive parameters
        wtl_penalty_mult = self._get_param("wtl_penalty_multiplier")
        wtl_threshold = WTL_THRESHOLD  # Relative: independent of instrument scale
        # Self-calibrating baseline: median MFE observed on this instrument.
        # Starts at BASELINE_MFE_SEED and updates each trade via update_baselines().
        baseline_mfe = max(self._get_param("mfe_p50_baseline", BASELINE_MFE_SEED), 1.0)

        if not was_wtl or mfe < wtl_threshold:
            return 0.0

        # Normalize MFE by baseline
        mfe_normalized = SafeMath.safe_div(mfe, baseline_mfe, 0.0)

        # Calculate how much profit was given back
        giveback_ratio = (mfe - exit_pnl) / mfe

        # Time penalty: longer hold after MFE = worse
        time_penalty = 1.0 + (bars_from_mfe_to_exit / 10.0)

        # Final penalty (negative reward)
        penalty = -mfe_normalized * giveback_ratio * wtl_penalty_mult * time_penalty

        # Track statistics
        self.component_stats["wtl"]["sum"] += penalty
        self.component_stats["wtl"]["count"] += 1

        return penalty

    def calculate_opportunity_cost(self, potential_mfe: float, signal_strength: float = 1.0) -> float:
        """
        Penalty for missed opportunities (didn't enter when signal was strong).

        Formula from handbook:
        if potential_mfe > threshold AND signal_strength > 0.5:
            opportunity_normalized = potential_mfe / baseline_mfe
            r_opportunity = -opportunity_normalized * signal_strength * weight * 0.3

        Args:
            potential_mfe: Estimated profit if had entered (from backtesting)
            signal_strength: Confidence of entry signal (0 to 1)

        Returns:
            Opportunity cost penalty (negative reward)
        """
        # Get adaptive parameters
        opportunity_mult = self._get_param("opportunity_multiplier")
        # Self-calibrating threshold: p75 of MFE seen on this instrument.
        # Ensures "missed opportunity" only fires when the move is large
        # relative to what this instrument typically produces.
        opportunity_threshold = self._get_param("opportunity_p75_baseline", OPPORTUNITY_THRESHOLD_SEED)
        baseline_mfe = max(self._get_param("mfe_p50_baseline", BASELINE_MFE_SEED), 1.0)

        if potential_mfe < opportunity_threshold or signal_strength < OPPORTUNITY_SIGNAL_MIN:
            return 0.0

        # Normalize opportunity by baseline
        opportunity_normalized = SafeMath.safe_div(potential_mfe, baseline_mfe, 0.0)

        # Penalty scaled by signal strength and weight
        penalty = -opportunity_normalized * signal_strength * opportunity_mult * OPPORTUNITY_SCALE

        # Track statistics
        self.component_stats["opportunity"]["sum"] += penalty
        self.component_stats["opportunity"]["count"] += 1

        return penalty

    def calculate_total_reward(self, trade_data: dict) -> dict[str, float]:
        """
        Calculate total reward from all components.

        Args:
            trade_data: Dictionary with keys:
                - exit_pnl: Final P&L
                - mfe: Maximum favorable excursion
                - mae: Maximum adverse excursion
                - winner_to_loser: WTL flag
                - bars_from_mfe: Time from MFE to exit (optional)
                - potential_mfe: Missed opportunity (optional)
                - signal_strength: Entry signal confidence (optional)
                - ensemble_bonus: Exploration bonus from disagreement (optional, NEW)

        Returns:
            Dictionary with component rewards and total (6 components)
        """
        # Extract trade data
        exit_pnl = trade_data.get("exit_pnl", 0.0)
        mfe = trade_data.get("mfe", 0.0)
        was_wtl = trade_data.get("winner_to_loser", False)
        bars_from_mfe = trade_data.get("bars_from_mfe", 0)
        potential_mfe = trade_data.get("potential_mfe", 0.0)
        signal_strength = trade_data.get("signal_strength", 1.0)

        # NEW: Counterfactual analysis (optimal vs actual exit)
        entry_price = trade_data.get("entry_price", 0.0)
        exit_price = trade_data.get("exit_price", 0.0)
        direction = trade_data.get("direction", 1)
        mfe_bar_offset = trade_data.get("mfe_bar_offset", 0)

        # Calculate components
        r_capture = self.calculate_capture_efficiency_reward(exit_pnl, mfe)
        r_wtl = self.calculate_wtl_penalty(was_wtl, mfe, exit_pnl, bars_from_mfe)
        r_opportunity = self.calculate_opportunity_cost(potential_mfe, signal_strength)

        # NEW: Activity bonus (exploration when stagnant)
        r_activity = self.activity_monitor.get_exploration_bonus()
        if r_activity > 0:
            self.component_stats["activity"]["sum"] += r_activity
            self.component_stats["activity"]["count"] += 1

        # NEW: Counterfactual reward (penalty for early exits)
        r_counterfactual = 0.0
        if entry_price > 0 and exit_price > 0 and mfe > 0:
            r_counterfactual, _ = self.counterfactual.analyze_exit(
                entry_price, exit_price, mfe, mfe_bar_offset, direction
            )
            self.component_stats["counterfactual"]["sum"] += r_counterfactual
            self.component_stats["counterfactual"]["count"] += 1

        # NEW: Ensemble disagreement bonus (epistemic uncertainty reward)
        r_ensemble = trade_data.get("ensemble_bonus", 0.0)
        if r_ensemble > 0:
            self.component_stats["ensemble"]["sum"] += r_ensemble
            self.component_stats["ensemble"]["count"] += 1

        # Weighted total (6 components with adaptive weights)
        weight_capture = self._get_weight("reward_weight_capture", WEIGHT_CAPTURE)
        weight_wtl = self._get_weight("reward_weight_wtl", WEIGHT_WTL)
        weight_opportunity = self._get_weight("reward_weight_opportunity", WEIGHT_OPPORTUNITY)
        weight_activity = self._get_weight("reward_weight_activity", WEIGHT_ACTIVITY)
        weight_counterfactual = self._get_weight("reward_weight_counterfactual", WEIGHT_COUNTERFACTUAL)
        weight_ensemble = self._get_weight("reward_weight_ensemble", WEIGHT_ENSEMBLE)

        total_reward = (
            weight_capture * r_capture
            + weight_wtl * r_wtl
            + weight_opportunity * r_opportunity
            + weight_activity * r_activity
            + weight_counterfactual * r_counterfactual
            + weight_ensemble * r_ensemble
        )

        self.total_rewards_calculated += 1

        # After each closed trade, soft-update the MFE baselines so they
        # stay calibrated to this instrument's typical move size.
        if mfe > 0:
            self.update_baselines(mfe)

        return {
            "capture_efficiency": r_capture,
            "wtl_penalty": r_wtl,
            "opportunity_cost": r_opportunity,
            "activity_bonus": r_activity,
            "counterfactual_adjustment": r_counterfactual,
            "ensemble_bonus": r_ensemble,  # NEW: 6th component
            "total_reward": total_reward,
            "components_active": sum(
                [
                    1 if r_capture != 0 else 0,
                    1 if r_wtl != 0 else 0,
                    1 if r_opportunity != 0 else 0,
                    1 if r_activity != 0 else 0,
                    1 if r_counterfactual != 0 else 0,
                    1 if r_ensemble != 0 else 0,
                ]
            ),
        }

    def update_baselines(self, mfe: float):
        """
        Soft-update the per-instrument MFE baselines after each trade.

        Uses an exponential moving average with alpha=0.05 (slow decay so the
        baseline tracks the instrument's typical move scale without overreacting
        to individual outliers).

        Uses `set_value()` (direct assignment) rather than the gradient/momentum
        `update()` path because the tanh sigmoid in AdaptiveParam is designed for
        gradient descent and produces distorted results when used to write absolute
        EMA values with wide bounds like [0.01, 100000].

        This is the mechanism that makes the reward shaper instrument-agnostic:
        after ~20 trades on a new instrument the baselines will have converged
        to the right scale and no manual tuning is required.

        Args:
            mfe: Actual Maximum Favorable Excursion for the closed trade
        """
        if mfe <= 0:
            return

        alpha = 0.05  # EMA smoothing — slow enough to avoid single-trade whiplash

        # p50 baseline (median proxy via EMA)
        current_p50 = self._get_param("mfe_p50_baseline", BASELINE_MFE_SEED)
        new_p50 = (1 - alpha) * current_p50 + alpha * mfe
        self.param_manager.set_value(
            self.symbol, "mfe_p50_baseline", new_p50,
            timeframe=self.timeframe, broker=self.broker
        )

        # p75 baseline (high-end proxy: EMA with upward bias on large moves)
        # Large MFEs pull the baseline up faster than small ones pull it down,
        # giving an approximate upper-quartile tracker.
        current_p75 = self._get_param("opportunity_p75_baseline", OPPORTUNITY_THRESHOLD_SEED)
        p75_alpha = 0.10 if mfe > current_p75 else 0.03
        new_p75 = (1 - p75_alpha) * current_p75 + p75_alpha * mfe
        self.param_manager.set_value(
            self.symbol, "opportunity_p75_baseline", new_p75,
            timeframe=self.timeframe, broker=self.broker
        )

    def adapt_weights(self, performance_delta: float):
        """
        Adjust reward component weights based on trade outcome feedback.

        Uses component-outcome correlation: if a component's recent average
        reward correlates with positive trade outcomes (performance_delta > 0),
        that component's weight is nudged up.  Conversely, components whose
        signals correlate with losses get nudged down.

        Weights are bounded to [0.2, 2.0] via _get_weight() so no component
        can be eliminated or dominate entirely.

        Args:
            performance_delta: Trade outcome proxy (positive = profitable trade,
                negative = losing trade). Typically exit_pnl or capture_ratio.
        """
        if not self.param_manager:
            return

        alpha = 0.02  # Very slow adaptation to prevent whiplash
        direction = 1.0 if performance_delta > 0 else -1.0

        for param_name, default in self._weight_names:
            component_key = param_name.replace("reward_weight_", "")
            stats = self.component_stats.get(component_key, {})
            count = stats.get("count", 0)
            if count == 0:
                continue

            # Component's recent average: positive avg on a winning trade
            # means this component correctly identified a good trade.
            avg = stats["sum"] / count
            # Nudge: strengthen components aligned with outcome,
            # weaken those anti-aligned.
            gradient = alpha * direction * (1.0 if avg * direction > 0 else -0.5)

            current = self._get_weight(param_name, default)
            new_val = max(0.2, min(2.0, current + gradient))
            self.param_manager.set_value(
                self.symbol, param_name, new_val,
                timeframe=self.timeframe, broker=self.broker
            )

    def get_statistics(self) -> dict:
        """Return statistics about reward components."""
        stats = {
            "total_rewards_calculated": self.total_rewards_calculated,
            "parameters": {
                "capture_multiplier": self._get_param("capture_multiplier"),
                "wtl_penalty_multiplier": self._get_param("wtl_penalty_multiplier"),
                "opportunity_multiplier": self._get_param("opportunity_multiplier"),
            },
            "weights": {
                name.replace("reward_weight_", ""): self._get_weight(name, default)
                for name, default in self._weight_names
            },
        }

        # Calculate averages
        for component in ["capture", "wtl", "opportunity"]:
            count = self.component_stats[component]["count"]
            if count > 0:
                avg = self.component_stats[component]["sum"] / count
                stats[f"avg_{component}_reward"] = avg
            else:
                stats[f"avg_{component}_reward"] = 0.0

        return stats

    def print_summary(self) -> str:
        """Generate human-readable summary of reward shaper state."""
        stats = self.get_statistics()
        context = f"{self.symbol}_{self.timeframe}_{self.broker}"

        summary = f"""
╔══════════════════════════════════════════════════════════════════╗
    ║               REWARD SHAPER SUMMARY - {context:^20}          ║
╚══════════════════════════════════════════════════════════════════╝

📊 REWARD STATISTICS
   Total Rewards Calculated: {stats['total_rewards_calculated']}

⚙️  ADAPTIVE MULTIPLIERS (from LearnedParametersManager)
   Capture Multiplier:       {stats['parameters']['capture_multiplier']:.2f}
   WTL Penalty Multiplier:   {stats['parameters']['wtl_penalty_multiplier']:.2f}
   Opportunity Multiplier:   {stats['parameters']['opportunity_multiplier']:.2f}

🎚️  COMPONENT WEIGHTS (Adaptive)
   Capture Efficiency:       {stats['weights']['capture']:.1f}
   WTL Penalty:              {stats['weights']['wtl']:.1f}
   Opportunity Cost:         {stats['weights']['opportunity']:.1f}

📈 AVERAGE COMPONENT REWARDS
   Capture Efficiency:       {stats['avg_capture_reward']:+.4f}
   WTL Penalty:              {stats['avg_wtl_reward']:+.4f}
   Opportunity Cost:         {stats['avg_opportunity_reward']:+.4f}
"""
        return summary

    # ========================================================================
    # Phase 3.2: Specialized Dual-Agent Rewards
    # ========================================================================

    def calculate_trigger_reward(
        self,
        actual_mfe: float,
        predicted_runway: float,
        direction: int = 1,       # noqa: ARG002  # NOSONAR
        entry_price: float = 0.0,  # noqa: ARG002  # NOSONAR
    ) -> dict[str, float]:
        """Calculate reward for TriggerAgent (entry specialist).

        Measures runway utilization: How well did we predict MFE?

        Formula:
            runway_utilization = actual_MFE / predicted_runway
            reward = log(runway_utilization) if runway > 0 else large_penalty

        Asymmetric:
            - Underprediction (actual > predicted): Small positive reward
            - Good prediction (actual ≈ predicted): Maximum reward
            - Overprediction (actual < predicted): Larger penalty

        Args:
            actual_mfe: Actual maximum favorable excursion achieved
            predicted_runway: Predicted MFE from TriggerAgent

        Returns:
            Dict with 'runway_reward', 'utilization', 'error_pct'
        """
        if predicted_runway <= 0:
            # Invalid prediction - large penalty
            return {
                "runway_reward": RUNWAY_PENALTY_INVALID,
                "utilization": 0.0,
                "error_pct": 100.0,
                "prediction_quality": "INVALID",
            }

        # Runway utilization ratio
        utilization = actual_mfe / predicted_runway

        # Logarithmic reward (symmetric around 1.0): >0 → log(util); =0 → floor penalty
        # - utilization = 1.0 → 0.0 (perfect); > 1.0 → positive; < 1.0 → negative
        base_reward = math.log(utilization) if utilization > 0 else RUNWAY_LOG_PENALTY

        # Scale reward
        try:
            runway_mult = self._get_param("runway_multiplier")
        except KeyError:
            runway_mult = RUNWAY_MULT_DEFAULT  # Default multiplier for runway prediction
        runway_reward = base_reward * runway_mult

        # Clip extreme values
        runway_reward = max(min(runway_reward, RUNWAY_CLAMP_ABS), -RUNWAY_CLAMP_ABS)

        # Calculate error percentage
        error_pct = abs(actual_mfe - predicted_runway) / predicted_runway * 100

        # Quality assessment
        if RUNWAY_EXCELLENT_MIN <= utilization <= RUNWAY_EXCELLENT_MAX:
            quality = "EXCELLENT"  # Within 20%
        elif RUNWAY_GOOD_MIN <= utilization <= RUNWAY_GOOD_MAX:
            quality = "GOOD"  # Within 50%
        elif utilization < RUNWAY_OVERPREDICT_THRESHOLD:
            quality = "OVERPREDICTED"  # Predicted too high
        else:
            quality = "UNDERPREDICTED"  # Predicted too low

        return {
            "runway_reward": runway_reward,
            "utilization": utilization,
            "error_pct": error_pct,
            "prediction_quality": quality,
            "actual_mfe": actual_mfe,
            "predicted_runway": predicted_runway,
        }

    def calculate_harvester_reward(  # noqa: PLR0913
        self,
        exit_pnl: float,
        mfe: float,
        was_wtl: bool = False,
        bars_held: int = 0,
        bars_from_mfe_to_exit: int = 0,
        mae: float = 0.0,
        exit_time: str = "",
    ) -> dict[str, float]:
        """Calculate reward for HarvesterAgent (exit specialist).

        Combines:
        1. Capture efficiency (how much of MFE captured, magnitude-scaled)
        2. WTL penalty (winner-to-loser prevention, proportional)
        3. Undeveloped-MFE penalty (MAE/MFE ratio — result-based, not bar-based)
        4. Session quality multiplier (London/NY overlap > overnight)

        All components are timeframe-agnostic: no bar counts in reward signal.
        bars_held / bars_from_mfe_to_exit accepted for backward compat but
        not used in reward calculation.

        Args:
            exit_pnl: Final P&L at exit
            mfe: Maximum favorable excursion
            was_wtl: Winner-to-loser flag
            bars_held: (compat) Total bars position was held
            bars_from_mfe_to_exit: (compat) Bars between MFE and exit
            mae: Maximum adverse excursion (absolute, positive)
            exit_time: ISO timestamp of exit (for session quality weighting)

        Returns:
            Dict with component rewards and total
        """
        # 1. Capture efficiency (with magnitude scaling)
        if mfe > 0:
            # Clamp ratio to [-5, 5] — same guard as calculate_capture_efficiency_reward.
            # Prevents explosion when mfe is tiny relative to a large adverse pnl.
            raw_ratio = exit_pnl / mfe
            capture_ratio = max(-5.0, min(5.0, raw_ratio))
            target_capture = TARGET_CAPTURE_RATIO  # Aim for 70% of MFE
            try:
                capture_mult = self._get_param("capture_multiplier")
            except KeyError:
                capture_mult = CAPTURE_MULT_FALLBACK  # Default

            # Magnitude scaling: larger MFE moves relative to baseline get
            # full reward signal; micro-moves that barely cover spread are
            # down-weighted.  Timeframe-agnostic: scales with the instrument.
            try:
                baseline_mfe = max(self._get_param("mfe_p50_baseline", BASELINE_MFE_SEED), 0.01)
            except (KeyError, TypeError):
                baseline_mfe = max(BASELINE_MFE_SEED, 0.01)
            magnitude_scale = min(mfe / baseline_mfe, 2.0)  # Cap at 2x
            magnitude_scale = max(magnitude_scale, 0.3)       # Floor at 0.3 (micro-moves still learn)

            r_capture = max(-3.0, min(3.0, (capture_ratio - target_capture) * capture_mult * magnitude_scale))
        else:
            # No favorable movement — this is a bad entry, penalize
            capture_ratio = 0.0
            r_capture = ZERO_MFE_PENALTY

        # 2. WTL penalty (proportional to profit giveback, not flat)
        try:
            wtl_mult = self._get_param("wtl_multiplier")
        except KeyError:
            wtl_mult = WTL_MULT_DEFAULT  # Default WTL penalty multiplier
        if was_wtl:
            # Proportional penalty: worse giveback = worse penalty
            # If MFE was $10 and exit_pnl is -$2, giveback_ratio = 1.2 (gave back 120% of MFE)
            if mfe > 0:
                giveback_ratio = 1.0 - (exit_pnl / mfe)  # 0 = perfect capture, 2 = lost as much as gained
                giveback_ratio = max(0.5, min(giveback_ratio, 2.0))  # Clamp [0.5, 2.0]
            else:
                giveback_ratio = 1.0
            r_wtl = -wtl_mult * giveback_ratio
        else:
            r_wtl = 0.0

        # 3. Undeveloped-MFE penalty (replaces bar-based timing penalty)
        # Result-based: if MAE is large relative to MFE, the position was
        # held through an adverse move without protecting the gain.  This is
        # timeframe-agnostic — 10 bars overnight or 1 bar on NY open, what
        # matters is the MAE/MFE outcome ratio.
        r_timing = 0.0
        if mfe > 0 and mae > 0:
            # drawdown_ratio: how much adverse move vs favorable move
            drawdown_ratio = min(mae / mfe, 3.0)  # Cap at 3x
            # Only penalize when drawdown is significant relative to MFE
            if drawdown_ratio > 0.3:
                r_timing = UNDEVELOPED_MFE_PENALTY_SCALE * (drawdown_ratio - 0.3)

        # 4. Session quality weighting (optional)
        # MFE captured during London/NY overlap is a stronger signal than
        # the same MFE overnight.  Modulates total reward, not individual
        # components — keeps the gradient direction intact.
        session_mult = self._get_session_quality(exit_time)

        # Total harvester reward
        total_reward = (r_capture + r_wtl + r_timing) * session_mult

        # Quality assessment
        quality = self._harvest_quality(capture_ratio)

        return {
            "harvester_reward": total_reward,
            "capture_efficiency": r_capture,
            "wtl_penalty": r_wtl,
            "timing_penalty": r_timing,
            "capture_ratio": capture_ratio,
            "quality": quality,
            "was_wtl": was_wtl,
            "session_quality": session_mult,
        }

    @staticmethod
    def _harvest_quality(capture_ratio: float) -> str:
        """Classify harvest capture quality from the capture ratio."""
        if capture_ratio >= CAPTURE_QUALITY_EXCELLENT:
            return "EXCELLENT"
        if capture_ratio >= CAPTURE_QUALITY_GOOD:
            return "GOOD"
        if capture_ratio >= CAPTURE_QUALITY_FAIR:
            return "FAIR"
        return "POOR"

    def calculate_dual_agent_rewards(  # noqa: PLR0913
        self,
        # Trigger data
        actual_mfe: float,
        predicted_runway: float,
        direction: int = 1,
        entry_price: float = 0.0,
        # Harvester data
        exit_pnl: float = 0.0,
        mae: float = 0.0,
        was_wtl: bool = False,
        bars_held: int = 0,
        bars_from_mfe_to_exit: int = 0,
        exit_time: str = "",
    ) -> dict[str, float]:
        """
        Calculate rewards for both trigger and harvester agents.

        This is the main reward method for dual-agent mode.

        Returns:
            Dict with:
                - trigger_reward: TriggerAgent reward
                - harvester_reward: HarvesterAgent reward
                - total_reward: Combined reward for overall performance
                - All component breakdowns
        """
        # Calculate individual agent rewards
        trigger_result = self.calculate_trigger_reward(
            actual_mfe, predicted_runway, direction, entry_price
        )
        harvester_result = self.calculate_harvester_reward(
            exit_pnl, actual_mfe, was_wtl, bars_held, bars_from_mfe_to_exit,
            mae=mae, exit_time=exit_time,
        )
        # Trigger: 40% weight (entry quality)
        # Harvester: 60% weight (exit execution is harder)
        total_reward = 0.4 * trigger_result["runway_reward"] + 0.6 * harvester_result["harvester_reward"]

        return {
            "total_reward": total_reward,
            "trigger_reward": trigger_result["runway_reward"],
            "harvester_reward": harvester_result["harvester_reward"],
            "trigger_breakdown": trigger_result,
            "harvester_breakdown": harvester_result,
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing RewardShaper module...")

    shaper = RewardShaper(symbol="BTCUSD", timeframe="M15")

    # Test 1: Good capture efficiency
    print("\n=== Test 1: Good Capture (exit_pnl=80, MFE=100) ===")
    reward1 = shaper.calculate_total_reward({"exit_pnl": 80.0, "mfe": 100.0, "mae": 20.0, "winner_to_loser": False})
    print(f"Capture Efficiency: {reward1['capture_efficiency']:+.4f}")
    print(f"Total Reward: {reward1['total_reward']:+.4f}")

    # Test 2: Winner-to-Loser scenario
    print("\n=== Test 2: Winner-to-Loser (MFE=150, exit_pnl=-30) ===")
    reward2 = shaper.calculate_total_reward(
        {"exit_pnl": -30.0, "mfe": 150.0, "mae": 50.0, "winner_to_loser": True, "bars_from_mfe": 20}
    )
    print(f"Capture Efficiency: {reward2['capture_efficiency']:+.4f}")
    print(f"WTL Penalty: {reward2['wtl_penalty']:+.4f}")
    print(f"Total Reward: {reward2['total_reward']:+.4f}")

    # Test 3: Missed opportunity
    print("\n=== Test 3: Missed Opportunity (potential_mfe=200, signal=0.8) ===")
    reward3 = shaper.calculate_total_reward(
        {
            "exit_pnl": 0.0,
            "mfe": 0.0,
            "mae": 0.0,
            "winner_to_loser": False,
            "potential_mfe": 200.0,
            "signal_strength": 0.8,
        }
    )
    print(f"Opportunity Cost: {reward3['opportunity_cost']:+.4f}")
    print(f"Total Reward: {reward3['total_reward']:+.4f}")

    # Show summary
    print(shaper.print_summary())

    # ===== Phase 3.2: Dual-Agent Reward Tests =====
    print("\n" + "=" * 70)
    print("Phase 3.2: Dual-Agent Reward Tests")
    print("=" * 70)

    # Test 4: TriggerAgent - Perfect prediction
    print("\n=== Test 4: TriggerAgent - Perfect Prediction ===")
    trigger_result = shaper.calculate_trigger_reward(
        actual_mfe=0.0025,  # 25 pips achieved
        predicted_runway=0.0025,  # 25 pips predicted
    )
    print(f"Runway Reward: {trigger_result['runway_reward']:+.4f}")
    print(f"Utilization: {trigger_result['utilization']:.2f}x")
    print(f"Error: {trigger_result['error_pct']:.1f}%")
    print(f"Quality: {trigger_result['prediction_quality']}")

    # Test 5: TriggerAgent - Exceeded prediction
    print("\n=== Test 5: TriggerAgent - Exceeded Prediction ===")
    trigger_result2 = shaper.calculate_trigger_reward(
        actual_mfe=0.0040,  # 40 pips achieved
        predicted_runway=0.0025,  # 25 pips predicted (underpredicted)
    )
    print(f"Runway Reward: {trigger_result2['runway_reward']:+.4f}")
    print(f"Utilization: {trigger_result2['utilization']:.2f}x")
    print(f"Error: {trigger_result2['error_pct']:.1f}%")
    print(f"Quality: {trigger_result2['prediction_quality']}")

    # Test 6: TriggerAgent - Fell short
    print("\n=== Test 6: TriggerAgent - Fell Short ===")
    trigger_result3 = shaper.calculate_trigger_reward(
        actual_mfe=0.0010,  # 10 pips achieved
        predicted_runway=0.0025,  # 25 pips predicted (overpredicted)
    )
    print(f"Runway Reward: {trigger_result3['runway_reward']:+.4f}")
    print(f"Utilization: {trigger_result3['utilization']:.2f}x")
    print(f"Error: {trigger_result3['error_pct']:.1f}%")
    print(f"Quality: {trigger_result3['prediction_quality']}")

    # Test 7: HarvesterAgent - Excellent capture
    print("\n=== Test 7: HarvesterAgent - Excellent Capture (85%) ===")
    harvester_result = shaper.calculate_harvester_reward(
        exit_pnl=0.0034,  # 34 pips captured
        mfe=0.0040,  # 40 pips MFE
        was_wtl=False,
        bars_held=15,
        bars_from_mfe_to_exit=3,
    )
    print(f"Harvester Reward: {harvester_result['harvester_reward']:+.4f}")
    print(f"Capture Ratio: {harvester_result['capture_ratio']:.1%}")
    print(f"Quality: {harvester_result['quality']}")
    print(
        f"Components: capture={harvester_result['capture_efficiency']:+.4f}, "
        f"wtl={harvester_result['wtl_penalty']:+.4f}, timing={harvester_result['timing_penalty']:+.4f}"
    )

    # Test 8: HarvesterAgent - WTL scenario
    print("\n=== Test 8: HarvesterAgent - Winner-to-Loser ===")
    harvester_result2 = shaper.calculate_harvester_reward(
        exit_pnl=-0.0010,  # -10 pips (loss)
        mfe=0.0040,  # Had 40 pips profit
        was_wtl=True,
        bars_held=30,
        bars_from_mfe_to_exit=25,  # Waited 25 bars after MFE
    )
    print(f"Harvester Reward: {harvester_result2['harvester_reward']:+.4f}")
    print(f"Capture Ratio: {harvester_result2['capture_ratio']:.1%}")
    print(f"Quality: {harvester_result2['quality']}")
    print(f"WTL Penalty: {harvester_result2['wtl_penalty']:+.4f} (severe)")

    # Test 9: Full dual-agent rewards
    print("\n=== Test 9: Full Dual-Agent Trade ===")
    dual_result = shaper.calculate_dual_agent_rewards(
        # Trigger: predicted 25 pips, got 30 pips
        actual_mfe=0.0030,
        predicted_runway=0.0025,
        # Harvester: captured 75% of MFE
        exit_pnl=0.0022,
        was_wtl=False,
        bars_held=20,
        bars_from_mfe_to_exit=5,
    )
    print(f"Total Reward: {dual_result['total_reward']:+.4f}")
    print(f"  Trigger Reward (40%): {dual_result['trigger_reward']:+.4f}")
    print(f"  Harvester Reward (60%): {dual_result['harvester_reward']:+.4f}")
    print(f"Trigger Quality: {dual_result['trigger_breakdown']['prediction_quality']}")
    print(f"Harvester Quality: {dual_result['harvester_breakdown']['quality']}")

    print("\n" + "=" * 70)
    print("✅ All dual-agent reward tests complete!")
    print("=" * 70)
