#!/usr/bin/env python3
"""
Dual Policy - Orchestrates Trigger and Harvester Agents (Phase 3)
=================================================================
Coordinates entry and exit specialists for dual-agent trading.

Architecture:
- TriggerAgent: Entry specialist (when to enter, which direction)
- HarvesterAgent: Exit specialist (when to close position)

From MASTER_HANDBOOK.md Section 2.2: Dual-Agent Architecture
"""

import logging
from typing import Tuple
from collections import deque
import numpy as np

from trigger_agent import TriggerAgent
from harvester_agent import HarvesterAgent
from regime_detector import RegimeDetector  # Phase 3.4
from path_geometry import PathGeometry  # Phase 3.5: Entry trigger features

LOG = logging.getLogger(__name__)


class DualPolicy:
    """
    Orchestrates TriggerAgent and HarvesterAgent for specialized trading.
    
    Workflow:
    1. On bar close (flat): trigger.decide_entry() → LONG/SHORT/NONE
    2. On bar close (in position): harvester.decide_exit() → HOLD/CLOSE
    3. Track position state (MFE, MAE, bars_held) for harvester
    
    Backward Compatibility:
    - If DDQN_DUAL_AGENT=0: Falls back to single Policy
    - If DDQN_DUAL_AGENT=1: Uses dual-agent architecture
    """
    
    def __init__(self, window: int = 64, enable_regime_detection: bool = True, path_geometry=None, enable_training: bool = False, enable_event_features: bool = True):
        """
        Initialize DualPolicy with trigger and harvester agents.
        
        Args:
            window: Lookback window for state
            enable_regime_detection: Enable Phase 3.4 regime detection (default True)
            path_geometry: PathGeometry instance for entry features (optional)
            enable_training: Enable online learning with PER buffer (default False)
            enable_event_features: Enable Phase 3 event-relative time features (default True)
        """
        self.window = window
        self.enable_training = enable_training
        self.enable_event_features = enable_event_features
        
        # Calculate feature dimension: 7 base + 5 geometry (if enabled) + 6 event (if enabled)
        n_features = 7
        if path_geometry:
            n_features += 5
        if enable_event_features:
            n_features += 6
            
        self.trigger = TriggerAgent(window=window, n_features=n_features, enable_training=enable_training)
        self.harvester = HarvesterAgent(window=window, n_features=10, enable_training=enable_training)
        
        LOG.info("[DUAL_POLICY] TriggerAgent: %d features (7 base + %d geometry + %d event)", 
                 n_features, 5 if path_geometry else 0, 6 if enable_event_features else 0)
        LOG.info("[DUAL_POLICY] Online learning: %s", "ENABLED" if enable_training else "DISABLED")
        
        # Path geometry for entry features
        self.path_geometry = path_geometry
        
        # Phase 3.4: Regime detection
        self.enable_regime_detection = enable_regime_detection
        if self.enable_regime_detection:
            self.regime_detector = RegimeDetector(window_size=50, update_interval=5)
            LOG.info("[DUAL_POLICY] Phase 3.4 Regime Detection ENABLED")
        else:
            self.regime_detector = None
            LOG.info("[DUAL_POLICY] Regime Detection DISABLED")
        
        # Position tracking for harvester
        self.current_position = 0  # -1=SHORT, 0=FLAT, +1=LONG
        self.entry_price = 0.0
        self.entry_bar_time = None
        self.mfe = 0.0  # Maximum favorable excursion
        self.mae = 0.0  # Maximum adverse excursion
        self.bars_held = 0
        self.predicted_runway = 0.0  # From trigger agent
        
        # Phase 3.4: Regime state
        self.current_regime = "UNKNOWN"
        self.current_zeta = 1.0
        
        LOG.info("[DUAL_POLICY] Initialized with TriggerAgent + HarvesterAgent")
    
    def decide_entry(
        self,
        bars: deque,
        imbalance: float = 0.0,
        vpin_z: float = 0.0,
        depth_ratio: float = 1.0,
        realized_vol: float = 0.005,  # For economics calculations
        event_features: dict = None  # Phase 3: Event-relative time features
    ) -> Tuple[int, float, float]:
        """
        Decide entry action using TriggerAgent.
        
        Args:
            bars: Deque of (t, o, h, l, c) tuples (closed bars)
            imbalance: Order book imbalance [-1, 1]
            vpin_z: VPIN z-score
            depth_ratio: Depth ratio
            realized_vol: Rogers-Satchell volatility for economics calculations
            event_features: Dict of event-relative time features (30+ features)
        
        Returns:
            (action, confidence, predicted_runway)
            - action: 0=NO_ENTRY, 1=LONG, 2=SHORT
            - confidence: [0, 1] Platt-calibrated probability
            - predicted_runway: Expected MFE
        """
        # Phase 3.4: Update regime detection with latest price
        if self.regime_detector and len(bars) > 0:
            latest_bar = bars[-1]
            _, _, _, _, close_price = latest_bar
            self.current_regime, self.current_zeta = self.regime_detector.add_price(close_price)
        
        # Build state (includes path geometry and event features if available)
        state = self._build_state(bars, imbalance, vpin_z, depth_ratio, realized_vol, event_features)
        
        # Phase 3.4: Get regime threshold adjustment for trigger
        regime_threshold_adj = 0.0
        if self.regime_detector:
            regime_threshold_adj = self.regime_detector.get_trigger_threshold_adjustment()
        
        # Phase 2: Get path geometry feasibility
        feasibility = 1.0
        if self.path_geometry:
            feasibility = self.path_geometry.last.get('feasibility', 1.0)
        
        # Phase 2: Calculate economics parameters
        # Expected gain/loss based on realized volatility and typical move sizes
        expected_gain = realized_vol * 2.0  # Expect 2σ move on winning trades
        expected_loss = realized_vol * 1.0  # Risk 1σ on losing trades
        friction_cost = realized_vol * 0.1  # Friction ~10% of volatility (spread + slippage)
        
        # Trigger decides entry with all Phase 2 gates
        action, confidence, predicted_runway = self.trigger.decide(
            state, 
            current_position=self.current_position,
            regime_threshold_adj=regime_threshold_adj,  # Phase 3.4
            feasibility=feasibility,  # Phase 2: Hard gate
            expected_gain=expected_gain,  # Phase 2: Economics
            expected_loss=expected_loss,
            friction_cost=friction_cost
        )
        
        # Phase 3.4: Apply regime-aware runway adjustment
        if action in [1, 2] and self.regime_detector:  # LONG or SHORT
            regime_multiplier = self.regime_detector.get_regime_multiplier()
            predicted_runway_adjusted = predicted_runway * regime_multiplier
            
            LOG.info(
                "[DUAL_POLICY] TRIGGER: %s entry, conf=%.2f, runway=%.4f (base=%.4f, regime=%s, mult=%.2fx)",
                "LONG" if action == 1 else "SHORT", confidence, predicted_runway_adjusted,
                predicted_runway, self.current_regime, regime_multiplier
            )
            
            self.predicted_runway = predicted_runway_adjusted
        elif action in [1, 2]:  # No regime detection
            self.predicted_runway = predicted_runway
            LOG.info(
                "[DUAL_POLICY] TRIGGER: %s entry, conf=%.2f, predicted_runway=%.4f",
                "LONG" if action == 1 else "SHORT", confidence, predicted_runway
            )
        
        return action, confidence, predicted_runway
    
    def decide_exit(
        self,
        bars: deque,
        current_price: float,
        imbalance: float = 0.0,
        vpin_z: float = 0.0,
        depth_ratio: float = 1.0
    ) -> Tuple[int, float]:
        """
        Decide exit action using HarvesterAgent.
        
        Args:
            bars: Deque of (t, o, h, l, c) tuples (closed bars)
            current_price: Current close price
            imbalance: Order book imbalance
            vpin_z: VPIN z-score
            depth_ratio: Depth ratio
        
        Returns:
            (action, confidence)
            - action: 0=HOLD, 1=CLOSE
            - confidence: [0, 1]
        """
        # Update MFE/MAE
        self._update_mfe_mae(current_price)
        self.bars_held += 1
        
        # Build market state
        market_state = self._build_state(bars, imbalance, vpin_z, depth_ratio)
        
        # Harvester decides exit
        action, confidence = self.harvester.decide(
            market_state=market_state,
            mfe=self.mfe,
            mae=self.mae,
            bars_held=self.bars_held,
            entry_price=self.entry_price,
            direction=self.current_position
        )
        
        if action == 1:  # CLOSE
            LOG.info(
                "[DUAL_POLICY] HARVESTER: CLOSE signal, conf=%.2f, "
                "MFE=%.4f, MAE=%.4f, bars=%d",
                confidence, self.mfe, self.mae, self.bars_held
            )
        
        return action, confidence
    
    def on_entry(self, direction: int, entry_price: float, entry_time):
        """
        Called when position is entered.
        
        Args:
            direction: +1 for LONG, -1 for SHORT
            entry_price: Entry price
            entry_time: Entry bar timestamp
        """
        self.current_position = direction
        self.entry_price = entry_price
        self.entry_bar_time = entry_time
        self.mfe = 0.0
        self.mae = 0.0
        self.bars_held = 0
        LOG.info(
            "[DUAL_POLICY] Position entered: %s @ %.2f",
            "LONG" if direction == 1 else "SHORT", entry_price
        )
    
    def on_exit(self, exit_price: float, capture_ratio: float, was_wtl: bool):
        """
        Called when position is closed.
        
        Args:
            exit_price: Exit price
            capture_ratio: exit_pnl / MFE
            was_wtl: Was this a winner-to-loser trade?
        """
        # Update agents with trade outcome
        self.trigger.update_from_trade(
            actual_mfe=self.mfe,
            predicted_runway=self.predicted_runway
        )
        self.harvester.update_from_trade(
            capture_ratio=capture_ratio,
            was_wtl=was_wtl
        )
        
        LOG.info(
            "[DUAL_POLICY] Position closed @ %.2f, MFE=%.4f, Capture=%.2f%%",
            exit_price, self.mfe, capture_ratio * 100
        )
        
        # Reset position state
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_bar_time = None
        self.mfe = 0.0
        self.mae = 0.0
        self.bars_held = 0
        self.predicted_runway = 0.0
    
    def _update_mfe_mae(self, current_price: float):
        """Update MFE and MAE based on current price."""
        if self.entry_price == 0:
            return
        
        if self.current_position == 1:  # LONG
            profit = current_price - self.entry_price
            self.mfe = max(self.mfe, profit)
            self.mae = max(self.mae, -profit)
        elif self.current_position == -1:  # SHORT
            profit = self.entry_price - current_price
            self.mfe = max(self.mfe, profit)
            self.mae = max(self.mae, -profit)
    
    def _build_state(
        self,
        bars: deque,
        imbalance: float,
        vpin_z: float,
        depth_ratio: float,
        realized_vol: float = 0.005,  # Provide RS volatility for geometry calculation
        event_features: dict = None  # Phase 3: Event-relative time features
    ) -> np.ndarray:
        """
        Build normalized state features.
        
        Features (expandable based on enabled modules):
        Base (7):
            - ret1: 1-bar return
            - ret5: 5-bar return  
            - ma_diff: MA fast/slow difference
            - vol: 20-bar volatility
            - imbalance: Order book imbalance
            - vpin_z: VPIN z-score
            - depth_ratio: Bid+ask depth ratio
        
        Geometry (5) - from handbook:
            - efficiency: Path displacement / path length
            - gamma: Acceleration (2nd derivative)
            - jerk: Rate of change of acceleration (3rd derivative)
            - runway: Inverse volatility pressure
            - feasibility: Composite entry quality score
        
        Event Time (6) - key session features:
            - london_active: London session active [0, 1]
            - ny_active: New York session active [0, 1]
            - tokyo_active: Tokyo session active [0, 1]
            - london_ny_overlap: High liquidity overlap [0, 1]
            - rollover_proximity: Proximity to 22:00 UTC rollover [-1, 1]
            - week_progress: Week progress [0, 1]
        
        Returns:
            State array (window, n_features) with features normalized
        """
        # Calculate expected feature dimension
        n_features = 7  # Base
        if self.path_geometry:
            n_features += 5  # Geometry
        if event_features and self.enable_event_features:
            n_features += 6  # Event time
            
        if len(bars) < 70:
            return np.zeros((self.window, n_features), dtype=np.float32)
        
        closes = [b[4] for b in bars]
        c = np.array(closes, dtype=np.float64)
        
        # Calculate returns
        ret1 = np.zeros_like(c)
        if len(c) >= 2:
            ret1[1:] = np.divide(
                c[1:], c[:-1], out=np.ones_like(c[1:]), where=c[:-1] != 0
            ) - 1.0
        
        ret5 = np.zeros_like(c)
        if len(c) >= 6:
            ret5[5:] = np.divide(
                c[5:], c[:-5], out=np.ones_like(c[5:]), where=c[:-5] != 0
            ) - 1.0
        
        # Moving averages
        def rolling_mean(x, n):
            out = np.full_like(x, np.nan, dtype=np.float64)
            if len(x) >= n:
                cs = np.cumsum(np.insert(x, 0, 0.0))
                out[n - 1:] = (cs[n:] - cs[:-n]) / n
            return out
        
        def rolling_std(x, n):
            out = np.full_like(x, np.nan, dtype=np.float64)
            if len(x) >= n:
                for i in range(n - 1, len(x)):
                    w = x[i - n + 1 : i + 1]
                    out[i] = np.std(w)
            return out
        
        ma_fast = rolling_mean(c, 10)
        ma_slow = rolling_mean(c, 30)
        ma_diff = np.divide(
            ma_fast, ma_slow, out=np.ones_like(ma_fast), where=ma_slow != 0
        ) - 1.0
        vol = rolling_std(ret1, 20)
        
        # Microstructure features (broadcast to window)
        imb = np.full(len(c), imbalance, dtype=np.float64)
        vpz = np.full(len(c), vpin_z, dtype=np.float64)
        dpr = np.full(len(c), depth_ratio, dtype=np.float64)
        
        # Base features (7-dim)
        base_feats = [
            np.nan_to_num(ret1, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(ret5, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(ma_diff, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(imb, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(vpz, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(dpr, nan=1.0, posinf=1.0, neginf=1.0)
        ]
        
        # Add path geometry features if available (5-dim)
        if self.path_geometry:
            # Update geometry with current bars and volatility
            geom = self.path_geometry.update(bars, realized_vol)
            
            # Broadcast geometry features to window length
            eff = np.full(len(c), geom['efficiency'], dtype=np.float64)
            gamma = np.full(len(c), geom['gamma'], dtype=np.float64)
            jerk = np.full(len(c), geom['jerk'], dtype=np.float64)
            runway = np.full(len(c), geom['runway'], dtype=np.float64)
            feasibility = np.full(len(c), geom['feasibility'], dtype=np.float64)
            
            base_feats.extend([
                np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(jerk, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(runway, nan=0.5, posinf=0.5, neginf=0.5),
                np.nan_to_num(feasibility, nan=0.5, posinf=0.5, neginf=0.5)
            ])
        
        # Add event time features if available (6 key features)
        if event_features:
            # Select key temporal features (already normalized in event_time_features.py)
            london_active = event_features.get('london_active', 0.0)
            ny_active = event_features.get('ny_active', 0.0)
            tokyo_active = event_features.get('tokyo_active', 0.0)
            london_ny_overlap = event_features.get('london_ny_overlap', 0.0)
            rollover_proximity = event_features.get('rollover_proximity_norm', 0.0)
            week_progress = event_features.get('week_progress', 0.5)
            
            # Broadcast event features to window length
            base_feats.extend([
                np.full(len(c), london_active, dtype=np.float64),
                np.full(len(c), ny_active, dtype=np.float64),
                np.full(len(c), tokyo_active, dtype=np.float64),
                np.full(len(c), london_ny_overlap, dtype=np.float64),
                np.full(len(c), rollover_proximity, dtype=np.float64),
                np.full(len(c), week_progress, dtype=np.float64)
            ])
        
        # Stack features (7, 12, 13, or 18-dim depending on modules enabled)
        feats = np.vstack(base_feats).T
        
        # Take last window bars
        feats = feats[-self.window:].astype(np.float32)
        
        # Normalize
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-8
        feats = (feats - mu) / sd
        
        return feats

    # -------------------------------------------------------------------------
    # Online Learning Methods
    # -------------------------------------------------------------------------
    def add_trigger_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = True
    ):
        """
        Add experience to TriggerAgent buffer for online learning.
        
        Args:
            state: State at entry decision time (12-dim)
            action: 0=NO_ENTRY, 1=LONG, 2=SHORT
            reward: Shaped reward from RewardShaper
            next_state: State after trade closed
            done: Episode terminal (True for completed trade)
        """
        if not self.enable_training:
            return
        
        self.trigger.add_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
    
    def add_harvester_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = True
    ):
        """
        Add experience to HarvesterAgent buffer for online learning.
        
        Args:
            state: State at exit decision time (10-dim)
            action: 0=HOLD, 1=CLOSE
            reward: Shaped reward from RewardShaper
            next_state: State after action
            done: Episode terminal (True for position closed)
        """
        if not self.enable_training:
            return
        
        self.harvester.add_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
    
    def train_step(self, adaptive_reg=None) -> dict:
        """
        Execute one training step on both agents.
        
        Args:
            adaptive_reg: Optional AdaptiveRegularization instance for L2/dropout adjustment
        
        Returns:
            Dictionary with training metrics from both agents
        """
        if not self.enable_training:
            return {'trigger': None, 'harvester': None}
        
        metrics = {}
        
        # Get current regularization if provided
        if adaptive_reg:
            reg_params = adaptive_reg.get_current()
        else:
            reg_params = None
        
        # Train TriggerAgent
        trigger_metrics = self.trigger.train_step()
        metrics['trigger'] = trigger_metrics
        
        # Train HarvesterAgent
        harvester_metrics = self.harvester.train_step()
        metrics['harvester'] = harvester_metrics
        
        # Log training summary
        if trigger_metrics or harvester_metrics:
            LOG.info(
                "[TRAIN] Trigger: loss=%.4f td=%.4f | Harvester: loss=%.4f td=%.4f",
                trigger_metrics.get('loss', 0.0) if trigger_metrics else 0.0,
                trigger_metrics.get('mean_td_error', 0.0) if trigger_metrics else 0.0,
                harvester_metrics.get('loss', 0.0) if harvester_metrics else 0.0,
                harvester_metrics.get('mean_td_error', 0.0) if harvester_metrics else 0.0
            )
        
        return metrics
    
    def get_training_stats(self) -> dict:
        """Get training statistics from both agents."""
        return {
            'trigger': self.trigger.get_training_stats() if hasattr(self.trigger, 'get_training_stats') else {},
            'harvester': self.harvester.get_training_stats() if hasattr(self.harvester, 'get_training_stats') else {},
            'enable_training': self.enable_training
        }


# ============================================================================
# Self-Test
# ============================================================================
if __name__ == "__main__":
    import datetime as dt
    
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("DualPolicy Self-Test")
    print("=" * 70)
    
    # Test 1: Initialize
    print("\n[TEST 1] Initialize DualPolicy")
    policy = DualPolicy(window=64)
    assert policy.current_position == 0
    assert policy.trigger is not None
    assert policy.harvester is not None
    print("✓ DualPolicy initialized with trigger + harvester")
    
    # Test 2: Entry decision (flat)
    print("\n[TEST 2] Entry decision (flat position)")
    bars = deque(maxlen=100)
    for i in range(100):
        t = dt.datetime.now()
        o = h = l = c = 100000.0 + i * 10
        bars.append((t, o, h, l, c))
    
    action, conf, runway = policy.decide_entry(bars, imbalance=0.1)
    assert action in [0, 1, 2]
    assert 0 <= conf <= 1
    assert runway >= 0
    print(f"✓ Entry decision: action={action}, conf={conf:.2f}, runway={runway:.4f}")
    
    # Test 3: Enter position
    print("\n[TEST 3] Enter LONG position")
    policy.on_entry(direction=1, entry_price=100000.0, entry_time=dt.datetime.now())
    assert policy.current_position == 1
    assert policy.entry_price == 100000.0
    print("✓ Position entered: LONG @ 100000.0")
    
    # Test 4: Exit decision (in position)
    print("\n[TEST 4] Exit decision (in position)")
    current_price = 100050.0  # Small profit
    action, conf = policy.decide_exit(bars, current_price, imbalance=0.1)
    assert action in [0, 1]  # HOLD or CLOSE
    assert 0 <= conf <= 1
    assert policy.mfe > 0  # Should have tracked MFE
    print(f"✓ Exit decision: action={action} ({'CLOSE' if action == 1 else 'HOLD'}), "
          f"conf={conf:.2f}, MFE={policy.mfe:.2f}")
    
    # Test 5: Exit position
    print("\n[TEST 5] Exit position")
    policy.on_exit(exit_price=100050.0, capture_ratio=0.8, was_wtl=False)
    assert policy.current_position == 0
    assert policy.mfe == 0.0
    print("✓ Position closed, state reset")
    
    print("\n" + "=" * 70)
    print("✓ All DualPolicy tests passed!")
    print("=" * 70)
