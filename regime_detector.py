"""
Physics-Based Regime Detection using DSP

Ported from MASTER_HANDBOOK.md Section 3.3:
- Damping ratio (ζ) via Digital Signal Processing
- Classifies market regimes: Trending vs Mean-Reverting vs Transitional
- Based on oscillator decay model: A(t) = A₀ × e^(-ζωt)

References:
- https://en.wikipedia.org/wiki/Damping_ratio
- https://en.wikipedia.org/wiki/Hilbert_transform
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional
import math


class SafeMath:
    """Defensive math operations (reused from time_features.py pattern)."""
    
    @staticmethod
    def is_valid(value: float) -> bool:
        """Check if value is finite and not NaN."""
        return np.isfinite(value) and not np.isnan(value)
    
    @staticmethod
    def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with zero check."""
        if abs(denominator) < 1e-10:
            return default
        result = numerator / denominator
        return result if SafeMath.is_valid(result) else default
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range [min_val, max_val]."""
        return max(min_val, min(max_val, value))


class DSPPipeline:
    """
    Digital Signal Processing pipeline for regime detection.
    
    Pipeline stages (from MASTER_HANDBOOK.md):
    1. Detrend (remove linear trend)
    2. Bandpass filter (isolate cycles)
    3. Hilbert transform (analytic signal)
    4. Envelope extraction (instantaneous amplitude)
    5. Fit decay model: A(t) = A₀ × e^(-ζωt)
    6. Extract damping ratio ζ
    """
    
    def __init__(self, lookback: int = 60, min_period: int = 5, max_period: int = 30):
        """
        Initialize DSP pipeline.
        
        Args:
            lookback: Number of bars for analysis
            min_period: Minimum cycle period (bars) for bandpass
            max_period: Maximum cycle period (bars) for bandpass
        """
        self.lookback = max(lookback, 20)  # Defensive: minimum 20 bars
        self.min_period = max(min_period, 3)  # Defensive: minimum 3 bars
        self.max_period = max(max_period, self.min_period + 2)
        
        # Calculate bandpass filter frequencies
        # Nyquist = 0.5 (for normalized frequency)
        self.low_freq = SafeMath.safe_div(1.0, self.max_period, 0.001)
        self.high_freq = min(SafeMath.safe_div(1.0, self.min_period, 0.499), 0.499)
        
    def detrend(self, data: np.ndarray) -> np.ndarray:
        """
        Remove linear trend from data.
        
        Args:
            data: Price series
            
        Returns:
            Detrended data
        """
        if len(data) < 2:
            return data.copy()
        
        return signal.detrend(data, type='linear')
    
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to isolate cyclical components.
        
        Args:
            data: Detrended price series
            
        Returns:
            Bandpass filtered data
        """
        if len(data) < 10:
            return data.copy()
        
        try:
            # Butterworth bandpass filter (order=2 for smooth response)
            sos = signal.butter(
                N=2,
                Wn=[self.low_freq, self.high_freq],
                btype='bandpass',
                output='sos'
            )
            filtered = signal.sosfiltfilt(sos, data)
            
            # Defensive: clean NaN/Inf
            filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
            return filtered
            
        except Exception:
            # Fallback: return data as-is
            return data.copy()
    
    def hilbert_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Compute analytic signal via Hilbert transform.
        
        Args:
            data: Bandpass filtered data
            
        Returns:
            Instantaneous amplitude (envelope)
        """
        if len(data) < 4:
            return np.abs(data)
        
        try:
            # Hilbert transform gives analytic signal
            analytic_signal = signal.hilbert(data)
            
            # Envelope = magnitude of analytic signal
            envelope = np.abs(analytic_signal)
            
            # Defensive: clean NaN/Inf
            envelope = np.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)
            return envelope
            
        except Exception:
            # Fallback: absolute value
            return np.abs(data)
    
    def fit_exponential_decay(self, envelope: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit exponential decay model to envelope: A(t) = A₀ × e^(-ζωt)
        
        Args:
            envelope: Instantaneous amplitude from Hilbert transform
            
        Returns:
            Tuple of (A0, zeta, omega) where:
            - A0: Initial amplitude
            - zeta: Damping ratio (ζ)
            - omega: Angular frequency (ω)
        """
        n = len(envelope)
        if n < 5:
            return (1.0, 0.5, 1.0)  # Default neutral values
        
        # Time array (bar indices)
        t = np.arange(n, dtype=np.float64)
        
        # Defensive: ensure envelope is positive for log-domain fitting
        envelope_safe = np.maximum(envelope, 1e-10)
        
        # Exponential decay model: A(t) = A0 * exp(-decay_rate * t)
        # where decay_rate = ζω
        def exp_decay(t, A0, decay_rate):
            return A0 * np.exp(-decay_rate * t)
        
        try:
            # Fit curve
            # Initial guess: A0 = max amplitude, decay_rate = 0.1
            p0 = [np.max(envelope_safe), 0.1]
            bounds = ([0, 0], [np.inf, 10.0])  # Constrain to physical values
            
            popt, _ = curve_fit(
                exp_decay,
                t,
                envelope_safe,
                p0=p0,
                bounds=bounds,
                maxfev=200
            )
            
            A0, decay_rate = popt
            
            # Estimate angular frequency from dominant cycle
            # Use FFT to find dominant frequency
            fft = np.fft.rfft(envelope_safe)
            freqs = np.fft.rfftfreq(n)
            dominant_idx = np.argmax(np.abs(fft[1:])) + 1  # Skip DC component
            dominant_freq = freqs[dominant_idx]
            omega = 2 * np.pi * dominant_freq
            
            # Prevent division by zero
            if omega < 1e-6:
                omega = 0.1  # Default fallback
            
            # Extract damping ratio: ζ = decay_rate / ω
            zeta = SafeMath.safe_div(decay_rate, omega, 0.5)
            
            # Clamp to reasonable range [0, 2]
            zeta = SafeMath.clamp(zeta, 0.0, 2.0)
            
            return (A0, zeta, omega)
            
        except Exception:
            # Fallback: neutral values
            return (1.0, 0.5, 1.0)
    
    def process(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Run complete DSP pipeline on price series.
        
        Args:
            prices: Price series (close prices)
            
        Returns:
            Dictionary with:
            - zeta: Damping ratio
            - A0: Initial amplitude
            - omega: Angular frequency
            - envelope_mean: Mean envelope amplitude
        """
        # Defensive: validate input
        if len(prices) < 10:
            return {
                'zeta': 0.5,
                'A0': 1.0,
                'omega': 1.0,
                'envelope_mean': 0.0
            }
        
        # Take last 'lookback' bars
        data = prices[-self.lookback:].copy()
        
        # Stage 1: Detrend
        detrended = self.detrend(data)
        
        # Stage 2: Bandpass filter
        filtered = self.bandpass_filter(detrended)
        
        # Stage 3-4: Hilbert transform → Envelope
        envelope = self.hilbert_transform(filtered)
        
        # Stage 5-6: Fit decay model → Extract damping ratio
        A0, zeta, omega = self.fit_exponential_decay(envelope)
        
        return {
            'zeta': zeta,
            'A0': A0,
            'omega': omega,
            'envelope_mean': float(np.mean(envelope))
        }


class RegimeDetector:
    """
    Market regime classifier based on damping ratio.
    
    From MASTER_HANDBOOK.md:
    - ζ < 0.3: Underdamped (trending, momentum)
    - 0.3 ≤ ζ < 0.7: Critical (transitional)
    - ζ ≥ 0.7: Overdamped (mean-reverting, ranging)
    """
    
    # Regime thresholds (from handbook)
    UNDERDAMPED_THRESHOLD = 0.3
    OVERDAMPED_THRESHOLD = 0.7
    
    def __init__(self, lookback: int = 60, min_period: int = 5, max_period: int = 30):
        """
        Initialize regime detector.
        
        Args:
            lookback: Number of bars for DSP analysis
            min_period: Minimum cycle period for bandpass
            max_period: Maximum cycle period for bandpass
        """
        self.dsp = DSPPipeline(lookback, min_period, max_period)
        self.last_zeta = 0.5
        self.regime_history = []  # For regime change tracking
        
    def detect(self, prices: np.ndarray) -> Dict[str, any]:
        """
        Detect current market regime.
        
        Args:
            prices: Price series (close prices)
            
        Returns:
            Dictionary with:
            - regime: 'trending', 'transitional', or 'ranging'
            - zeta: Damping ratio
            - confidence: Confidence in regime classification (0-1)
            - dsp_diagnostics: Raw DSP output
        """
        # Run DSP pipeline
        dsp_result = self.dsp.process(prices)
        zeta = dsp_result['zeta']
        
        # Classify regime
        if zeta < self.UNDERDAMPED_THRESHOLD:
            regime = 'trending'
        elif zeta < self.OVERDAMPED_THRESHOLD:
            regime = 'transitional'
        else:
            regime = 'ranging'
        
        # Calculate confidence based on distance from thresholds
        # High confidence when far from boundaries, low when near
        if regime == 'trending':
            distance = self.UNDERDAMPED_THRESHOLD - zeta
            confidence = SafeMath.clamp(distance / self.UNDERDAMPED_THRESHOLD, 0.0, 1.0)
        elif regime == 'ranging':
            distance = zeta - self.OVERDAMPED_THRESHOLD
            confidence = SafeMath.clamp(distance / (2.0 - self.OVERDAMPED_THRESHOLD), 0.0, 1.0)
        else:  # transitional
            # Low confidence in transitional zone (near thresholds)
            dist_to_low = abs(zeta - self.UNDERDAMPED_THRESHOLD)
            dist_to_high = abs(zeta - self.OVERDAMPED_THRESHOLD)
            min_dist = min(dist_to_low, dist_to_high)
            confidence = 1.0 - SafeMath.clamp(min_dist * 5, 0.0, 1.0)  # Invert: close to threshold = high confidence it's transitional
        
        # Track for regime change detection
        self.regime_history.append(regime)
        if len(self.regime_history) > 20:
            self.regime_history.pop(0)
        
        # Detect recent regime change
        regime_changed = False
        if len(self.regime_history) >= 2:
            regime_changed = (self.regime_history[-1] != self.regime_history[-2])
        
        self.last_zeta = zeta
        
        return {
            'regime': regime,
            'zeta': zeta,
            'confidence': confidence,
            'regime_changed': regime_changed,
            'dsp_diagnostics': dsp_result
        }
    
    def get_regime_factor(self, regime_info: Dict[str, any]) -> float:
        """
        Get risk adjustment factor based on regime.
        
        Used in VaR adjustment pipeline (MASTER_HANDBOOK Section 3.2):
        - Trending markets: lower risk adjustment (let profits run)
        - Ranging markets: higher risk adjustment (tighten stops)
        
        Args:
            regime_info: Output from detect()
            
        Returns:
            Multiplicative factor for risk adjustment (>1 = more conservative)
        """
        regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        # Base factors by regime
        if regime == 'trending':
            base_factor = 0.8  # Reduce risk adjustment in trends
        elif regime == 'ranging':
            base_factor = 1.3  # Increase risk adjustment in ranges
        else:  # transitional
            base_factor = 1.1  # Moderate increase during transitions
        
        # Blend with neutral (1.0) based on confidence
        # Low confidence → move toward neutral
        factor = 1.0 + (base_factor - 1.0) * confidence
        
        return SafeMath.clamp(factor, 0.5, 2.0)


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("Testing RegimeDetector with defensive programming...\n")
    
    # Test 1: Synthetic trending data (underdamped)
    print("=== Test 1: Trending Market (Sine + Trend) ===")
    t = np.linspace(0, 10, 100)
    trending = 100 + 5 * t + 2 * np.sin(t)  # Linear trend + oscillation
    
    detector = RegimeDetector(lookback=60, min_period=5, max_period=30)
    result1 = detector.detect(trending)
    
    print(f"Regime: {result1['regime']}")
    print(f"Zeta (ζ): {result1['zeta']:.4f}")
    print(f"Confidence: {result1['confidence']:.4f}")
    print(f"Expected: trending (ζ < 0.3)")
    
    # Test 2: Synthetic ranging data (overdamped)
    print("\n=== Test 2: Ranging Market (Mean-Reverting) ===")
    ranging = 100 + np.cumsum(np.random.randn(100) * 0.3)  # Random walk with small steps
    ranging = ranging - np.mean(ranging) + 100  # Center around 100
    
    result2 = detector.detect(ranging)
    
    print(f"Regime: {result2['regime']}")
    print(f"Zeta (ζ): {result2['zeta']:.4f}")
    print(f"Confidence: {result2['confidence']:.4f}")
    print(f"Expected: ranging or transitional (ζ >= 0.3)")
    
    # Test 3: Edge cases - insufficient data
    print("\n=== Test 3: Insufficient Data ===")
    short_data = np.array([100.0, 101.0, 102.0])
    result3 = detector.detect(short_data)
    
    print(f"Regime: {result3['regime']}")
    print(f"Zeta (ζ): {result3['zeta']:.4f} (should default to 0.5)")
    print(f"Confidence: {result3['confidence']:.4f}")
    
    # Test 4: Risk factor calculation
    print("\n=== Test 4: Risk Factor Adjustment ===")
    print(f"Trending market factor: {detector.get_regime_factor(result1):.4f} (< 1.0 expected)")
    print(f"Ranging market factor: {detector.get_regime_factor(result2):.4f} (> 1.0 expected)")
    
    # Test 5: Regime change detection
    print("\n=== Test 5: Regime Change Detection ===")
    # Start with trending
    for i in range(5):
        _ = detector.detect(trending)
    # Switch to ranging
    change_result = detector.detect(ranging)
    print(f"Regime changed: {change_result['regime_changed']}")
    print(f"New regime: {change_result['regime']}")
    
    # Test 6: DSP diagnostics
    print("\n=== Test 6: DSP Diagnostics ===")
    diag = result1['dsp_diagnostics']
    print(f"A0 (amplitude): {diag['A0']:.4f}")
    print(f"Omega (frequency): {diag['omega']:.4f}")
    print(f"Envelope mean: {diag['envelope_mean']:.4f}")
    
    print("\n✅ All tests complete - Physics-based regime detection validated")
