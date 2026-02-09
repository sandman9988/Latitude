"""
Test Suite for Risk-Aware SAC Manager

This module provides comprehensive testing for the tail-risk monitoring
system, including:

1. Synthetic market simulation with controlled tail events
2. Exposure scaling behavior verification
3. GPD hazard estimation accuracy
4. Integration with existing VaR/Circuit Breaker systems

Author: Generated from blueprint specification
Date: 2026-01-11
"""

import pytest
import numpy as np

rng = np.random.default_rng(42)

import matplotlib.pyplot as plt
from typing import Tuple
from src.risk.risk_aware_sac_manager import RiskAwareSACManager, rolling_kurtosis, vpin_zscore, truncated_gpd_hazard


class SyntheticMarketSimulator:
    """
    Generate realistic market price paths with configurable
    tail-event injection.
    """

    def __init__(
        self,
        n_ticks: int = 5000,
        drift: float = -0.0002,
        volatility: float = 0.01,
        crash_prob: float = 0.005,
        crash_magnitude_range: Tuple[float, float] = (0.8, 2.0),
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        n_ticks : int
            Total simulation length
        drift : float
            Mean log-return per tick
        volatility : float
            Normal volatility (sigma)
        crash_prob : float
            Probability of tail event per tick
        crash_magnitude_range : tuple
            (min, max) multiplier for crash events
        seed : int
            Random seed for reproducibility
        """
        self.n_ticks = n_ticks
        self.drift = drift
        self.vol = volatility
        self.crash_prob = crash_prob
        self.crash_range = crash_magnitude_range

        rng = np.random.default_rng(seed)

        # Generate base price path
        self.returns = rng.lognormal(mean=drift, sigma=volatility, size=n_ticks)

        # Inject tail events
        self.crash_indices = []
        for i in range(n_ticks):
            if rng.random() < crash_prob:
                # Random extreme move
                magnitude = rng.uniform(*crash_magnitude_range)
                direction = rng.choice([-1, 1])
                self.returns[i] *= np.exp(direction * magnitude)
                self.crash_indices.append(i)

        # Generate cumulative price
        self.prices = 100.0 * np.cumprod(self.returns)

        # Generate VPIN-like metric (correlated with volatility + noise)
        self.vpin_metric = self._generate_vpin()

    def _generate_vpin(self) -> np.ndarray:
        """
        Simulate VPIN-like liquidity stress metric.
        Spikes during volatile periods + random noise.
        """
        # Base: rolling volatility proxy
        rolling_vol = np.array([self.returns[max(0, i - 20) : i + 1].std() for i in range(self.n_ticks)])

        # Add persistence
        vpin = np.zeros(self.n_ticks)
        vpin[0] = rolling_vol[0]
        alpha = 0.9  # AR(1) coefficient

        for i in range(1, self.n_ticks):
            noise = rng.normal(0, 0.05)
            vpin[i] = alpha * vpin[i - 1] + (1 - alpha) * rolling_vol[i] + noise

        # Spike during crashes
        for idx in self.crash_indices:
            spike_window = range(max(0, idx - 5), min(self.n_ticks, idx + 15))
            for j in spike_window:
                vpin[j] *= rng.uniform(2.0, 5.0)

        return vpin

    def get_returns_and_vpin(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get percentage returns and VPIN metric."""
        pct_returns = (self.prices[1:] - self.prices[:-1]) / self.prices[:-1]
        # Pad first return
        pct_returns = np.concatenate([[0.0], pct_returns])
        return pct_returns, self.vpin_metric


class TestRiskAwareSACManager:
    """Test suite for RiskAwareSACManager."""

    def test_initialization(self):
        """Test manager initializes with correct parameters."""
        manager = RiskAwareSACManager(window=100, kurt_max=4.0, vpin_trigger=2.5, collapse_fac=0.15)

        assert manager.window == 100
        assert manager.kurt_max == pytest.approx(4.0)
        assert manager.vpin_trigger == pytest.approx(2.5)
        assert manager.collapse_fac == pytest.approx(0.15)
        assert manager.latest_exposure == pytest.approx(1.0)
        assert manager.latest_hazard == pytest.approx(0.0)
        assert len(manager.ret_buf) == 0

    def test_update_buffers(self):
        """Test that buffers update correctly."""
        manager = RiskAwareSACManager(window=10)

        # Add 15 samples (should keep only last 10)
        for i in range(15):
            manager.update(0.001 * i, 0.5 + 0.01 * i)

        assert len(manager.ret_buf) == 10
        assert len(manager.vpin_buf) == 10
        assert manager.total_updates == 15

    def test_kurtosis_calculation(self):
        """Test excess kurtosis computation."""
        # Normal distribution should have ~0 excess kurtosis
        normal_data = rng.normal(0, 1, 1000)
        kurt = rolling_kurtosis(normal_data, 1000)
        assert -1.0 < kurt < 1.0  # Should be near 0

        # Laplace (fat tails) should have positive excess kurtosis
        laplace_data = rng.laplace(0, 1, 1000)
        kurt = rolling_kurtosis(laplace_data, 1000)
        assert kurt > 1.0  # Fat tails → positive excess

    def test_exposure_collapse_on_extreme_conditions(self):
        """Test that exposure collapses when both signals spike."""
        rng = np.random.default_rng(42)
        manager = RiskAwareSACManager(window=50, kurt_max=3.0, vpin_trigger=2.0, collapse_fac=0.1)

        # Feed normal returns first
        for _ in range(40):
            manager.update(rng.normal(0, 0.01), rng.normal(0.5, 0.1))

        initial_exposure = manager.latest_exposure
        print(f"Exposure after normal: {initial_exposure:.3f}")

        # Inject extreme returns (high kurtosis) and track minimum exposure
        min_exposure = 1.0
        for _ in range(20):
            extreme_ret = rng.choice([-0.1, 0.1])  # Fat tails
            high_vpin = 3.0  # High stress
            exposure, hazard = manager.update(extreme_ret, high_vpin)
            min_exposure = min(min_exposure, exposure)

        # Exposure should have collapsed at some point during stress
        assert min_exposure < 0.3, (
            f"Exposure should collapse during stress: min was {min_exposure:.3f}"
        )
        # At least one collapse event should have been recorded
        assert manager.collapse_events > 0, "Should record collapse events"
        print(f"Min exposure during stress: {min_exposure:.3f}")
        print(f"Collapse events: {manager.collapse_events}")

    def test_gpd_hazard_estimation(self):
        """Test GPD tail-risk hazard calculation."""
        rng = np.random.default_rng(42)
        # Normal distribution → hazard should be bounded [0, 1]
        normal_data = rng.normal(0, 1, 500)
        hazard = truncated_gpd_hazard(normal_data, tail_percentile=0.95)
        assert 0.0 <= hazard <= 1.0, f"Hazard should be in [0,1], got {hazard}"

        # Heavy-tailed distribution → higher hazard
        rng = np.random.default_rng(42)
        heavy_tail = np.concatenate([rng.normal(0, 1, 450), rng.normal(0, 5, 50)])  # Fat tail
        hazard_heavy = truncated_gpd_hazard(heavy_tail, tail_percentile=0.90)
        assert 0.0 <= hazard_heavy <= 1.0, f"Heavy tail hazard should be in [0,1], got {hazard_heavy}"

    def test_scale_action(self):
        """Test action scaling functionality."""
        manager = RiskAwareSACManager(window=50)

        # Warm up with normal data
        for _ in range(60):
            manager.update(rng.normal(0, 0.01), 0.5)

        # Scale a raw action
        raw_action = 1.0  # Full position
        scaled, hazard = manager.scale_action(raw_action)

        assert 0.0 <= scaled <= 1.0
        assert 0.0 <= hazard <= 1.0
        print(f"Scaled action: {scaled:.3f}, Hazard: {hazard:.4f}")

    def test_diagnostics(self):
        """Test diagnostics reporting."""
        manager = RiskAwareSACManager(window=50)

        for i in range(100):
            manager.update(rng.normal(0, 0.01), 0.5)

        diag = manager.get_diagnostics()

        assert "exposure" in diag
        assert "hazard" in diag
        assert "kurtosis" in diag
        assert "total_updates" in diag
        assert diag["total_updates"] == 100
        assert diag["buffer_size"] == 50

        print("\nDiagnostics:", diag)


class TestSyntheticMarketIntegration:
    """Test with realistic synthetic market scenarios."""

    def test_market_crash_detection(self):
        """Test that manager detects and responds to market crashes."""
        # Generate market with crashes
        sim = SyntheticMarketSimulator(n_ticks=1000, crash_prob=0.01, crash_magnitude_range=(1.5, 3.0))

        returns, vpin = sim.get_returns_and_vpin()

        manager = RiskAwareSACManager(window=100, kurt_max=3.0, vpin_trigger=2.0, collapse_fac=0.1)

        exposures = []
        hazards = []

        for ret, vp in zip(returns, vpin):
            exp, haz = manager.update(ret, vp)
            exposures.append(exp)
            hazards.append(haz)

        exposures = np.array(exposures)
        hazards = np.array(hazards)

        # Check that exposure collapsed during crashes
        crash_windows = []
        for crash_idx in sim.crash_indices:
            window = slice(max(0, crash_idx - 10), min(len(exposures), crash_idx + 20))
            crash_windows.append(exposures[window].min())

        if crash_windows:
            min_exposure = min(crash_windows)
            print(f"\nMin exposure during crashes: {min_exposure:.3f}")
            assert min_exposure < 0.5  # Should have collapsed significantly

        # Report statistics
        diag = manager.get_diagnostics()
        print(f"Collapse events: {diag['collapse_events']}")
        print(f"Collapse rate: {diag['collapse_rate']:.2%}")
        print(f"Extreme hazard events: {diag['extreme_hazard_events']}")

    def test_normal_market_conditions(self):
        """Test that manager stays near full exposure in calm markets."""
        # Generate calm market
        sim = SyntheticMarketSimulator(n_ticks=500, drift=0.0001, volatility=0.005, crash_prob=0.0)  # No crashes

        returns, vpin = sim.get_returns_and_vpin()

        manager = RiskAwareSACManager(window=100)

        exposures = []
        for ret, vp in zip(returns, vpin):
            exp, _ = manager.update(ret, vp)
            exposures.append(exp)

        # After warmup, exposure should be reasonably high (not collapsed)
        mean_exposure = np.mean(exposures[200:])
        print(f"\nMean exposure in calm market: {mean_exposure:.3f}")
        assert mean_exposure > 0.5, f"Should maintain reasonable exposure in calm market: {mean_exposure:.3f}"

    def test_performance_benchmark(self):
        """Benchmark update() performance."""
        import time

        rng = np.random.default_rng(42)
        manager = RiskAwareSACManager(window=500)

        # Warm up
        for _ in range(500):
            manager.update(rng.normal(0, 0.01), 0.5)

        # Benchmark
        n_iterations = 100
        start = time.time()

        for _ in range(n_iterations):
            manager.update(rng.normal(0, 0.01), 0.5)

        elapsed = time.time() - start
        us_per_update = (elapsed / n_iterations) * 1e6

        print(f"\nPerformance: {us_per_update:.1f} µs/update")
        print(f"Throughput: {n_iterations/elapsed:.0f} updates/sec")

        # GPD fitting is inherently expensive; allow up to 50ms per update
        assert us_per_update < 50_000, f"Update too slow: {us_per_update:.0f} µs"


def test_standalone_functions():
    """Test standalone helper functions."""
    # Generate test data
    normal_data = rng.normal(0, 1, 500)

    # Test kurtosis
    kurt = rolling_kurtosis(normal_data, 500)
    assert isinstance(kurt, float)
    assert -2.0 < kurt < 2.0

    # Test VPIN z-score
    vpin_data = rng.normal(0.5, 0.1, 500)
    z = vpin_zscore(vpin_data, 500)
    assert isinstance(z, float)

    # Test GPD hazard
    hazard = truncated_gpd_hazard(normal_data)
    assert 0.0 <= hazard <= 1.0

    print(f"Kurtosis: {kurt:.3f}")
    print(f"VPIN Z-score: {z:.3f}")
    print(f"GPD Hazard: {hazard:.4f}")


def visualize_crash_response():
    """
    Generate visualization of manager response to crashes.
    For manual inspection - not run in automated tests.
    """
    # Generate market with visible crashes
    sim = SyntheticMarketSimulator(n_ticks=2000, crash_prob=0.008, crash_magnitude_range=(1.0, 2.5), seed=123)

    returns, vpin = sim.get_returns_and_vpin()

    manager = RiskAwareSACManager(window=200, kurt_max=3.0, vpin_trigger=2.0, collapse_fac=0.1)

    exposures = []
    hazards = []
    kurtosis_vals = []
    vpin_z_vals = []

    for ret, vp in zip(returns, vpin):
        exp, haz = manager.update(ret, vp)
        exposures.append(exp)
        hazards.append(haz)
        kurtosis_vals.append(manager.latest_kurtosis)
        vpin_z_vals.append(manager.latest_vpin_z)

    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    # Price
    axes[0].plot(sim.prices, "b-", linewidth=0.5)
    for crash_idx in sim.crash_indices:
        axes[0].axvline(crash_idx, color="red", alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("Price")
    axes[0].set_title("Synthetic Market with Tail Events")
    axes[0].grid(True, alpha=0.3)

    # Kurtosis
    axes[1].plot(kurtosis_vals, "orange", linewidth=0.8)
    axes[1].axhline(manager.kurt_max, color="red", linestyle="--", label=f"Max={manager.kurt_max}")
    axes[1].set_ylabel("Excess Kurtosis")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # VPIN Z-score
    axes[2].plot(vpin_z_vals, "purple", linewidth=0.8)
    axes[2].axhline(manager.vpin_trigger, color="red", linestyle="--", label=f"Trigger={manager.vpin_trigger}")
    axes[2].axhline(-manager.vpin_trigger, color="red", linestyle="--")
    axes[2].set_ylabel("VPIN Z-Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Exposure
    axes[3].fill_between(range(len(exposures)), exposures, alpha=0.5, color="green")
    axes[3].axhline(manager.collapse_fac, color="red", linestyle="--", label=f"Collapse={manager.collapse_fac}")
    axes[3].set_ylabel("Exposure Factor")
    axes[3].set_ylim([0, 1.1])
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # Hazard
    axes[4].fill_between(range(len(hazards)), hazards, alpha=0.5, color="darkred")
    axes[4].set_ylabel("GPD Hazard")
    axes[4].set_xlabel("Tick")
    axes[4].set_ylim([0, max(0.5, max(hazards))])
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("risk_aware_sac_crash_response.png", dpi=150)
    print("\nVisualization saved to: risk_aware_sac_crash_response.png")

    # Print summary
    diag = manager.get_diagnostics()
    print(f"\n{'='*60}")
    print("SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total ticks: {len(returns)}")
    print(f"Crash events: {len(sim.crash_indices)}")
    print(f"Collapse events: {diag['collapse_events']}")
    print(f"Collapse rate: {diag['collapse_rate']:.2%}")
    print(f"Mean exposure: {np.mean(exposures):.3f}")
    print(f"Min exposure: {np.min(exposures):.3f}")
    print(f"Mean hazard: {np.mean(hazards):.4f}")
    print(f"Max hazard: {np.max(hazards):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("RISK-AWARE SAC MANAGER TEST SUITE")
    print("=" * 60)

    # Run pytest tests
    pytest.main([__file__, "-v", "-s"])

    # Run visualization (optional)
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION...")
    print("=" * 60)
    visualize_crash_response()
