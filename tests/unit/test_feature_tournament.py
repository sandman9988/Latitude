import pytest

"""Tests for FeatureTournament."""

import numpy as np

from src.features.feature_tournament import FeatureTournament


class TestFeatureTournamentInit:
    def test_init(self):
        names = ["f1", "f2", "f3"]
        ft = FeatureTournament(n_features=3, feature_names=names)
        assert ft.n_features == 3
        assert ft.tournaments_run == 0
        assert all(ft.feature_active)

    def test_all_features_start_active(self):
        names = [f"f{i}" for i in range(10)]
        ft = FeatureTournament(n_features=10, feature_names=names)
        assert np.sum(ft.feature_active) == 10


class TestFeatureTournamentSamples:
    def test_add_sample(self):
        names = ["f1", "f2"]
        ft = FeatureTournament(n_features=2, feature_names=names, tournament_window=10)
        ft.add_sample(np.array([1.0, 2.0]), target=1.0)
        assert len(ft.feature_values) == 1

    def test_not_ready_below_window(self):
        names = ["f1", "f2"]
        ft = FeatureTournament(n_features=2, feature_names=names, tournament_window=50)
        for i in range(10):
            ft.add_sample(np.array([float(i), float(i * 2)]), target=float(i))
        result = ft.run_tournament()
        assert result["ready"] is False


class TestFeatureTournamentRun:
    def _make_tournament(self, n_features=5, window=50):
        names = [f"f{i}" for i in range(n_features)]
        return FeatureTournament(
            n_features=n_features,
            feature_names=names,
            tournament_window=window,
            survival_threshold=0.3,
        )

    def test_tournament_with_predictive_feature(self):
        rng = np.random.default_rng(42)
        ft = self._make_tournament(n_features=3, window=50)

        for _ in range(50):
            target = rng.choice([-1.0, 1.0])
            features = np.array([
                target + rng.normal(0, 0.1),  # f0: Highly predictive
                rng.normal(0, 1.0),            # f1: Random noise
                rng.normal(0, 1.0),            # f2: Random noise
            ])
            ft.add_sample(features, target=target)

        result = ft.run_tournament()
        assert result["ready"] is True
        assert result["n_active"] >= 1
        # f0 should score highest
        assert ft.feature_scores[0] > ft.feature_scores[1]

    def test_tournament_increments_count(self):
        ft = self._make_tournament(n_features=2, window=10)
        rng = np.random.default_rng(42)
        for _ in range(10):
            ft.add_sample(rng.standard_normal(2), target=rng.normal())
        ft.run_tournament()
        assert ft.tournaments_run == 1

    def test_at_least_some_features_active(self):
        """Even with terrible features, at least 3 stay active."""
        ft = self._make_tournament(n_features=5, window=10)
        ft.survival_threshold = 99.0  # Impossibly high threshold
        rng = np.random.default_rng(42)
        for _ in range(10):
            ft.add_sample(rng.standard_normal(5), target=rng.normal())
        ft.run_tournament()
        assert np.sum(ft.feature_active) >= 3

    def test_regime_stability(self):
        ft = self._make_tournament(n_features=2, window=100)
        rng = np.random.default_rng(42)
        for i in range(100):
            regime = 0 if i < 50 else 1
            target = rng.choice([-1.0, 1.0])
            # f0 is predictive in all regimes, f1 only in regime 0
            noise = rng.normal(0, 0.1)
            f0 = target + noise
            f1 = target + noise if regime == 0 else rng.normal(0, 1.0)
            ft.add_sample(np.array([f0, f1]), target=target, regime=regime)

        result = ft.run_tournament()
        assert result["ready"] is True


class TestFeatureTournamentHelpers:
    def _make(self, n=3):
        names = [f"f{i}" for i in range(n)]
        return FeatureTournament(n_features=n, feature_names=names)

    def test_get_active_features(self):
        ft = self._make()
        mask = ft.get_active_features()
        assert mask.dtype == bool
        assert all(mask)

    def test_get_active_indices(self):
        ft = self._make()
        indices = ft.get_active_indices()
        assert indices == [0, 1, 2]

    def test_filter_features(self):
        ft = self._make()
        ft.feature_active = np.array([True, False, True])
        filtered = ft.filter_features(np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(filtered, [10.0, 30.0])

    def test_reset(self):
        ft = self._make()
        ft.tournaments_run = 5
        ft.feature_active[0] = False
        ft.reset()
        assert ft.tournaments_run == 0
        assert all(ft.feature_active)
        assert len(ft.feature_values) == 0

    def test_safe_correlation_constant_input(self):
        ft = self._make()
        assert ft._safe_correlation(np.ones(10), np.arange(10)) == pytest.approx(0.0)

    def test_safe_correlation_perfect(self):
        ft = self._make()
        x = np.arange(10, dtype=float)
        corr = ft._safe_correlation(x, x)
        assert abs(corr - 1.0) < 1e-10
