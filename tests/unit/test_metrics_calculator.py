import pytest

from src.utils.metrics_calculator import period_metrics


def test_period_metrics_prefers_normalized_capture_ratio():
    trades = [
        {
            "pnl": 10.0,
            "mfe": 0.1,
            "capture_ratio": 0.5,
        },
        {
            "pnl": -5.0,
            "mfe": 0.2,
            "capture_ratio": -0.25,
        },
    ]

    metrics = period_metrics(trades)

    assert metrics["avg_capture_ratio"] == pytest.approx(0.125)
