import pytest

"""Tests for OrderBook and VPINCalculator."""

from src.core.order_book import OrderBook, VPINCalculator


class TestOrderBook:
    def test_init(self):
        ob = OrderBook(depth=5)
        assert ob.depth == 5
        assert len(ob.bids) == 0
        assert len(ob.asks) == 0

    def test_update_bid(self):
        ob = OrderBook()
        ob.update_level("BID", 100.0, 10.0)
        assert ob.bids[100.0] == pytest.approx(10.0)

    def test_update_ask(self):
        ob = OrderBook()
        ob.update_level("ASK", 101.0, 5.0)
        assert ob.asks[101.0] == pytest.approx(5.0)

    def test_remove_level_on_zero_size(self):
        ob = OrderBook()
        ob.update_level("BID", 100.0, 10.0)
        ob.update_level("BID", 100.0, 0.0)
        assert 100.0 not in ob.bids

    def test_invalid_side_ignored(self):
        ob = OrderBook()
        ob.update_level("INVALID", 100.0, 10.0)
        assert len(ob.bids) == 0
        assert len(ob.asks) == 0

    def test_invalid_price_ignored(self):
        ob = OrderBook()
        ob.update_level("BID", -1.0, 10.0)
        ob.update_level("BID", 0.0, 10.0)
        ob.update_level("BID", float("inf"), 10.0)
        assert len(ob.bids) == 0

    def test_invalid_size_ignored(self):
        ob = OrderBook()
        ob.update_level("BID", 100.0, float("nan"))
        assert len(ob.bids) == 0

    def test_best_bid_ask(self):
        ob = OrderBook()
        ob.update_level("BID", 99.0, 10.0)
        ob.update_level("BID", 100.0, 5.0)
        ob.update_level("ASK", 101.0, 8.0)
        ob.update_level("ASK", 102.0, 3.0)
        bid, ask = ob.best_bid_ask()
        assert bid == pytest.approx(100.0)
        assert ask == pytest.approx(101.0)

    def test_best_bid_ask_empty(self):
        ob = OrderBook()
        bid, ask = ob.best_bid_ask()
        assert bid is None
        assert ask is None

    def test_spread(self):
        ob = OrderBook()
        ob.update_level("BID", 100.0, 10.0)
        ob.update_level("ASK", 101.0, 10.0)
        assert ob.spread() == pytest.approx(1.0)

    def test_spread_empty(self):
        ob = OrderBook()
        assert ob.spread() is None

    def test_spread_crossed_book(self):
        ob = OrderBook()
        ob.update_level("BID", 101.0, 10.0)
        ob.update_level("ASK", 100.0, 10.0)
        assert ob.spread() == pytest.approx(0.0)

    def test_depth_sum(self):
        ob = OrderBook()
        ob.update_level("BID", 99.0, 10.0)
        ob.update_level("BID", 100.0, 20.0)
        ob.update_level("ASK", 101.0, 5.0)
        ob.update_level("ASK", 102.0, 15.0)
        b_sum, a_sum = ob.depth_sum(levels=5)
        assert b_sum == pytest.approx(30.0)
        assert a_sum == pytest.approx(20.0)

    def test_depth_sum_empty(self):
        ob = OrderBook()
        b_sum, a_sum = ob.depth_sum()
        assert b_sum == pytest.approx(0.0)
        assert a_sum == pytest.approx(0.0)

    def test_imbalance(self):
        ob = OrderBook()
        ob.update_level("BID", 100.0, 60.0)
        ob.update_level("ASK", 101.0, 40.0)
        imb = ob.imbalance()
        assert abs(imb - 0.2) < 1e-10  # (60-40)/100

    def test_imbalance_empty(self):
        ob = OrderBook()
        assert ob.imbalance() == pytest.approx(0.0)

    def test_pruning(self):
        ob = OrderBook(depth=2)
        ob.update_level("BID", 98.0, 1.0)
        ob.update_level("BID", 99.0, 2.0)
        ob.update_level("BID", 100.0, 3.0)
        # Only top 2 bids should remain (100, 99)
        assert len(ob.bids) == 2
        assert 98.0 not in ob.bids

    def test_reset(self):
        ob = OrderBook()
        ob.update_level("BID", 100.0, 10.0)
        ob.update_level("ASK", 101.0, 5.0)
        ob.reset()
        assert len(ob.bids) == 0
        assert len(ob.asks) == 0


class TestVPINCalculator:
    def test_init(self):
        vpin = VPINCalculator(bucket_volume=10.0, window=5)
        assert vpin.bucket_volume == pytest.approx(10.0)
        assert vpin.get_vpin() == pytest.approx(0.0)

    def test_incomplete_bucket(self):
        vpin = VPINCalculator(bucket_volume=100.0)
        result = vpin.update(10.0, "BUY")
        assert result is None  # Bucket not filled

    def test_complete_bucket(self):
        vpin = VPINCalculator(bucket_volume=10.0)
        vpin.update(5.0, "BUY")
        result = vpin.update(5.0, "SELL")
        assert result is not None  # Bucket filled
        assert 0.0 <= result <= 1.0

    def test_pure_buy_bucket(self):
        vpin = VPINCalculator(bucket_volume=10.0)
        result = vpin.update(10.0, "BUY")
        assert result == pytest.approx(1.0)  # All buy = max imbalance

    def test_balanced_bucket(self):
        vpin = VPINCalculator(bucket_volume=10.0)
        vpin.update(5.0, "BUY")
        result = vpin.update(5.0, "SELL")
        assert result == pytest.approx(0.0)  # Perfectly balanced

    def test_vpin_rolling_average(self):
        vpin = VPINCalculator(bucket_volume=10.0, window=5)
        # Fill several buckets
        for _ in range(5):
            vpin.update(10.0, "BUY")  # All-buy buckets
        assert vpin.get_vpin() == pytest.approx(1.0)

    def test_invalid_volume_ignored(self):
        vpin = VPINCalculator(bucket_volume=10.0)
        assert vpin.update(-5.0, "BUY") is None
        assert vpin.update(0.0, "BUY") is None
        assert vpin.update(float("inf"), "BUY") is None

    def test_invalid_side_ignored(self):
        vpin = VPINCalculator(bucket_volume=10.0)
        assert vpin.update(5.0, "INVALID") is None

    def test_reset(self):
        vpin = VPINCalculator(bucket_volume=10.0)
        vpin.update(10.0, "BUY")
        vpin.reset()
        assert vpin.get_vpin() == pytest.approx(0.0)
        assert vpin.current_buy == pytest.approx(0.0)
        assert vpin.current_sell == pytest.approx(0.0)

    def test_get_stats(self):
        vpin = VPINCalculator(bucket_volume=10.0)
        stats = vpin.get_stats()
        assert stats["vpin"] == pytest.approx(0.0)
        assert "mean" in stats
        assert "std" in stats
        assert "zscore" in stats

    def test_get_stats_with_data(self):
        vpin = VPINCalculator(bucket_volume=10.0, window=5)
        for _ in range(3):
            vpin.update(10.0, "BUY")
        stats = vpin.get_stats()
        assert stats["vpin"] == pytest.approx(1.0)
        assert stats["std"] == pytest.approx(0.0)  # All same values
