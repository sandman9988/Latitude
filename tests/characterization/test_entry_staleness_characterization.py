"""Characterization of the entry execution-staleness fix + guard (phase 1).

Exits run on every tick and fill at the *live* tick mid
(``_handle_exit_on_tick`` -> ``_force_close_price(mid, ...)``). Entries are
decided on completed-bar features but now also execute at the live mid
(``self.last_mid``) rather than the close of the just-completed bar, removing
the entry/exit price-staleness asymmetry.

To stop the stale-feature edge from being applied to a market that has already
moved on (or to a stalled/gap-filled feed), a staleness guard blocks the entry
when the live mid has drifted from the bar close by more than
``_ENTRY_STALE_DRIFT_VOL_MULT`` * realized-vol, or when the live tick is older
than ``_ENTRY_STALE_MAX_AGE_TF_MULT`` bar periods.
"""

import datetime as dt
import types

import pytest

from src.core.openapi_hub import TFAgent


def _flat_hub() -> TFAgent:
    hub = TFAgent.__new__(TFAgent)
    hub.position = None
    hub.symbol = "XAUUSD"
    hub.tf_label = "M5"
    hub.timeframe_minutes = 5
    hub.last_mid = 0.0
    hub.last_ts = None
    hub.friction_calc = types.SimpleNamespace(depth_buffer=0.0)
    hub._entry_circuit_breaker_blocks = lambda: False
    hub._entry_depth_gate_blocks = lambda floor: False
    hub._realized_vol = lambda: 0.01
    hub._entry_soft_gates = lambda: []
    hub._depth_ratio = lambda: 1.0
    hub._apply_entry_dynamic_floor = lambda action, conf, gated: (action, 0.5)
    hub._apply_paper_entry_guard = lambda action, conf, gated: action
    hub._snapshot_entry_lifecycle = lambda **kw: None
    hub._log_entry_decision = lambda *a, **k: None
    hub._maybe_add_no_entry_experience = lambda trig_state: None
    return hub


class TestEntryExecutionAtLiveMid:
    def test_entry_long_fills_at_live_mid_not_stale_bar_close(self):
        hub = _flat_hub()
        bucket_start = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        bar_close = 2000.0
        hub.last_mid = 2002.0  # fresh live price, within drift tolerance
        hub.last_ts = bucket_start + dt.timedelta(seconds=30)
        half_spread = 0.5
        bar = (bucket_start, 1998.0, 2003.0, 1996.0, bar_close)
        hub._decide_flat_entry = lambda vol, dr: (1, 0.7, 0.002, object())

        captured: dict = {}
        hub._open_position = lambda ts, direction, fill_price, *a: captured.update(
            ts=ts, direction=direction, fill_price=fill_price,
        )
        hub._handle_flat(bar, half_spread)

        assert captured["fill_price"] == pytest.approx(hub.last_mid + half_spread)
        assert captured["fill_price"] != pytest.approx(bar_close + half_spread)
        assert captured["ts"] == hub.last_ts

    def test_entry_short_fills_at_live_mid(self):
        hub = _flat_hub()
        bucket_start = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        bar_close = 2000.0
        hub.last_mid = 1998.0
        hub.last_ts = bucket_start + dt.timedelta(seconds=30)
        half_spread = 0.5
        bar = (bucket_start, 2001.0, 2004.0, 1997.0, bar_close)
        hub._decide_flat_entry = lambda vol, dr: (2, 0.7, 0.002, object())

        captured: dict = {}
        hub._open_position = lambda ts, direction, fill_price, *a: captured.update(
            direction=direction, fill_price=fill_price,
        )
        hub._handle_flat(bar, half_spread)

        assert captured["direction"] == -1
        assert captured["fill_price"] == pytest.approx(hub.last_mid - half_spread)

    def test_entry_falls_back_to_bar_close_when_no_live_mid(self):
        hub = _flat_hub()
        bucket_start = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        bar_close = 2000.0
        hub.last_mid = 0.0  # no live tick yet
        hub.last_ts = None
        half_spread = 0.5
        bar = (bucket_start, 1998.0, 2003.0, 1996.0, bar_close)
        hub._decide_flat_entry = lambda vol, dr: (1, 0.7, 0.002, object())

        captured: dict = {}
        hub._open_position = lambda ts, direction, fill_price, *a: captured.update(
            ts=ts, fill_price=fill_price,
        )
        hub._handle_flat(bar, half_spread)

        assert captured["fill_price"] == pytest.approx(bar_close + half_spread)
        assert captured["ts"] == bucket_start


class TestEntryStalenessGuard:
    def test_entry_blocked_when_drift_exceeds_vol_multiple(self):
        hub = _flat_hub()
        bucket_start = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        bar_close = 2000.0
        # vol=0.01 -> max drift = 1.5% = 30.0; 2050 drifts 2.5% -> blocked.
        hub.last_mid = 2050.0
        hub.last_ts = bucket_start + dt.timedelta(seconds=30)
        bar = (bucket_start, 1998.0, 2003.0, 1996.0, bar_close)
        hub._decide_flat_entry = lambda vol, dr: (1, 0.7, 0.002, object())

        opened: list = []
        hub._open_position = lambda *a, **k: opened.append(a)
        hub._handle_flat(bar, 0.5)

        assert opened == []

    def test_entry_blocked_when_tick_too_old(self):
        hub = _flat_hub()
        bucket_start = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        bar_close = 2000.0
        hub.last_mid = bar_close  # no drift, isolate the age guard
        # tf=5 -> max age = 3 * 5 * 60 = 900s; 1000s is stale.
        hub.last_ts = bucket_start + dt.timedelta(seconds=1000)
        bar = (bucket_start, 1998.0, 2003.0, 1996.0, bar_close)
        hub._decide_flat_entry = lambda vol, dr: (1, 0.7, 0.002, object())

        opened: list = []
        hub._open_position = lambda *a, **k: opened.append(a)
        hub._handle_flat(bar, 0.5)

        assert opened == []

    def test_entry_proceeds_within_drift_and_age_tolerance(self):
        hub = _flat_hub()
        bucket_start = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        bar_close = 2000.0
        hub.last_mid = 2005.0  # 0.25% drift, well under 1.5%
        hub.last_ts = bucket_start + dt.timedelta(seconds=120)  # under 900s
        bar = (bucket_start, 1998.0, 2003.0, 1996.0, bar_close)
        hub._decide_flat_entry = lambda vol, dr: (1, 0.7, 0.002, object())

        opened: list = []
        hub._open_position = lambda *a, **k: opened.append(a)
        hub._handle_flat(bar, 0.5)

        assert len(opened) == 1
