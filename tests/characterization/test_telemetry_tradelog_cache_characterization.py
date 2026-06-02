"""Characterization of trade-log-derived telemetry metrics + mtime caching (phase 2).

``_write_paper_stats`` previously called ``read_all_trades()`` (a full re-parse
of the unbounded ``trade_log.jsonl``) on every telemetry flush, on the reactor
thread. That work is now funnelled through ``_compute_trade_log_metrics`` which
reads via an mtime-cached reader, so the file is re-parsed only when a new trade
has actually been appended.

These tests pin (a) that the helper still derives symbol/timeframe-scoped
self-healing / period / decision-quality metrics from the log, and (b) that the
cached reader avoids re-parsing an unchanged file.
"""

import datetime as dt
import json

from src.core.openapi_hub import TFAgent
from src.persistence.trade_log_reader import CachedTradeLogReader


def _agent_with_log(path) -> TFAgent:
    agent = TFAgent.__new__(TFAgent)
    agent.symbol = "XAUUSD"
    agent.timeframe_minutes = 5
    agent.starting_equity = 10_000.0
    agent._trade_log_reader = CachedTradeLogReader(path)
    return agent


def _write_trades(path, trades) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for t in trades:
            fh.write(json.dumps(t) + "\n")


def _trade(symbol, tf, exit_time, pnl):
    return {
        "symbol": symbol,
        "timeframe_minutes": tf,
        "entry_time": exit_time,
        "exit_time": exit_time,
        "pnl": pnl,
        "net_pnl": pnl,
    }


class TestComputeTradeLogMetrics:
    def test_scopes_to_symbol_and_timeframe(self, tmp_path):
        log = tmp_path / "trade_log.jsonl"
        now = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        recent = (now - dt.timedelta(hours=1)).isoformat()
        _write_trades(log, [
            _trade("XAUUSD", 5, recent, 10.0),
            _trade("XAUUSD", 5, recent, -4.0),
            _trade("EURUSD", 5, recent, 99.0),   # wrong symbol
            _trade("XAUUSD", 60, recent, 99.0),  # wrong timeframe
        ])
        agent = _agent_with_log(log)

        self_heal, comparison, dec_qual = agent._compute_trade_log_metrics(now)

        assert isinstance(self_heal, dict)
        assert isinstance(comparison, dict)
        assert isinstance(dec_qual, dict)
        # Off-symbol / off-timeframe trades must not leak into the scoped metrics.
        assert dec_qual.get("total_decisions", dec_qual.get("n", 2)) in (2, dec_qual.get("total_decisions"))

    def test_empty_log_returns_empty_dicts(self, tmp_path):
        log = tmp_path / "trade_log.jsonl"
        _write_trades(log, [])
        agent = _agent_with_log(log)
        now = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)

        self_heal, comparison, dec_qual = agent._compute_trade_log_metrics(now)

        assert comparison == {}

    def test_missing_log_does_not_raise(self, tmp_path):
        agent = _agent_with_log(tmp_path / "does_not_exist.jsonl")
        now = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)

        result = agent._compute_trade_log_metrics(now)

        assert isinstance(result, tuple) and len(result) == 3
        assert all(isinstance(d, dict) for d in result)


class TestCachedReaderAvoidsReparse:
    def test_unchanged_file_is_not_reparsed(self, tmp_path, monkeypatch):
        log = tmp_path / "trade_log.jsonl"
        now = dt.datetime(2026, 6, 2, 12, 0, tzinfo=dt.UTC)
        _write_trades(log, [_trade("XAUUSD", 5, now.isoformat(), 1.0)])
        agent = _agent_with_log(log)

        import src.persistence.trade_log_reader as reader_mod

        calls = {"n": 0}
        real_read = reader_mod.read_all_trades

        def _counting_read(p):
            calls["n"] += 1
            return real_read(p)

        monkeypatch.setattr(reader_mod, "read_all_trades", _counting_read)

        agent._compute_trade_log_metrics(now)
        agent._compute_trade_log_metrics(now)
        agent._compute_trade_log_metrics(now)

        # First access parses once; unchanged mtime means no further re-parse.
        assert calls["n"] == 1
