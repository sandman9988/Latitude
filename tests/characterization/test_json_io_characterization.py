"""Characterization tests for the consolidated persistence.json_io primitives.

Pins the observable behaviour of the canonical atomic-write and durable
JSONL-append helpers that existing module-local writers delegate to.
"""

import json
import os
from datetime import UTC, datetime

import pytest

from src.persistence.json_io import append_jsonl_durable, save_json_atomic


def test_save_json_atomic_round_trip(tmp_path):
    target = tmp_path / "snap.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": "x"}
    save_json_atomic(target, payload)
    assert json.loads(target.read_text()) == payload


def test_save_json_atomic_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "snap.json"
    save_json_atomic(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}


def test_save_json_atomic_honours_indent(tmp_path):
    target = tmp_path / "snap.json"
    save_json_atomic(target, {"a": 1}, indent=2)
    text = target.read_text()
    assert "\n" in text
    assert "  " in text


def test_save_json_atomic_uses_default_serializer(tmp_path):
    target = tmp_path / "snap.json"
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    save_json_atomic(target, {"t": when}, default=lambda o: o.isoformat())
    assert json.loads(target.read_text())["t"] == when.isoformat()


def test_save_json_atomic_cleans_temp_and_preserves_original_on_failure(tmp_path):
    target = tmp_path / "snap.json"
    save_json_atomic(target, {"orig": 1})

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        save_json_atomic(target, {"bad": Unserializable()})

    assert json.loads(target.read_text()) == {"orig": 1}
    leftovers = [p for p in tmp_path.iterdir() if p.name != "snap.json"]
    assert leftovers == []


def test_append_jsonl_durable_appends_multiple_lines(tmp_path):
    target = tmp_path / "log.jsonl"
    append_jsonl_durable(target, {"i": 0})
    append_jsonl_durable(target, {"i": 1})
    lines = target.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"i": 0}, {"i": 1}]


def test_append_jsonl_durable_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "log.jsonl"
    append_jsonl_durable(target, {"x": 1})
    assert json.loads(target.read_text().strip()) == {"x": 1}


def test_append_jsonl_durable_default_serializer(tmp_path):
    target = tmp_path / "log.jsonl"
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    append_jsonl_durable(target, {"t": when})
    assert json.loads(target.read_text().strip())["t"] == str(when)


def test_append_jsonl_durable_compact_separators(tmp_path):
    target = tmp_path / "log.jsonl"
    append_jsonl_durable(target, {"a": 1, "b": 2})
    line = target.read_text().strip()
    assert line == '{"a":1,"b":2}'
    assert line.endswith("}")
    assert os.path.exists(target)
