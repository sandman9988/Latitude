from src.core.openapi_hub import OpenAPIHub, _CTRL_PARAM_RELOAD


class _ReloadAgent:
    def __init__(self):
        self.reload_count = 0

    def reload_learned_parameters(self) -> None:
        self.reload_count += 1


def test_param_reload_control_reloads_agents_and_removes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    reload_file = data_dir / _CTRL_PARAM_RELOAD
    reload_file.write_text("{}", encoding="utf-8")

    agent = _ReloadAgent()
    hub = object.__new__(OpenAPIHub)
    hub.symbol = "XAUUSD"
    hub.agents = {5: agent}

    hub._poll_param_reload()

    assert agent.reload_count == 1
    assert not reload_file.exists()
