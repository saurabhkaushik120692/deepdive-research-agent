import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Keep tests independent of a developer's local .env and data folder."""
    monkeypatch.chdir(tmp_path)
