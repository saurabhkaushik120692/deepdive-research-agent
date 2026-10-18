from fastapi.testclient import TestClient

from app.main import create_app


def test_healthz():
    r = TestClient(create_app()).get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
