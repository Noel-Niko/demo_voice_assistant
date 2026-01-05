from fastapi.testclient import TestClient

from src.gateway.routes import create_app


def test_config_reports_rest_mode_by_default(monkeypatch):
    app = create_app()
    client = TestClient(app)

    # Ensure default is REST via constants
    from src.constants import common_constants as cc

    monkeypatch.setattr(cc.ASRModes, "DEFAULT", cc.ASRModes.REST, raising=True)

    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["stt"]["mode"] == "REST"


def test_config_reports_grpc_mode_when_default_grpc(monkeypatch):
    app = create_app()
    client = TestClient(app)

    from src.constants import common_constants as cc

    monkeypatch.setattr(cc.ASRModes, "DEFAULT", cc.ASRModes.GRPC, raising=True)

    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["stt"]["mode"] == "GRPC"
