from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api import create_app


def test_health_endpoint_returns_ok():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_uses_agent_response(monkeypatch):
    fake_agent = Mock()
    fake_agent.invoke.return_value = {"messages": [Mock(content="comparison result")]}

    monkeypatch.setattr("app.api.get_agent", lambda: fake_agent)
    client = TestClient(create_app())

    response = client.post("/chat", json={"message": "compare tools"})

    assert response.status_code == 200
    assert response.json() == {"response": "comparison result"}
    fake_agent.invoke.assert_called_once()


def test_create_app_uses_config_metadata(monkeypatch):
    monkeypatch.setenv("APP_TITLE", "Travel Demo API")
    monkeypatch.setenv("APP_DESCRIPTION", "Configured description")
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    from app.config import get_config

    get_config.cache_clear()
    app = create_app()

    assert app.title == "Travel Demo API"
    assert app.description == "Configured description"
    assert app.version == "9.9.9"

    get_config.cache_clear()
