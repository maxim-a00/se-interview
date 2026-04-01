from app.config import get_config


def test_get_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.25")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "demo-project")
    get_config.cache_clear()

    config = get_config()

    assert config.openai_model == "gpt-4.1-mini"
    assert config.openai_temperature == 0.25
    assert config.phoenix_project_name == "demo-project"

    get_config.cache_clear()


def test_get_config_uses_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
    monkeypatch.delenv("PHOENIX_PROJECT_NAME", raising=False)
    get_config.cache_clear()

    config = get_config()

    assert config.openai_model == "gpt-4o"
    assert config.openai_temperature == 0.0
    assert config.phoenix_project_name == "se-interview"

    get_config.cache_clear()
