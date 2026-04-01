"""Centralized environment-backed configuration.

Design note:
- Configuration was consolidated here after environment values started being
  read in multiple modules.
- A single cached config object keeps model settings, API metadata, and
  Phoenix settings easy to test and reason about.
- This stays intentionally simple for the assignment: a small dataclass-based
  config avoids the overhead of a heavier settings framework.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class AppConfig:
    app_title: str
    app_description: str
    app_version: str
    openai_model: str
    openai_temperature: float
    phoenix_collector_endpoint: str
    phoenix_project_name: str
    phoenix_working_dir: Path


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    default_working_dir = Path(__file__).resolve().parents[1] / ".phoenix"
    return AppConfig(
        app_title=_get_env("APP_TITLE", "LangGraph Agent API"),
        app_description=_get_env(
            "APP_DESCRIPTION",
            "A simple API for interacting with a LangGraph travel assistant.",
        ),
        app_version=_get_env("APP_VERSION", "0.1.0"),
        openai_model=_get_env("OPENAI_MODEL", "gpt-4o"),
        openai_temperature=_get_env_float("OPENAI_TEMPERATURE", 0.0),
        phoenix_collector_endpoint=_get_env(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:6006/v1/traces",
        ),
        phoenix_project_name=_get_env("PHOENIX_PROJECT_NAME", "se-interview"),
        phoenix_working_dir=Path(_get_env("PHOENIX_WORKING_DIR", str(default_working_dir))),
    )


__all__ = ["AppConfig", "get_config"]
