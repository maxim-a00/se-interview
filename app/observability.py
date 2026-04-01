import os
from functools import lru_cache

from app.config import get_config


@lru_cache(maxsize=1)
def configure_phoenix():
    """Configure Phoenix tracing once for the current process."""
    config = get_config()
    os.environ.setdefault("PHOENIX_WORKING_DIR", str(config.phoenix_working_dir))

    from phoenix.otel import register

    return register(
        endpoint=config.phoenix_collector_endpoint,
        project_name=config.phoenix_project_name,
        auto_instrument=True,
    )
