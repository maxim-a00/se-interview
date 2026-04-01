FROM python:3.13-slim

ENV POETRY_VERSION=2.3.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml README.md ./
COPY app ./app
COPY api.py agent.py ./

RUN poetry install --only main --no-interaction --no-ansi

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
