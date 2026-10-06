FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Exactly the versions in uv.lock: a rebuild cannot pick up a breaking release.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY *.py ./
# RAG index, built locally by build_index.py.
COPY data/index.json ./data/
# The DSL specification and cookbook: templates and required fields for dsl.py.
COPY data/docs/tutorial_dsl_native_apps.md data/docs/tutorial_dsl_device_apps.md data/docs/tutorial_cookbook.md ./data/docs/

ENV IDE_BACKEND_URL=http://host.docker.internal:3001/api

# Don't run as root.
RUN useradd --create-home --uid 10001 hyperion
USER hyperion

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["/app/.venv/bin/python", "main.py"]
