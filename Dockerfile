FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependencies only -> this layer is cached until pyproject.toml / uv.lock change.
#    --no-default-groups: runtime deps only (no torch, no notebooks, no pytest)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

# 2) Application code + the ONNX model
COPY src ./src
COPY models/cnn.onnx ./models/cnn.onnx
RUN uv sync --frozen --no-default-groups

# 3) Don't run as root
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["/app/.venv/bin/uvicorn", "bearing_fault_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]