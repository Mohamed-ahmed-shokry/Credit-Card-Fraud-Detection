FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FRAUD_MODEL_PATH=/app/artifacts/model

WORKDIR /app

RUN groupadd --system --gid 10001 fraud \
    && useradd --system --uid 10001 --gid fraud --no-create-home fraud

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "--factory", "fraud_detection.api:app_from_environment", "--host", "0.0.0.0", "--port", "8000"]
