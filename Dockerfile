FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHODOIRUNLIKE_API_ARTIFACT_ROOT=/app/artifacts/api_runs \
    WHODOIRUNLIKE_MODEL_DIR=/app/models/mediapipe

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "whodoirunlike.api:app", "--host", "0.0.0.0", "--port", "8000"]
