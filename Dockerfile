# Slim runtime image. The model artifact is baked in rather than downloaded at
# start-up so the container is self-contained and its behaviour is pinned to a
# known model version — no surprise swap under a running service.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first: this layer is cached unless requirements change, so code
# edits rebuild in seconds instead of reinstalling scipy every time.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY serve/ ./serve/
COPY artifacts/pipeline.joblib ./artifacts/pipeline.joblib

ENV MODEL_PATH=/app/artifacts/pipeline.joblib \
    HISTORY_LEN=50 \
    PYTHONUNBUFFERED=1

# Run as a non-root user: a scoring service has no reason to be root.
RUN useradd --create-home --uid 10001 scorer && chown -R scorer /app
USER scorer

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
