# Scoring service image. Build after `make run` has exported the champion to models/champion/.
#   docker build -t claims-pure-premium:local .
#   docker run --rm -p 8000:8000 claims-pure-premium:local
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CLAIMS_MODEL_URI=/app/models/champion \
    MLFLOW_DISABLE_TELEMETRY=true

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ src/
COPY models/champion/ models/champion/
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready').status==200 else 1)"
CMD ["uvicorn", "claims_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
