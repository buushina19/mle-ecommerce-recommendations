FROM python:3.10-slim

WORKDIR /app

COPY services/requirements.txt /app/services/requirements.txt
RUN pip install --no-cache-dir -r /app/services/requirements.txt

COPY src /app/src
COPY services/ml_service /app/services/ml_service
COPY models /app/models
COPY artifacts /app/artifacts

ENV MODEL_PATH=/app/models/recommender.pkl
ENV PYTHONPATH=/app:/app/services

EXPOSE 8000
CMD ["uvicorn", "ml_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
