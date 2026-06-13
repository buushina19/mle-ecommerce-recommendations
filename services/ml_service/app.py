import os
import time

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from ml_service.handler import RecommenderHandler
from ml_service.schemas import RecommendationResponse


MODEL_PATH = os.getenv("MODEL_PATH", "models/recommender.pkl")

app = FastAPI(
    title="E-commerce Recommendation Service",
    description="Recommend items for online store visitors (add-to-cart focus)",
    version="1.0.0",
)

handler = RecommenderHandler(MODEL_PATH)

recommendation_counter = Counter("recommendation_requests_total", "Recommendation requests")
recommendation_latency = Histogram("recommendation_latency_seconds", "Recommendation latency")
cold_start_counter = Gauge("recommendation_cold_start_total", "Cold start responses in last request")


Instrumentator().instrument(app).expose(app)


@app.get("/")
def root():
    return {"status": "ok", "service": "ecommerce_recommendations"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/recommendations/{visitorid}", response_model=RecommendationResponse)
def recommend(visitorid: int, k: int = 10):
    start = time.time()
    if k <= 0 or k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")

    items, strategy = handler.recommend(visitorid, k)
    recommendation_counter.inc()
    recommendation_latency.observe(time.time() - start)
    cold_start_counter.set(1 if strategy == "popularity_cold_start" else 0)

    return {"visitorid": visitorid, "recommendations": items, "strategy": strategy}
