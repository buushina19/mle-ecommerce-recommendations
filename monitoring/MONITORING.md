# Мониторинг сервиса рекомендаций e-commerce

## Online-метрики (Prometheus)

| Метрика | Тип | Описание |
|---------|-----|----------|
| `recommendation_requests_total` | Counter | Число запросов `/recommendations/{visitorid}` |
| `recommendation_latency_seconds` | Histogram | Latency API |
| `recommendation_cold_start_total` | Gauge | 1 — ответ popularity (новый пользователь) |
| `http_requests_total` | Counter | HTTP-коды (FastAPI instrumentator) |

Просмотр:
```bash
curl http://127.0.0.1:8000/metrics
```

## Offline-метрики модели

Файл `artifacts/metrics.json` после обучения:

| Метрика | Назначение |
|---------|------------|
| `recall@10` | Полнота по add-to-cart в hold-out |
| `precision@10` | Точность топ-10 |
| `map@10` | Ranking quality |
| `hit_rate@10` | Доля пользователей с хотя бы 1 верным item |
| `coverage@10` | Разнообразие рекомендаций |

## Data drift (proxy)

- доля cold-start запросов (`recommendation_cold_start_total`)
- рост latency p95
- падение offline `recall@10` после scheduled retrain (сравнение MLflow runs)

## Где реализовано

- `services/ml_service/app.py` — Counter, Histogram, Gauge
- `src/train.py` — offline metrics + MLflow
- `dags/retrain_ecommerce_recommendations.py` — scheduled retrain
