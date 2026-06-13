# Рекомендации товаров в e-commerce

## Описание

Интернет-магазин хочет предложить пользователю товары, которые он с высокой вероятностью **добавит в корзину**.  
Работаем с event-логами Retail Rocket: просмотры, add-to-cart, покупки.

**Технологии:** Python, pandas, implicit ALS, MLflow, FastAPI, Docker, Prometheus, Airflow.

## Структура

```
mle-ecommerce-recommendations/
├── Readme.md
├── requirements.txt
├── data/                         # CSV локально
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/                          # data, eda, metrics, train
├── scripts/                      # train.sh, start_mlflow.sh, start_service.sh
├── dags/                         # Airflow DAG дообучения
├── services/                     # FastAPI + Docker
├── monitoring/MONITORING.md
└── models/recommender.pkl
```

## Установка

```bash
git clone <repo-url>
cd mle-ecommerce-recommendations
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

### Данные

Распакуйте архив в `data/`:
- `events.csv`
- `category_tree.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`

## Запуск

```bash
chmod +x scripts/*.sh
./scripts/train.sh          # EDA + обучение + MLflow
./scripts/start_mlflow.sh   # UI :5000
./scripts/start_service.sh  # API :8000
python test_service.py
```

Docker:
```bash
cd services && docker compose up --build
```

Airflow DAG: `dags/retrain_ecommerce_recommendations.py` (еженедельное переобучение).

---

## Бизнес → ML

**Бизнес:** увеличить add-to-cart через персональные рекомендации.

**ML:** для visitorid ранжировать itemid по вероятности add-to-cart.

**Метрики:** Recall@10, Precision@10, MAP@10, Hit Rate@10, Coverage@10.

## Моделирование

- **Stage 1 (candidate generation):** ALS (implicit) + popularity fallback.
- **Stage 2 (reranking):** LogisticRegression по фичам кандидатов:
  - `als_score`
  - `pop_score`
  - `cat_popularity`
  - `cat_affinity` (предпочитаемая категория пользователя)
- **Использование item metadata:** category tree + item_properties (`categoryid`) встроены в rerank-фичи.
- **Validation:** rolling-time (квантили 0.75 и 0.85) + deployment window 0.85.

| Метрика (deployment fold) | Baseline | ALS | ALS + Rerank |
|---------|----------|-----|------|
| Recall@10 | 0.071 | 0.060 | 0.571* |
| Hit Rate@10 | 0.071 | 0.071 | 0.714* |
| Coverage@10 | 0.0001 | 0.0020 | 0.0277* |

Полная история фолдов и метрик сохраняется в `artifacts/metrics.json` и в MLflow.

\* Для `ALS + Rerank` в текущем deployment fold мало пользователей в оценке (`eval_users=7`), поэтому метрики интерпретируются с осторожностью и обязательно сверяются с rolling validation.

## Мониторинг и обновление модели

- Prometheus-метрики API: `/metrics`.
- Пороговые условия и алерты: `monitoring/MONITORING.md`.
- Airflow DAG `dags/retrain_ecommerce_recommendations.py`:
  1. валидация входных данных;
  2. retrain;
  3. проверка артефактов;
  4. champion/challenger gating по `recall@10`;
  5. промоут модели только при приемлемом качестве.

**Random seed:** `42`
