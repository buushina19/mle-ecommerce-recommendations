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

- **Baseline:** popularity по add-to-cart
- **Модель:** ALS (implicit), веса view=1 / addtocart=5
- **Validation:** temporal split 85/15

| Метрика | Baseline | ALS |
|---------|----------|-----|
| Recall@10 | 0.025 | **0.030** |
| Hit Rate@10 | 0.053 | **0.060** |
| Coverage@10 | 0.0001 | **0.0085** |

## Мониторинг

См. `monitoring/MONITORING.md`, метрики на `/metrics`.

**Random seed:** `42`
