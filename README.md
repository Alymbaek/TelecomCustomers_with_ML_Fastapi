# Telco Customer Churn Prediction 📉
> 🚀 Side-project для прогнозирования оттока клиентов телеком-компании.  
> **Stack**: Pandas · NumPy · Matplotlib/Seaborn · Scikit-learn · FastAPI · Uvicorn

---

## 1. Задача
* Предсказать, уйдёт ли клиент в ближайшем будущем (`Churn`).
* Сравнить две модели — **Logistic Regression** и **Decision Tree** — и выбрать лучшую.
* Поднять REST API, принимающее 7 признаков и возвращающее вероятность оттока.

## 2. Данные
| Признак         | Тип    | Описание                                   |
|-----------------|--------|--------------------------------------------|
| tenure          | int    | Срок пребывания клиента (мес.)             |
| MonthlyCharges  | float  | Месячный платёж                            |
| TotalCharges    | float  | Общая сумма платежей                       |
| Contract        | object | Month-to-month / One year / Two year       |
| InternetService | object | DSL / Fiber optic / No                     |
| OnlineSecurity  | object | Yes / No / No internet service            |
| TechSupport     | object | Yes / No / No internet service            |
| **Churn**       | object | Yes (ушёл) / No (остался)                  |

Чистый датасет после EDA лежит в `data/churn_clean.csv`.

## 3. Быстрый старт

### Локальный запуск
```bash
git clone https://github.com/<user>/telco-churn.git
cd telco-churn
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python notebooks/03_train_models.py                       # обучить и сохранить лучшую модель
uvicorn telecom.main:telco_app --reload --port 8001       # запустить API

curl -X POST http://localhost:8001/predict/ \
  -H "Content-Type: application/json" \
  -d '{
        "tenure": 12,
        "MonthlyCharges": 75.3,
        "TotalCharges": "835.4",
        "Contract": "Month-to-month",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "No"
      }'

# ➜ {"approved": true, "prob": 0.84}

4. Результаты
Модель	Accuracy	Precision	Recall	F1	ROC AUC
Logistic Regression	0.81	0.78	0.72	0.75	0.85
Decision Tree	0.76	0.70	0.78	0.74	0.82

Логистическая регрессия показала лучший баланс метрик, поэтому именно она развёрнута в API.
5. Структура проекта
├── data/                 # raw & cleaned datasets
├── notebooks/            # EDA, preprocessing, training
├── telecom/              # FastAPI app
│   ├── main.py
│   └── model_random_telco.pkl
├── requirements.txt
└── README.md

6. Roadmap
 Добавить GridSearchCV для дерева

 Перейти на Pipeline с ColumnTransformer

 Докеризация и деплой на AWS EC2

 Frontend-форма для ввода признаков

7. Полезные ссылки
Отчёт EDA (nbviewer) — coming soon

Dataset на Kaggle

Алымбек Ибрагимов · 17 y.o. Junior Python/ML Engineer · Kyrgyzstan
Telegram • Email
