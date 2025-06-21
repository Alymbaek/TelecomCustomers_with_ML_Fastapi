# Telco Customer Churn Prediction 📉

> 🚀 Side-project для прогнозирования оттока клиентов телеком-компании.  
> **Stack**: Pandas · NumPy · Matplotlib/Seaborn · Scikit-learn · FastAPI · Uvicorn

---

## 1. Задача

* Предсказать, уйдёт ли клиент в ближайшем будущем (`Churn`).
* Сравнить две модели — **Logistic Regression** и **Decision Tree** — и выбрать лучшую.
* Поднять REST API, принимающее 7 признаков и возвращающее вероят-ть оттока.

## 2. Данные

| Признак | Тип | Описание |
|---------|-----|----------|
| tenure | `int` | Срок пребывания клиента (мес.) |
| MonthlyCharges | `float` | Месячный платёж |
| TotalCharges | `float` | Общая сумма платежей |
| Contract | `object` | Month-to-month / One year / Two year |
| InternetService | `object` | DSL / Fiber optic / No |
| OnlineSecurity | `object` | Yes / No / No internet service |
| TechSupport | `object` | Yes / No / No internet service |
| **Churn** | `object` | Yes (ушёл) / No (остался) |

Чистый датасет после EDA лежит в `data/churn_clean.csv`.

## 3. Быстрый старт

### Локальный запуск

```bash
git clone https://github.com/<user>/telco-churn.git
cd telco-churn
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Обучить модели и сохранить лучшую в model_random_telco.pkl
python notebooks/03_train_models.py

# 2) Запустить API
uvicorn telecom.main:telco_app --reload --port 8001


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

На тестовой выборке логистическая регрессия показала чуть более высокий F1 и ROC AUC, поэтому именно она развёрнута в API.


5. Структура проекта
├── data/                 # raw & cleaned datasets
├── notebooks/            # EDA, preprocessing, training
├── telecom/              # FastAPI app
│   ├── main.py
│   └── model_random_telco.pkl
├── requirements.txt
└── README.md

6. Roadmap
 Добавить GridSearchCV для подбора гиперпараметров дерева

 Заменить one-hot вручную на ColumnTransformer в Pipeline

 Докеризация и деплой на AWS EC2

 Frontend-страница для ввода признаков

7. Dataset на Kaggle
* [link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


Автор
Алымбек Ибрагимов
17 y.o. Junior Python/ML Engineer from Kyrgyzstan
Telegram :@ml_engineer_man7 • Email: alymbekibragimov46@gmail.com

