import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

telco_app = FastAPI(title='Predict Avocado')


class TelcoSchema(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: str
    Contract: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str



BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / 'model_random_telco.pkl'
scaler_path = BASE_DIR / 'scaler_telco.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


@telco_app.post('/predict/')
async def predict_avocado(telco: TelcoSchema):
    telco_dict = telco.model_dump()
    nums = [
        telco_dict['tenure'],
        telco_dict['MonthlyCharges'],
        telco_dict['TotalCharges'],
    ]
    Contract_own = telco_dict.pop('Contract')
    Contract_1_or_0 = [
        1 if Contract_own == 'Two year' else 0,
        1 if Contract_own == 'One year' else 0,
    ]
    InternetService_own = telco_dict.pop('InternetService')
    InternetService_1_or_0 = [
        1 if InternetService_own == 'Fiber optic' else 0,
        1 if InternetService_own == 'No' else 0,
    ]
    OnlineSecurity_own = telco_dict.pop('OnlineSecurity')
    OnlineSecurity_1_or_0 = [
        1 if OnlineSecurity_own == 'Yes' else 0,
        1 if OnlineSecurity_own == 'No internet service' else 0,
    ]
    TechSupport_own = telco_dict.pop('TechSupport')
    TechSupport_1_or_0 = [
        1 if TechSupport_own == 'No internet service' else 0,
        1 if TechSupport_own == 'Yes' else 0,
    ]

    features = (nums + Contract_1_or_0 + InternetService_1_or_0 + OnlineSecurity_1_or_0 + TechSupport_1_or_0)

    scaled_features = scaler.transform([features])
    pred = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]

    return {'approved': bool(pred), 'prob': round(prob,2)}



if __name__ == '__main__':
    uvicorn.run(telco_app, host='127.0.0.1', port=8001)





