from fastapi.testclient import TestClient
from src.credit_scoring.api import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict_valid():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.15,
        "age": 35,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 3,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "probability_default" in data
    assert 0 <= data["probability_default"] <= 1
    assert data["risk_category"] in ("high", "low")