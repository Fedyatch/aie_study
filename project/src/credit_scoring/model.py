import sys
import numpy as np
import pandas as pd
import joblib

def log1p_transform(X):
    return np.log1p(X)

def add_total_past_due(X):
    X = X.copy()
    past_due_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate"
    ]
    available = [c for c in past_due_cols if c in X.columns]
    if available:
        X["TotalPastDue"] = X[available].sum(axis=1)
    return X

_main = sys.modules.get("__main__")
if _main is not None:
    _main.log1p_transform = log1p_transform
    _main.add_total_past_due = add_total_past_due

class ScoringModel:
    def __init__(self, model_path: str = "artifacts/modeling/models/scorer_pipeline_v1.0.joblib"):
        self.pipeline = joblib.load(model_path)
        # Оригинальные имена, которые ожидает пайплайн (с дефисами)
        self.original_features = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents"
        ]
        # Соответствие между ключами из API и оригинальными именами
        self.rename_map = {
            "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate": "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines": "NumberRealEstateLoansOrLines",
            "NumberOfOpenCreditLinesAndLoans": "NumberOfOpenCreditLinesAndLoans",
            "NumberOfDependents": "NumberOfDependents",
        }

    def predict(self, data: dict) -> dict:
        # Преобразуем ключи с подчёркиваниями в оригинальные с дефисами
        converted = {}
        for k, v in data.items():
            new_k = self.rename_map.get(k, k)  # если нет в словаре, оставляем как есть
            converted[new_k] = v
        df = pd.DataFrame([converted])[self.original_features]
        proba = self.pipeline.predict_proba(df)[0, 1]
        risk = "high" if proba > 0.5 else "low"
        return {
            "probability_default": round(float(proba), 4),
            "risk_category": risk,
            "version": "0.2.0"
        }