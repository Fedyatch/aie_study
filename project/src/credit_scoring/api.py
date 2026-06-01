from __future__ import annotations

from time import perf_counter
from fastapi import FastAPI
from .model import ScoringModel
from .schemas import CreditApplication, CreditResponse

app = FastAPI(
    title="Credit Scoring Service",
    version="0.2.0",
    description="Сервис оценки кредитного риска (Give Me Some Credit).",
    docs_url="/docs",
    redoc_url=None,
)

model = ScoringModel()
print("Credit Scoring сервис инициализирован")

@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "credit-scoring",
        "version": "0.2.0",
    }

@app.post("/predict", response_model=CreditResponse, tags=["scoring"])
def predict(application: CreditApplication) -> CreditResponse:
    start = perf_counter()
    result = model.predict(application.model_dump())
    latency_ms = round((perf_counter() - start) * 1000, 2)
    print(f"[predict] probability={result['probability_default']} risk={result['risk_category']} latency={latency_ms}ms")
    return CreditResponse(**result, latency_ms=latency_ms)