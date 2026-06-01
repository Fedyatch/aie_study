from pydantic import BaseModel, Field

class CreditApplication(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., description="Общий баланс по кредитным картам / сумма кредитных лимитов")
    age: int = Field(..., ge=18, le=120, description="Возраст заёмщика")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, description="Количество просрочек 30-59 дней за 2 года")
    DebtRatio: float = Field(..., ge=0, description="Отношение ежемесячных долгов к доходу")
    MonthlyIncome: float = Field(..., ge=0, description="Ежемесячный доход")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, description="Количество открытых кредитов и займов")
    NumberOfTimes90DaysLate: int = Field(..., ge=0, description="Количество просрочек 90+ дней за 2 года")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, description="Количество ипотечных кредитов")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, description="Количество просрочек 60-89 дней за 2 года")
    NumberOfDependents: int = Field(..., ge=0, description="Количество иждивенцев")

class CreditResponse(BaseModel):
    probability_default: float = Field(..., description="Вероятность дефолта (0..1)")
    risk_category: str = Field(..., description="Категория риска: high/low")
    version: str = "0.2.0"
    latency_ms: float = Field(..., description="Время обработки запроса в миллисекундах")