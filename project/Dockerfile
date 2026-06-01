FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (весь каталог project/)
COPY . .

# Говорим Python, где искать модули (src/ будет корнем)
ENV PYTHONPATH=/app/src

# Создаём непривилегированного пользователя для безопасности
RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8000

# Healthcheck с использованием встроенного Python (без curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Запуск сервиса (модуль credit_scoping.api:app)
CMD ["uvicorn", "credit_scoring.api:app", "--host", "0.0.0.0", "--port", "8000"]