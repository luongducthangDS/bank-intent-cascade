FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cascade/ ./cascade/
COPY artifacts/ ./artifacts/
COPY reports/results.json ./reports/results.json

EXPOSE 8000

CMD ["uvicorn", "cascade.service:app", "--host", "0.0.0.0", "--port", "8000"]
