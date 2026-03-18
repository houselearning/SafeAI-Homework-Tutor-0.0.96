# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY safeai_core ./safeai_core
COPY data ./data
COPY models ./models
CMD ["uvicorn", "safeai_core.inference:app", "--host", "0.0.0.0", "--port", "8000"]
