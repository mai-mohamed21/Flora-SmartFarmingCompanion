FROM python:3.9-slim

WORKDIR /app

COPY Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Backend/ .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
