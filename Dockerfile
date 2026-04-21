FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/train.py --num-users 200 --num-items 1000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
