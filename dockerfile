FROM python:3.10-slim

WORKDIR /service

COPY service_requirements.txt .
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install -r service_requirements.txt

COPY model.pkl .
COPY discountuplift.csv .
COPY service/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]