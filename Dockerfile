# AI_agent/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем 
COPY . .

# точка входа
CMD ["python", "agent.py"]