FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY app.py ./
COPY backend ./backend
COPY database ./database
COPY visualizacion ./visualizacion
RUN mkdir -p /app/modelo
COPY modelo/waste_classifier.keras ./modelo/waste_classifier.keras
COPY modelo/classes.txt ./modelo/classes.txt
RUN chmod 0444 /app/modelo/waste_classifier.keras /app/modelo/classes.txt
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
