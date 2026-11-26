FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 ecovision && useradd --system --uid 10001 --gid 10001 --home-dir /home/ecovision --create-home --shell /usr/sbin/nologin ecovision
ENV HOME=/home/ecovision KERAS_HOME=/tmp/keras TMPDIR=/tmp
COPY requirements-docker.txt ./
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements-docker.txt
COPY app.py ./
COPY backend ./backend
COPY database ./database
COPY migrations ./migrations
COPY visualizacion ./visualizacion
RUN mkdir -p /app/modelo
COPY modelo/waste_classifier.keras ./modelo/waste_classifier.keras
COPY modelo/classes.txt ./modelo/classes.txt
RUN chmod 0444 /app/modelo/waste_classifier.keras /app/modelo/classes.txt
USER 10001:10001
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
