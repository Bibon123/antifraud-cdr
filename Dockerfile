# Антифрод CDR: пайплайн + веб-консоль.
#   docker build -t antifraud .
#   docker run --rm -p 8000:8000 -v "$PWD/data:/data" antifraud            # ждёт /data/cdr_out.zip, база — SQLite внутри контейнера
#   docker compose up --build                                               # то же с PostgreSQL (см. docker-compose.yml)
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY antifraud ./antifraud
COPY app ./app
COPY tests ./tests
COPY config.toml pytest.ini run_all.py ./
EXPOSE 8000
# Прогон по данным из /data, затем сервис на 0.0.0.0:8000
CMD ["sh", "-c", "python -m antifraud run ${SOURCE:-/data/cdr_out.zip} --out /app/out && python -m antifraud evaluate --out /app/out && python -m antifraud serve --out /app/out --host 0.0.0.0 --port 8000"]
