# ========================
# Base (build dependencies)
# ========================
FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/prod.txt /app/requirements/
RUN pip install --no-cache-dir -r /app/requirements/prod.txt

# ========================
# Dev image
# ========================
FROM base AS dev
COPY requirements/dev.txt /app/requirements/
RUN pip install --no-cache-dir -r /app/requirements/dev.txt
CMD ["bash"]

# ========================
# Production image
# ========================
FROM python:3.12-slim AS prod

WORKDIR /app

COPY --from=base /usr/local /usr/local
COPY src ./src
COPY artifacts ./artifacts
COPY configs ./configs

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
