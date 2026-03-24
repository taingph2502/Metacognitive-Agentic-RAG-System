FROM python:3.13-slim

WORKDIR /app

# Avoid large caches
ENV UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv
RUN pip install --no-cache-dir uv

# Install CPU-only PyTorch BEFORE resolving dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-cache

# Copy source code
COPY app ./app
COPY tests ./tests

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]