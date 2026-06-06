# Using debian-slim instead of alpine to completely bypass glibc/musl mismatch issues
FROM python:3.11-slim

# Install supervisor and clean up apt caches to keep the image lightweight
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install uv cleanly from their official binary distribution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files and sync them using uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy the rest of your application code
COPY . .

# Copy supervisor configuration to its default standard Debian location
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Start supervisor using the standard Debian path
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]