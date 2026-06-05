FROM ghcr.io/astral-sh/uv:python3.11-alpine

# Install supervisor process manager
RUN apk add --no-cache supervisor bash

WORKDIR /app

# Copy dependency files and install them via uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy the rest of your application code
COPY . .

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Start supervisor to run both services together
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]