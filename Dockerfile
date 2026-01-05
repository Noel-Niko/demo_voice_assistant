ARG DOCKER_REGISTRY=graingerinc.jfrog.io/docker-shared-virtual
FROM ${DOCKER_REGISTRY}/aad-mlops-base-python:3.11-slim-bullseye AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1

# Build stage - install dependencies
FROM base AS builder

# Minimal tools for builder
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Use the same path as runtime stage
WORKDIR /home/appuser
COPY uv.lock pyproject.toml README.md ./

# Runtime stage
FROM base AS runtime

# Create user first (to ensure /home/appuser exists)
RUN useradd --create-home appuser

# Copy uv binary from builder stage
COPY --from=builder /root/.local/bin/uv /usr/local/bin/uv

# Copy application sources and lock/metadata
WORKDIR /home/appuser
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
COPY --chown=appuser:appuser src ./src

# Install runtime dependencies directly in runtime to avoid layer duplication
# Use non-root user to ensure .venv lives under /home/appuser
USER appuser
ENV PATH="/home/appuser/.venv/bin:/home/appuser/.local/bin:/usr/local/bin:$PATH"

# Sync dependencies (prod only) and prune caches to shrink layer size
RUN uv sync --frozen --no-dev \
    && rm -rf ~/.cache/pip ~/.cache/uv 2>/dev/null || true


ENV LOG_CONFIG="/home/appuser/logging.conf"
ENV PATH="/home/appuser/.venv/bin:/home/appuser/.local/bin:/usr/local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:6008/ping')"

# Run the application (bind to IPv6 to satisfy IPv6 kubelet probes)
CMD ["uvicorn", "src.gateway.routes:app", "--host", "::", "--port", "6008"]
