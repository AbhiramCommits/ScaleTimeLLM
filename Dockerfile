FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SCALETIMELLM_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE Makefile ./
COPY scaletimellm ./scaletimellm
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -e .

# The default action is the complete CPU-only smoke sweep on the mock backend.
CMD ["python", "-m", "scaletimellm.cli", "smoke", "--output-dir", "/app/results/smoke"]
