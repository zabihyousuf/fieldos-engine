FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY fieldos_engine/ ./fieldos_engine/
COPY scripts/ ./scripts/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Load demo data and start server
CMD python scripts/seed_demo_data.py && \
    uvicorn fieldos_engine.api.main:app --host 0.0.0.0 --port 8000
