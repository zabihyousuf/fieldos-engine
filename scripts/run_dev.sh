#!/bin/bash
# Development server startup script

echo "Starting FieldOS Engine development server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load demo data
echo "Loading demo data..."
python scripts/seed_demo_data.py

# Start server
echo "Starting FastAPI server on port 8000..."
uvicorn fieldos_engine.api.main:app --reload --port 8000
