#!/bin/bash
# Quick Demo - Just the essentials

echo "🏈 FieldOS Engine - Quick Demo"
echo ""

# Setup
echo "Setting up..."
source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
pip install -e . -q

# Load data
echo "Loading demo data..."
python scripts/seed_demo_data.py

# Start server
echo "Starting server..."
pkill -f "uvicorn fieldos_engine" 2>/dev/null || true
nohup uvicorn fieldos_engine.api.main:app --port 8000 > server.log 2>&1 &
sleep 3

# Test it works
echo ""
echo "Testing server..."
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "Running sample simulation..."
curl -s -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  --data '{
    "play_id": "play_trips_flood",
    "scenario_ids": ["scenario_zone_cover2"],
    "num_episodes": 10,
    "seed": 42,
    "mode": "EVAL",
    "trace_policy": {"mode": "NONE"}
  }' | python3 -m json.tool | head -30

echo ""
echo "✅ Server running at http://localhost:8000"
echo "📚 API docs at http://localhost:8000/docs"
echo ""
echo "To stop: pkill -f 'uvicorn fieldos_engine'"
