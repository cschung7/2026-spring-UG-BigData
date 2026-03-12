#!/bin/bash
# Start CryptoPulse platform

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== Starting CryptoPulse ==="

# Backend
echo "[1/2] Starting FastAPI backend on port 8000..."
cd "$ROOT/backend"
pip install -r requirements.txt -q
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend
sleep 2

# Frontend
echo "[2/2] Starting Next.js frontend on port 3000..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "=== CryptoPulse Running ==="
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait
