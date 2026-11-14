#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=================================================="
echo "   Starting FairLens AI Bias & Fairness Auditor   "
echo "=================================================="

# Activate virtualenv
source venv/bin/activate

# Start backend in background
echo "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..."
uvicorn app.main:app --app-dir backend --port 8000 &
BACKEND_PID=$!

# Trap cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null || true; exit" SIGINT SIGTERM EXIT

# Start frontend
echo "[2/2] Starting Vite Frontend on http://localhost:3000..."
cd frontend
npm run dev -- --host

wait
