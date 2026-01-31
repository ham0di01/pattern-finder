#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Pattern Trader Application...${NC}"

# --- Backend Setup ---
echo -e "${GREEN}[Backend] Setting up...${NC}"
cd backend

if [ ! -d "venv" ]; then
    echo -e "${BLUE}[Backend] Creating virtual environment...${NC}"
    python3 -m venv venv
fi

echo -e "${BLUE}[Backend] Installing dependencies...${NC}"
# Check if requirements.txt exists, if not create a minimal one for now or install packages directly
if [ -f "requirements.txt" ]; then
    ./venv/bin/pip install -r requirements.txt
else
    echo -e "${BLUE}[Backend] requirements.txt not found. Installing manually...${NC}"
    ./venv/bin/pip install fastapi uvicorn sqlalchemy pandas numpy opencv-python-headless ccxt pydantic-settings python-multipart python-dotenv matplotlib
fi

echo -e "${GREEN}[Backend] Starting Server on port 8000...${NC}"
./venv/bin/uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

cd ..

# --- Frontend Setup ---
echo -e "${GREEN}[Frontend] Setting up...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}[Frontend] Installing node modules...${NC}"
    npm install
fi

echo -e "${GREEN}[Frontend] Starting Development Server on port 3000...${NC}"
npm run dev &
FRONTEND_PID=$!

cd ..

# --- Cleanup Handler ---
cleanup() {
    echo -e "\n${BLUE}Shutting down processes...${NC}"
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

trap cleanup SIGINT

echo -e "${GREEN}Application is running!${NC}"
echo -e "Backend:  http://localhost:8000/api/v1/docs"
echo -e "Frontend: http://localhost:3000"
echo -e "${BLUE}Press Ctrl+C to stop.${NC}"

wait
