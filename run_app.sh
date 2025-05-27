#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up trap to catch Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM

# Start backend server
echo "Starting backend server..."
./run_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# Start frontend server
echo "Starting frontend server..."
./run_frontend.sh &
FRONTEND_PID=$!

# Wait for both processes
echo "BTC Price Predictor is running!"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo "- WebSocket: ws://localhost:8765"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for any process to exit
wait $BACKEND_PID $FRONTEND_PID

# Exit with the same code as the process that exited
exit $?