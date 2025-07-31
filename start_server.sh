#!/bin/bash

echo "🚀 Starting WinningFC API Server..."

# Activate virtual environment
source venv/bin/activate

# Start the server
python -m src.api.main &

# Get the PID
SERVER_PID=$!
echo "✅ Server started with PID: $SERVER_PID"
echo "🌐 API available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "To stop the server, run: kill $SERVER_PID"
echo "Or press Ctrl+C to stop"

# Wait for server to start
sleep 3

# Test the server
echo "🧪 Testing server..."
curl -s http://localhost:8000/health | jq '.' 2>/dev/null || echo "Server is starting up..."

# Keep script running
wait $SERVER_PID