#!/bin/bash

# Enhanced SLM API Server Restart Script
# This script helps restart the API server with the enhanced code

echo "🔄 Restarting Enhanced SLM API Server..."

# Kill existing server
echo "⏹️  Stopping existing server..."
pkill -f "python.*slm_api_code" || pkill -f "uvicorn.*slm_api_code" || echo "No existing server found"

# Wait a moment for cleanup
sleep 2

# Start the enhanced server
echo "🚀 Starting enhanced SLM API server..."
echo "📍 Starting server with complex prompt handling improvements:"
echo "   ✅ Enhanced JSON format handling"
echo "   ✅ Multiple generation fallback strategies"
echo "   ✅ Better prompt preprocessing"
echo "   ✅ Improved response cleaning"
echo ""

# Start server in background
nohup python slm_api_code.py > slm_api.log 2>&1 &

echo "⏳ Waiting for server to start..."
sleep 5

# Test server
echo "🧪 Testing server connectivity..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Server is running successfully!"
    echo "📊 Available endpoints:"
    echo "   • GET  /health          - Health check"
    echo "   • POST /generate        - Standard generation"  
    echo "   • POST /test_json       - Test JSON format handling"
    echo "   • POST /test_complex    - Test complex technical prompts"
    echo ""
    echo "🎯 Enhanced features:"
    echo "   • Better handling of JSON format requirements"
    echo "   • Multiple fallback generation strategies"
    echo "   • Improved technical prompt processing"
    echo "   • Automatic response cleaning and formatting"
else
    echo "❌ Server failed to start. Check slm_api.log for errors."
    tail -20 slm_api.log
fi

