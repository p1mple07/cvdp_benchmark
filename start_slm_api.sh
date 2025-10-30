#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to start the SLM API server and test integration

echo "==========================="
echo "SLM API Server Setup Script"
echo "==========================="

# Check if required dependencies are installed
echo "Checking dependencies..."

# Check if FastAPI is installed
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "FastAPI not found. Installing..."
    pip install fastapi uvicorn
fi

# Check if transformers is installed  
python3 -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Transformers not found. Installing..."
    pip install transformers torch accelerate
fi

echo "Dependencies check complete."
echo

# Option 1: Start server in background
if [ "$1" == "start" ]; then
    echo "Starting SLM API server in background..."
    echo "This will take a few minutes to load the models..."
    
    # Start the API server
    nohup uvicorn slm_api_code:app --host 0.0.0.0 --port 8000 > slm_api.log 2>&1 &
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    echo "Logs are being written to slm_api.log"
    echo
    echo "Waiting for server to start..."
    
    # Wait for server to be ready (check every 5 seconds for up to 2 minutes)
    for i in {1..24}; do
        sleep 5
        curl -s http://localhost:8000/ > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Server is ready!"
            break
        fi
        echo "  Still loading models... ($((i*5))s elapsed)"
        if [ $i -eq 24 ]; then
            echo "✗ Server took too long to start. Check slm_api.log for errors."
            exit 1
        fi
    done

# Option 2: Test existing server
elif [ "$1" == "test" ]; then
    echo "Testing existing SLM API server..."
    
# Option 3: Stop server
elif [ "$1" == "stop" ]; then
    echo "Stopping SLM API server..."
    pkill -f "uvicorn slm_api_code:app"
    echo "Server stopped."
    exit 0
    
# Default: Show usage
else
    echo "Usage:"
    echo "  ./start_slm_api.sh start  - Start the API server in background"
    echo "  ./start_slm_api.sh test   - Test integration with existing server"
    echo "  ./start_slm_api.sh stop   - Stop the API server"
    echo
    echo "Manual start:"
    echo "  uvicorn slm_api_code:app --host 0.0.0.0 --port 8000"
    echo
    exit 0
fi

echo
echo "==========================="
echo "Testing SLM Integration"
echo "==========================="

# Test the API directly first
echo "1. Testing API endpoints..."

# Test basic API
echo "Testing GET / ..."
curl -s http://localhost:8000/ | python3 -m json.tool
echo

echo "Testing GET /model_info ..."
curl -s http://localhost:8000/model_info | python3 -m json.tool  
echo

# Test text generation
echo "Testing POST /generate with SmolLM..."
curl -s -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Hello, how are you?","max_length":50, "model":"smollm"}' | python3 -m json.tool
echo

echo "Testing POST /generate with DeepSeek..."
curl -s -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Write a simple Python function to add two numbers","max_length":100, "model":"deepseek"}' | python3 -m json.tool
echo

# Test our integration
echo "2. Testing CVDP Integration..."
python3 test_slm_integration.py --model smollm

echo
echo "==========================="
echo "Integration Test Complete"
echo "==========================="

echo
echo "Next steps:"
echo "1. Run a simple benchmark:"
echo "   python run_benchmark.py -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl -l -m smollm-slm --custom-factory custom_slm_factory.py"
echo
echo "2. Run multi-sample evaluation:"
echo "   python run_samples.py -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl -l -m deepseek-slm -n 3 -k 1 --custom-factory custom_slm_factory.py"
echo