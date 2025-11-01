#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build script for iterative refinement agent

echo "Building iterative refinement agent Docker image..."
docker build -t iterative-agent --no-cache .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Agent image: iterative-agent"
    echo ""
    echo "To run the agent:"
    echo "  python run_benchmark.py -f dataset.jsonl -l -g iterative-agent"
    echo ""
    echo "To configure the SLM model, set environment variables in .env:"
    echo "  SLM_API_URL=http://localhost:8000"
    echo "  SLM_MODEL=deepseek"
    echo "  SLM_MAX_LENGTH=8192"
else
    echo "❌ Build failed!"
    exit 1
fi
