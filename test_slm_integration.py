#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for SLM API integration with CVDP Benchmark.

This script tests the custom SLM API model implementation to ensure it works correctly
with the benchmark framework.

Usage:
    python test_slm_integration.py [--api-url http://localhost:8000] [--model deepseek]
"""

import sys
import os
import argparse
import json

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the custom factory and model
from custom_slm_factory import CustomModelFactory
from src.llm_lib.slm_api_model import SLM_API_Instance

def test_direct_model_creation(api_url: str, model_name: str):
    """Test creating SLM model directly."""
    print("=" * 50)
    print("Testing direct SLM model creation...")
    print("=" * 50)
    
    try:
        # Create model directly
        model = SLM_API_Instance(
            context="You are a helpful assistant for hardware verification.",
            model=model_name,
            api_url=api_url
        )
        
        print(f"✓ Successfully created SLM model: {model.model}")
        print(f"✓ API URL: {model.api_url}")
        
        # Test connection
        if model.test_connection():
            print("✓ API connection test passed!")
            return model
        else:
            print("✗ API connection test failed")
            return None
            
    except Exception as e:
        print(f"✗ Error creating SLM model: {e}")
        return None

def test_factory_model_creation(api_url: str, model_name: str):
    """Test creating SLM model via factory."""
    print("=" * 50)
    print("Testing factory-based SLM model creation...")
    print("=" * 50)
    
    try:
        # Set environment variable for API URL
        os.environ["SLM_API_URL"] = api_url
        
        # Create factory and model
        factory = CustomModelFactory()
        model = factory.create_model(
            model_name=f"{model_name}-slm",
            context="You are a helpful assistant for hardware verification."
        )
        
        print(f"✓ Successfully created SLM model via factory: {model.model}")
        print(f"✓ API URL: {model.api_url}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error creating SLM model via factory: {e}")
        return None

def test_model_prompting(model, test_cases):
    """Test model prompting with various scenarios."""
    print("=" * 50)
    print("Testing model prompting...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = model.prompt(
                prompt=test_case['prompt'],
                schema=test_case.get('schema'),
                files=test_case.get('files'),
                category=test_case.get('category'),
                timeout=90  # Increased timeout for slower models
            )
            
            print(f"✓ Response received:")
            if isinstance(response, tuple) and len(response) == 2:
                content, success = response
                print(f"  Success: {success}")
                if isinstance(content, dict):
                    print(f"  Content keys: {list(content.keys())}")
                    if 'direct_text' in content:
                        preview = content['direct_text'][:100]
                        print(f"  Preview: {preview}...")
                    elif 'code' in content:
                        print(f"  Code entries: {len(content['code']) if isinstance(content['code'], list) else 'N/A'}")
                else:
                    print(f"  Content: {str(content)[:100]}...")
            else:
                print(f"  Response: {str(response)[:100]}...")
                
        except Exception as e:
            print(f"✗ Error in test case: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test SLM API integration")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="SLM API base URL (default: http://localhost:8000)")
    parser.add_argument("--model", default="deepseek",
                       help="Model name to use (default: deepseek)")
    parser.add_argument("--skip-connection-test", action="store_true",
                       help="Skip API connection test")
    
    args = parser.parse_args()
    
    print("SLM API Integration Test")
    print("=" * 50)
    print(f"API URL: {args.api_url}")
    print(f"Model: {args.model}")
    print("=" * 50)
    
    # Test cases for different scenarios
    test_cases = [
        {
            "name": "Simple text generation",
            "prompt": "Write a simple hello world program in Python",
            "files": ["hello.py"],
            "category": 2
        },
        {
            "name": "Question answering (no files)",
            "prompt": "What is the difference between blocking and non-blocking assignments in Verilog?",
            "category": 9
        },
        {
            "name": "Multiple file generation", 
            "prompt": "Create a basic RTL counter module with testbench",
            "files": ["counter.v", "testbench.v"],
            "category": 3
        },
        {
            "name": "JSON schema response",
            "prompt": "Generate a simple AND gate module in Verilog",
            "schema": '{ "code": [{ "<filename>" : "<code>"}] }',
            "files": ["and_gate.v"],
            "category": 2
        }
    ]
    
    success = True
    
    # Test 1: Direct model creation
    model = test_direct_model_creation(args.api_url, args.model)
    if model is None:
        success = False
        if not args.skip_connection_test:
            print("\n⚠️  Skipping further tests due to connection failure.")
            print("   Make sure your SLM API is running and accessible.")
            return 1
    
    # Test 2: Factory-based model creation
    factory_model = test_factory_model_creation(args.api_url, args.model)
    if factory_model is None:
        success = False
    else:
        model = factory_model  # Use factory model for subsequent tests
    
    # Test 3: Model prompting (if we have a working model)
    if model is not None:
        test_model_prompting(model, test_cases)
    
    # Summary
    print("=" * 50)
    if success:
        print("✓ All tests completed successfully!")
        print("\nNext steps:")
        print("1. Run the benchmark with your custom factory:")
        print(f"   python run_benchmark.py -f example_dataset/your_dataset.jsonl -l -m {args.model}-slm --custom-factory custom_slm_factory.py")
        print("\n2. Or set the environment variable and run:")
        print("   export CUSTOM_MODEL_FACTORY=custom_slm_factory.py")
        print(f"   python run_benchmark.py -f example_dataset/your_dataset.jsonl -l -m {args.model}-slm")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())