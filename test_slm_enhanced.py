#!/usr/bin/env python3
"""
Enhanced test script for SLM API integration with better error handling and timeout management.
"""

import sys
import os
import argparse
import json
import time

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the custom factory and model
from custom_slm_factory import CustomModelFactory
from src.llm_lib.slm_api_model import SLM_API_Instance

def test_api_connection(api_url: str, model_name: str):
    """Test basic API connectivity"""
    print("=" * 50)
    print("Testing API connectivity...")
    print("=" * 50)
    
    import requests
    
    # Test endpoints
    endpoints = [
        ("/health", "Health check"),
        ("/model_info", "Model info"),
        ("/", "Root endpoint")
    ]
    
    for endpoint, description in endpoints:
        try:
            print(f"Testing {description} ({endpoint})...")
            response = requests.get(f"{api_url}{endpoint}", timeout=10)
            print(f"  ✓ Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if 'available_models' in data:
                    print(f"  ✓ Available models: {data['available_models']}")
                elif 'models' in data:
                    print(f"  ✓ Models: {[m['name'] for m in data['models']]}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    return True

def test_model_creation_and_setup(api_url: str, model_name: str):
    """Test model creation with proper configuration"""
    print("=" * 50)
    print("Testing model creation and configuration...")
    print("=" * 50)
    
    try:
        # Set environment variables for better configuration
        os.environ["SLM_API_URL"] = api_url
        os.environ["SLM_TIMEOUT"] = "120"  # 2 minute timeout
        os.environ["SLM_MAX_LENGTH"] = "20480"  # 10x enhanced
        
        # Create factory and model
        factory = CustomModelFactory()
        model = factory.create_model(
            model_name=f"{model_name}-slm",
            context="You are a helpful assistant for hardware design and verification."
        )
        
        # Enable debug mode for detailed logging
        model.set_debug(True)
        
        print(f"✓ Model created successfully")
        print(f"✓ Model name: {model.model}")
        print(f"✓ API URL: {model.api_url}")
        print(f"✓ Timeout: {model.timeout}s")
        print(f"✓ Max length: {model.max_length}")
        
        return model
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def test_progressive_prompts(model):
    """Test with progressively more complex prompts"""
    print("=" * 50)
    print("Testing with progressive complexity...")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Ultra Simple",
            "prompt": "Hello",
            "files": None,
            "category": 9,  # Changed from None to valid category
            "expected_time": 30
        },
        {
            "name": "Simple Question",
            "prompt": "What is 2+2?",
            "files": None,
            "category": 9,  # Changed from None to valid category
            "expected_time": 45
        },
        {
            "name": "Code Request (Simple)",
            "prompt": "Write print('hello')",
            "files": ["hello.py"],
            "category": 2,
            "expected_time": 60
        },
        {
            "name": "Hardware Question",
            "prompt": "What is a flip-flop in digital circuits?",
            "files": None,
            "category": 9,
            "expected_time": 90
        },
        {
            "name": "Verilog Code Generation",
            "prompt": "Create a simple 2-input AND gate in Verilog",
            "files": ["and_gate.v"],
            "category": 2,
            "expected_time": 120
        }
    ]
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        start_time = time.time()
        try:
            print(f"Prompt: {test_case['prompt']}")
            print(f"Expected time: {test_case['expected_time']}s")
            print("Generating response...")
            
            response = model.prompt(
                prompt=test_case['prompt'],
                schema=None,
                files=test_case['files'],
                category=test_case['category'],
                timeout=test_case['expected_time'] + 30  # Add buffer
            )
            
            elapsed = time.time() - start_time
            
            if isinstance(response, tuple) and len(response) == 2:
                content, success = response
                results[test_case['name']] = {
                    'success': success,
                    'time': elapsed,
                    'content_type': type(content).__name__,
                    'has_content': bool(content)
                }
                
                print(f"✓ Success: {success}")
                print(f"✓ Time: {elapsed:.2f}s")
                print(f"✓ Content type: {type(content).__name__}")
                
                if isinstance(content, dict):
                    print(f"✓ Content keys: {list(content.keys())}")
                    if 'direct_text' in content and content['direct_text']:
                        preview = content['direct_text'][:100].replace('\n', '\\n')
                        print(f"✓ Preview: {preview}...")
                    elif 'code' in content:
                        print(f"✓ Code entries: {len(content['code']) if isinstance(content['code'], list) else 'N/A'}")
                elif content:
                    preview = str(content)[:100].replace('\n', '\\n')
                    print(f"✓ Preview: {preview}...")
                else:
                    print("⚠ Empty content")
            else:
                results[test_case['name']] = {
                    'success': False,
                    'time': elapsed,
                    'error': 'Unexpected response format'
                }
                print(f"⚠ Unexpected response format: {type(response)}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            results[test_case['name']] = {
                'success': False,
                'time': elapsed,
                'error': str(e)
            }
            print(f"✗ Failed after {elapsed:.2f}s: {e}")
        
        # Add delay between tests to avoid overwhelming the API
        if i < len(test_cases):
            print("Waiting 2s before next test...")
            time.sleep(2)
    
    return results

def print_summary(results):
    """Print test summary"""
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✓" if result.get('success', False) else "✗"
        time_taken = result.get('time', 0)
        print(f"  {status} {test_name}: {time_taken:.2f}s")
        if not result.get('success', False) and 'error' in result:
            print(f"    Error: {result['error']}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced SLM API Integration Test")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="SLM API base URL")
    parser.add_argument("--model", default="smollm",
                       help="Model name (smollm or deepseek)")
    parser.add_argument("--skip-api-test", action="store_true",
                       help="Skip API connectivity test")
    
    args = parser.parse_args()
    
    print("Enhanced SLM API Integration Test")
    print("=" * 50)
    print(f"API URL: {args.api_url}")
    print(f"Model: {args.model}")
    print("=" * 50)
    
    # Step 1: Test API connectivity
    if not args.skip_api_test:
        if not test_api_connection(args.api_url, args.model):
            print("\n⚠️ API connectivity issues detected. Continuing anyway...")
    
    # Step 2: Test model creation
    model = test_model_creation_and_setup(args.api_url, args.model)
    if model is None:
        print("\n✗ Cannot proceed without a working model")
        return 1
    
    # Step 3: Progressive testing
    results = test_progressive_prompts(model)
    
    # Step 4: Summary
    print_summary(results)
    
    # Step 5: Recommendations
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    if successful_tests > 0:
        print("\n" + "=" * 50)
        print("RECOMMENDATIONS")
        print("=" * 50)
        print("✓ Integration is working! Next steps:")
        print(f"  1. Use model '{args.model}-slm' in CVDP benchmark")
        print("  2. Consider using smollm for faster responses")
        print("  3. Adjust SLM_TIMEOUT based on your model's performance")
        print(f"  4. Example command:")
        print(f"     export CUSTOM_MODEL_FACTORY=custom_slm_factory.py")
        print(f"     python run_benchmark.py -f example_dataset/your_file.jsonl -l -m {args.model}-slm")
        
        if args.model == "deepseek":
            print("\n⚠️ Note: DeepSeek model may be slower. Consider:")
            print("  - Using smollm for faster testing")
            print("  - Increasing timeout values")
            print("  - Using GPU acceleration if available")
    else:
        print("\n✗ Integration has issues. Check:")
        print("  - Is your SLM API server running?")
        print("  - Are the models loaded correctly?")
        print("  - Try the debug_api.py script for more details")
    
    return 0 if successful_tests > 0 else 1

if __name__ == "__main__":
    sys.exit(main())