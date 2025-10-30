#!/usr/bin/env python3
"""
Test script for GPT OSS integration in SLM API
"""

import requests
import json

def test_gptoss_api():
    """Test GPT OSS API integration"""
    
    # Test basic health check
    try:
        health_response = requests.get("http://localhost:8000/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ API Health Check:")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Available models: {len(health_data.get('models', []))}")
            
            # Check if GPT OSS is listed
            models = health_data.get('models', [])
            gptoss_found = any(model.get('type') == 'GPT-OSS' for model in models)
            if gptoss_found:
                print("✅ GPT OSS model found in available models")
            else:
                print("❌ GPT OSS model not found in available models")
                print("   Available models:", [m.get('type') for m in models])
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test GPT OSS generation
    try:
        test_prompt = "Write a simple 'Hello World' program in Python."
        
        payload = {
            "prompt": test_prompt,
            "model": "gptoss",
            "max_length": 200,
            "temperature": 0.7
        }
        
        print(f"\n🧪 Testing GPT OSS generation...")
        print(f"   Prompt: {test_prompt}")
        
        response = requests.post(
            "http://localhost:8000/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ GPT OSS generation successful!")
            print(f"   Model used: {result.get('model')}")
            print(f"   Generation time: {result.get('generation_time', 'N/A')}s")
            print(f"   Response preview: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation test error: {e}")
        return False

def test_gptoss_endpoint():
    """Test the dedicated GPT OSS endpoint"""
    
    try:
        payload = {
            "prompt": "Explain what GPT stands for in one sentence.",
            "max_length": 100,
            "temperature": 0.5
        }
        
        print(f"\n🧪 Testing dedicated GPT OSS endpoint...")
        
        response = requests.post(
            "http://localhost:8000/gptoss/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ GPT OSS dedicated endpoint successful!")
            print(f"   Response: {result.get('response', '')}")
            return True
        else:
            print(f"❌ Dedicated endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Dedicated endpoint test error: {e}")
        return False

def main():
    """Run all GPT OSS tests"""
    
    print("🚀 Testing GPT OSS Integration in SLM API")
    print("=" * 50)
    
    # Test basic API functionality
    api_test = test_gptoss_api()
    
    # Test dedicated endpoint
    endpoint_test = test_gptoss_endpoint()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Basic API: {'✅ PASS' if api_test else '❌ FAIL'}")
    print(f"   Dedicated Endpoint: {'✅ PASS' if endpoint_test else '❌ FAIL'}")
    
    if api_test and endpoint_test:
        print("\n🎉 All GPT OSS tests passed!")
        print("   GPT OSS is ready to use in benchmarks with model='gptoss-slm'")
    else:
        print("\n❌ Some tests failed. Check the SLM API server status.")
        print("   Make sure the server is running and GPT OSS model loaded successfully.")

if __name__ == "__main__":
    main()