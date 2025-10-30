# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom model factory implementation with SLM API support.

This factory extends the base ModelFactory to add support for custom SLM (Small Language Model) APIs
that accept JSON requests via HTTP POST.

To use this factory:
1. Set environment variables or update .env file with your SLM API configuration:
   SLM_API_URL=http://localhost:8000  # Your API base URL
   SLM_MAX_LENGTH=20480               # Maximum response length - 10x enhanced
   SLM_TIMEOUT=60                     # API timeout in seconds

2. Run benchmark with the --custom-factory flag:
   python run_benchmark.py -f input.jsonl -l -m deepseek-slm --custom-factory /path/to/this/custom_slm_factory.py

3. Or export the environment variable:
   export CUSTOM_MODEL_FACTORY=/path/to/this/custom_slm_factory.py
   python run_benchmark.py -f input.jsonl -l -m deepseek-slm
"""

import logging
import os
import sys
from typing import Optional, Any

# Add the src directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import the base ModelFactory
from src.llm_lib.model_factory import ModelFactory
from src.config_manager import config

# Import our custom SLM API implementation
from src.llm_lib.slm_api_model import SLM_API_Instance

class CustomModelFactory(ModelFactory):
    """
    Custom model factory that extends the base ModelFactory to add support for SLM API models.
    """
    
    def __init__(self):
        # Initialize the base factory first
        super().__init__()
        
        # Register SLM API model types
        # These will handle models with patterns like:
        # - "deepseek-slm" -> uses deepseek model via SLM API
        # - "phi35-slm" -> uses phi35 model via SLM API
        # - "gptoss-slm" -> uses gptoss model via SLM API
        # - "nemotron-slm" -> uses nemotron model via SLM API
        # - "jamba-slm" -> uses jamba model via SLM API
        # - "llama-slm" -> uses llama model via SLM API  
        # - "custom-slm" -> uses custom model via SLM API
        # - "slm" -> uses default model via SLM API
        
        self.model_types["slm"] = self._create_slm_instance
        self.model_types["smollm-slm"] = self._create_slm_instance
        self.model_types["deepseek-slm"] = self._create_slm_instance
        self.model_types["phi35-slm"] = self._create_slm_instance
        self.model_types["gptoss-slm"] = self._create_slm_instance
        self.model_types["nemotron-slm"] = self._create_slm_instance
        self.model_types["jamba-slm"] = self._create_slm_instance
        self.model_types["llama-slm"] = self._create_slm_instance
        self.model_types["custom-slm"] = self._create_slm_instance
        
        # You can add more specific model names as needed
        # self.model_types["your-model-name"] = self._create_slm_instance
        
        logging.info("Custom SLM model factory initialized with SLM API support")

    def _create_slm_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> SLM_API_Instance:
        """
        Create an SLM API model instance.
        
        Args:
            model_name: Name of the model (e.g., "deepseek-slm", "llama-slm")
            context: Context to pass to the model constructor
            key: API key (may not be used for local APIs)
            **kwargs: Additional arguments
            
        Returns:
            SLM_API_Instance: Configured SLM API model instance
        """
        # Extract the actual model name from the pattern
        # e.g., "deepseek-slm" -> "deepseek", "llama-slm" -> "llama"
        if model_name.endswith("-slm"):
            actual_model = model_name[:-4]  # Remove "-slm" suffix
        elif model_name == "slm":
            actual_model = config.get("SLM_DEFAULT_MODEL", "smollm")
        else:
            actual_model = model_name
        
        # Get API URL from kwargs or config
        api_url = kwargs.get('api_url') or config.get("SLM_API_URL")
        
        return SLM_API_Instance(
            context=context, 
            key=key, 
            model=actual_model,
            api_url=api_url
        )


# Example of how to use the custom factory directly (for testing)
if __name__ == "__main__":
    # Create instance of the custom factory
    factory = CustomModelFactory()
    
    # Test creating SLM models with the factory
    try:
        print("Testing SLM API model creation...")
        
        # Test with deepseek-slm model
        slm_model = factory.create_model(
            model_name="deepseek-slm", 
            context="You are a helpful coding assistant."
        )
        print(f"Successfully created SLM model: {slm_model.model}")
        
        # Test API connection
        if slm_model.test_connection():
            print("API connection test passed!")
            
            # Test with a simple prompt
            try:
                response = slm_model.prompt(
                    prompt="Write a simple hello world program in Python", 
                    schema=None,
                    prompt_log="", 
                    files=["hello.py"],
                    timeout=30,
                    category=2  # RTL Code Completion category as example
                )
                print(f"Model response received: {response}")
            except Exception as e:
                print(f"Error during prompt test: {e}")
        else:
            print("API connection test failed - make sure your SLM API is running")
        
    except Exception as e:
        print(f"Error creating SLM model: {e}")
        print("Make sure your SLM API is running at the configured URL")