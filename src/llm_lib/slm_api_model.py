# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import requests
import json
import logging
import time
from typing import Optional, Any, Dict, List
from src.config_manager import config
from src.model_helpers import ModelHelpers

logging.basicConfig(level=logging.INFO)

RETRY_CODES = [429, 502, 503, 504]
WAIT_TIME = 1.5  # Seconds to wait between API calls for error resilience

class SLM_API_Instance:
    """
    Custom SLM (Small Language Model) API model instance that makes HTTP calls to a local API endpoint.
    
    This model is designed to work with APIs that accept JSON requests with prompt, max_length, and model parameters.
    """

    def __init__(self, context: str = "You are a helpful assistant.", key: str = None, model: str = "smollm", api_url: str = None):
        """
        Initialize the SLM API model instance.
        
        Args:
            context: System context/prompt for the model
            key: API key (maintained for interface compatibility, not used for local APIs)
            model: Model name to use (e.g., "deepseek")
            api_url: Base URL for the API (defaults to http://localhost:8000)
        """
        self.context = context
        self.model = model
        self.debug = False
        
        # Set API URL
        if api_url is None:
            self.api_url = config.get("SLM_API_URL", "http://localhost:8000")
        else:
            self.api_url = api_url
        
        # Ensure URL doesn't end with slash for consistent endpoint building
        self.api_url = self.api_url.rstrip('/')
        
        # Default parameters - optimized for different models
        self.max_length = config.get("SLM_MAX_LENGTH", 4096)  # Balanced: was 8192, now 4096 for better speed
        self.timeout = config.get("SLM_TIMEOUT", 300)  # Increased timeout for very long responses: 120→300s
        
        # Initialize model helpers
        self.helper = ModelHelpers()
        
        logging.info(f"Created SLM API Model. API URL: {self.api_url}, Model: {self.model}")

    def set_debug(self, debug: bool = True) -> None:
        """
        Enable or disable debug mode.
        
        Args:
            debug: Whether to enable debug mode (default: True)
        """
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")

    @property
    def requires_evaluation(self) -> bool:
        """
        Whether this model requires harness evaluation.
        
        Returns:
            bool: True (standard models require evaluation)
        """
        return True

    def key(self, key: str):
        """Set API key (for interface compatibility - may not be used by local APIs)."""
        self.api_key = key

    def prompt(self, prompt: str, schema: str = None, prompt_log: str = "", 
               files: Optional[List] = None, timeout: int = 60, category: Optional[int] = None) -> str:
        """
        Send a prompt to the SLM API and get a response.
        
        Args:
            prompt: The user prompt/query
            schema: Optional JSON schema for structured output
            prompt_log: Path to log the prompt (if not empty)
            files: List of expected output files (if any)
            timeout: Timeout in seconds for the API call (default: 60)
            category: Optional integer indicating the category/problem ID
            
        Returns:
            The parsed model response
        """
        # Import and use ModelHelpers
        system_prompt = self.helper.create_system_prompt(self.context, schema, category)
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Use timeout from config if not specified or use default
        if timeout == 60:
            timeout = self.timeout
        
        # For slow models, ensure minimum timeout
        timeout = max(timeout, 30)

        # Determine if we're expecting a single file (direct text mode)
        expected_single_file = files and len(files) == 1 and schema is None
        expected_file_name = files[0] if expected_single_file else None

        if self.debug:
            logging.debug(f"Requesting prompt using SLM API: {self.api_url}")
            logging.debug(f"Model: {self.model}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {prompt}")
            if files:
                logging.debug(f"Expected files: {files}")
                if expected_single_file:
                    logging.debug(f"Using direct text mode for single file: {expected_file_name}")
            logging.debug(f"Request parameters: model={self.model}, timeout={timeout}, max_length={self.max_length}")

        # Create directories for prompt log if needed
        if prompt_log:
            try:
                import os
                # Ensure directory exists
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                
                # Write to a temporary file first
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(system_prompt + "\n\n----------------------------------------\n" + prompt)
                
                # Atomic rename to final file
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log to {prompt_log}: {str(e)}")
                # Don't continue if we can't write the log file
                raise

        # Prepare the API request
        endpoint = f"{self.api_url}/generate"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": full_prompt,
            "max_length": self.max_length,
            "model": self.model
        }

        # Add retry logic for resilience
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.debug:
                    logging.debug(f"Making API call (attempt {attempt + 1}/{max_retries})")
                    logging.debug(f"Endpoint: {endpoint}")
                    logging.debug(f"Headers: {headers}")
                    logging.debug(f"Payload: {json.dumps(payload, indent=2)}")

                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

                if self.debug:
                    logging.debug(f"Response status code: {response.status_code}")
                    logging.debug(f"Response headers: {dict(response.headers)}")
                    logging.debug(f"Response content length: {len(response.content)} bytes")
                    logging.debug(f"Response content type: {response.headers.get('content-type', 'unknown')}")
                    logging.debug(f"Raw response content: {response.content[:500]}")
                    logging.debug(f"Raw response text: {repr(response.text[:500])}")                # Handle successful response
                if response.status_code == 200:
                    try:
                        # Try to parse as JSON first
                        response_data = response.json()
                        if self.debug:
                            logging.debug(f"Parsed JSON response: {response_data}")
                        
                        # Extract text from various possible JSON structures
                        content = self._extract_content_from_response(response_data)
                        
                    except json.JSONDecodeError as json_err:
                        # If not JSON, treat as plain text
                        content = response.text.strip()
                        if self.debug:
                            logging.debug(f"Plain text response: {content}")
                        
                        # If still empty, log the raw response for debugging
                        if not content:
                            logging.warning(f"Empty response received. Status: {response.status_code}, "
                                          f"Headers: {dict(response.headers)}, "
                                          f"Raw content length: {len(response.content)}")
                    
                    # Handle empty responses
                    if not content or content.strip() == "":
                        logging.warning("Received empty response from API")
                        if files and len(files) > 0:
                            content = f"// Generated content for {files[0] if len(files) == 1 else 'files'}\n// The model returned an empty response"
                        else:
                            content = "The model did not provide a response to your query."

                    # Process the response using helper functions
                    # The ModelHelpers.parse_model_response expects either:
                    # 1. Single file + no_schema = True -> direct text
                    # 2. Schema requested -> JSON format
                    # 3. Multiple files -> JSON format
                    
                    if expected_single_file:
                        # Single file expected - use direct text mode
                        return self.helper.parse_model_response(content, files, True)
                    elif files is None or len(files) == 0:
                        # No files expected - create a dummy single file scenario for direct text
                        dummy_files = ["response.txt"]
                        return self.helper.parse_model_response(content, dummy_files, True)
                    elif schema is not None:
                        # Schema requested - content should be JSON formatted
                        if content.startswith('{') and content.endswith('}'):
                            # Fix common JSON formatting issues
                            content = self.helper.fix_json_formatting(content)
                        return self.helper.parse_model_response(content, files, False)
                    else:
                        # Multiple files but no schema - wrap content in JSON format
                        json_content = json.dumps({"response": content})
                        return self.helper.parse_model_response(json_content, files, False)

                # Handle retryable errors
                elif response.status_code in RETRY_CODES:
                    if attempt < max_retries - 1:
                        wait_time = WAIT_TIME * (2 ** attempt)  # Exponential backoff
                        logging.warning(f"API call failed with retryable status {response.status_code}, retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise requests.exceptions.RequestException(
                            f"API call failed after {max_retries} attempts. Status: {response.status_code}, Response: {response.text}"
                        )
                
                # Handle other HTTP errors
                else:
                    error_message = f"API call failed with status {response.status_code}: {response.text}"
                    logging.error(error_message)
                    raise requests.exceptions.RequestException(error_message)

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = WAIT_TIME * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"API call timed out (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"API call timed out after {max_retries} attempts with {timeout}s timeout")
                    raise ValueError(f"SLM API call timed out after {max_retries} attempts. Try increasing timeout or using a faster model.")
            
            except requests.exceptions.ConnectionError as e:
                error_message = f"Failed to connect to SLM API at {endpoint}: {str(e)}"
                logging.error(error_message)
                raise ValueError(error_message)
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = WAIT_TIME * (2 ** attempt)
                    logging.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"Final attempt failed: {str(e)}")
                    raise ValueError(f"Unable to get response from SLM API after {max_retries} attempts: {str(e)}")

    def _extract_content_from_response(self, response_data: Dict) -> str:
        """
        Extract content from various possible API response formats.
        
        Args:
            response_data: Parsed JSON response from the API
            
        Returns:
            The extracted text content
        """
        # Try common response field names
        possible_fields = [
            'generated_text',    # Common field name
            'text',             # Simple field name
            'response',         # Generic field name
            'completion',       # OpenAI-style
            'output',           # Generic output field
            'result',           # Generic result field
            'content'           # Generic content field
        ]
        
        for field in possible_fields:
            if field in response_data:
                content = response_data[field]
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, dict):
                    # Sometimes the content might be nested
                    return str(content)
        
        # If no known field found, try to convert the whole response to string
        if isinstance(response_data, dict) and len(response_data) == 1:
            # If there's only one field, use its value
            key, value = next(iter(response_data.items()))
            if isinstance(value, str):
                return value.strip()
        
        # Last resort: convert entire response to string
        return str(response_data)

    def test_connection(self) -> bool:
        """
        Test if the SLM API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            test_payload = {
                "prompt": "Hello, world!",
                "max_length": 10,
                "model": self.model
            }
            
            response = requests.post(
                f"{self.api_url}/generate",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logging.info("SLM API connection test successful")
                return True
            else:
                logging.warning(f"SLM API connection test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"SLM API connection test failed: {str(e)}")
            return False