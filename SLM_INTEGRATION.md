# SLM API Integration for CVDP Benchmark

This directory contains a custom model implementation that allows the CVDP Benchmark to integrate with Small Language Models (SLMs) via HTTP API calls.

## Features

- **HTTP API Integration**: Makes POST requests to your SLM API endpoint
- **Flexible Response Parsing**: Handles various JSON response formats and plain text
- **Retry Logic**: Built-in retry mechanism for robust API calls
- **Full CVDP Integration**: Works with all benchmark features including sampling, evaluation, and reporting
- **Configurable**: Support for different models, timeouts, and API endpoints

## API Requirements

Your SLM API should accept POST requests with this format:

```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"<your_prompt>","max_length":100, "model":"smollm"}'
```

**Expected Response Format**:

```json
{"response": "Your model response here", "model": "smollm"}
```

**Error Response Format**:
```json
{"error": "Error description", "model": "smollm"}
```

## Setup Instructions

### 1. Configure Environment Variables

Edit your `.env` file or set environment variables:

```bash
# Required: Your SLM API endpoint
SLM_API_URL=http://localhost:8000

# Optional: Additional configuration
SLM_MAX_LENGTH=2048           # Maximum response length
SLM_TIMEOUT=60               # API timeout in seconds
SLM_DEFAULT_MODEL=deepseek   # Default model name
CUSTOM_MODEL_FACTORY=custom_slm_factory.py  # Path to custom factory
```

### 2. Test the Integration

Run the test script to verify everything works:

```bash
# Test with default settings (localhost:8000, deepseek model)
python test_slm_integration.py

# Test with custom API URL and model
python test_slm_integration.py --api-url http://your-server:8000 --model llama

# Skip connection test if API is not running yet
python test_slm_integration.py --skip-connection-test
```

### 3. Run the Benchmark

#### Option A: Using --custom-factory flag

```bash
python run_benchmark.py \
  -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl \
  -l -m deepseek-slm \
  --custom-factory custom_slm_factory.py
```

#### Option B: Using environment variable

```bash
export CUSTOM_MODEL_FACTORY=custom_slm_factory.py
python run_benchmark.py \
  -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl \
  -l -m deepseek-slm
```

#### Option C: Multi-sample evaluation

```bash
python run_samples.py \
  -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl \
  -l -m deepseek-slm -n 5 -k 1 \
  --custom-factory custom_slm_factory.py
```

## Supported Model Names

The factory recognizes these patterns:

- `smollm-slm` → Uses "smollm" model (SmolLM2-1.7B-Instruct) via SLM API
- `deepseek-slm` → Uses "deepseek" model (DeepSeek-R1-Distill-Qwen-7B) via SLM API  
- `custom-slm` → Uses "custom" model via SLM API
- `slm` → Uses default model (smollm) via SLM API

You can easily add more patterns by editing `custom_slm_factory.py`.

## File Structure

```
cvdp_benchmark/
├── custom_slm_factory.py           # Custom model factory
├── test_slm_integration.py         # Test script
├── src/llm_lib/slm_api_model.py   # SLM API model implementation
└── SLM_INTEGRATION.md             # This documentation
```

## Customization

### Adding New Model Types

Edit `custom_slm_factory.py` to add more model patterns:

```python
# Add this line in CustomModelFactory.__init__()
self.model_types["your-model-name"] = self._create_slm_instance
```

### Modifying API Request Format

Edit `slm_api_model.py` in the `prompt()` method to change the API request format:

```python
payload = {
    "prompt": full_prompt,
    "max_length": self.max_length,
    "model": self.model,
    # Add your custom parameters here
    "temperature": 0.7,
    "top_p": 0.9
}
```

### Adding Authentication

If your API requires authentication, modify the headers in `slm_api_model.py`:

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {your_api_key}",
    # Add other headers as needed
}
```

## Troubleshooting

### Connection Issues

1. **"Failed to connect to SLM API"**
   - Ensure your SLM API is running
   - Check the URL and port
   - Verify firewall settings

2. **"API call failed with status XXX"**
   - Check your API's error response format
   - Verify the request format matches your API's expectations
   - Check API logs for detailed error information

### Response Parsing Issues

1. **"Failed to parse JSON response"**
   - Check if your API returns valid JSON
   - Verify the response field names match the expected patterns
   - Enable debug mode for detailed logging

2. **Empty or incorrect responses**
   - Verify your API's response format
   - Check if response extraction logic needs customization
   - Enable debug mode to see raw API responses

### Debugging

Enable debug mode for detailed logging:

```python
# In your test script or when creating the model
model.set_debug(True)
```

This will show:
- Full API request details
- Response status codes and headers
- Response parsing steps
- Error details

## Performance Considerations

- **Timeout Settings**: Adjust `SLM_TIMEOUT` based on your model's response time
- **Concurrent Requests**: The framework handles parallelization automatically
- **Retry Logic**: Built-in retry for failed requests with exponential backoff
- **Connection Pooling**: Uses `requests` library which handles connection reuse

## Integration with CVDP Features

This integration supports all CVDP Benchmark features:

- ✅ **Single Problem Evaluation**: `run_benchmark.py`
- ✅ **Multi-Sample Evaluation**: `run_samples.py` 
- ✅ **Statistical Analysis**: Pass@k metrics
- ✅ **Problem Categories**: All 16 problem categories
- ✅ **Schema-based Responses**: JSON format requirements
- ✅ **File Generation**: Single and multi-file outputs
- ✅ **Prompt Logging**: Full prompt/response logging
- ✅ **Error Reporting**: Detailed error analysis
- ✅ **Docker Integration**: Works with verification harnesses

The integration follows the same interface as OpenAI models, ensuring seamless compatibility with all benchmark features.