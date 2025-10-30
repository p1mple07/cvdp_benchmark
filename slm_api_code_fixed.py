from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, Dict, Any
import logging
import time
import traceback
from contextlib     def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt to improve generation quality for complex prompts"""
        # Check if prompt requires JSON format
        if '"response":' in prompt or '{ "response":' in prompt:
            # Add clear instruction for JSON format with detailed content
            prompt = f"You must respond in valid JSON format with a comprehensive, detailed technical explanation. {prompt}\n\nProvide a thorough, detailed JSON response with complete technical analysis:"
        
        # Handle technical prompts by adding instruction for comprehensive response
        if len(prompt) > 300 and any(term in prompt.lower() for term in ["barrel shifter", "testing", "circular shift", "verification"]):
            prompt = f"Provide a comprehensive, detailed technical explanation with specific examples, step-by-step analysis, and thorough coverage of all relevant concepts.\n\n{prompt}"
        
        # Add instruction to encourage longer responses for any explanatory content
        if any(term in prompt.lower() for term in ["explain", "describe", "analyze", "discuss", "why", "how"]):
            prompt = prompt + "\n\nProvide a detailed, comprehensive explanation with multiple sentences, specific examples, and thorough technical analysis."
            
        return promptcontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmolLMModel:
    """SmolLM2-1.7B-Instruct model implementation - lightweight SLM for API use"""
    def __init__(self, model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading {model}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    trust_remote_code=True
                ).to(self.device)
            
            logger.info(f"SmolLM2 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load SmolLM2 model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            start_time = time.time()
            
            # Enhanced prompt preprocessing for complex prompts
            processed_prompt = self._preprocess_prompt(prompt)
            
            inputs = self.tokenizer(processed_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            # Ensure we generate longer responses - increased minimum and maximum
            max_new_tokens = min(max(100, max_length - input_length), 2048)  # Increased from 20 and 1024
            
            if max_new_tokens <= 0:
                logger.warning(f"Input too long, no room for generation. Input length: {input_length}, Max length: {max_length}")
                return "Input prompt too long for generation."
            
            # Try multiple generation strategies for complex prompts
            response = self._generate_with_fallback(inputs, max_new_tokens, temperature, top_p)
            
            # Post-process response for JSON format if needed
            response = self._postprocess_response(response, prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"SmolLM generation completed in {generation_time:.2f}s")
            
            return response if response else "Unable to generate response."
        except Exception as e:
            logger.error(f"SmolLM text generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt to improve generation quality for complex prompts"""
        # Check if prompt requires JSON format
        if '"response":' in prompt or '{ "response":' in prompt:
            # Add instruction for JSON format with emphasis on detailed response
            if not prompt.startswith("Generate a detailed JSON response"):
                prompt = f"Generate a detailed JSON response with comprehensive explanation. {prompt}\n\nProvide a thorough, detailed response:"
        
        # Handle very long technical prompts by adding focus instruction
        if len(prompt) > 500 and ("barrel shifter" in prompt.lower() or "testing" in prompt.lower()):
            prompt = f"Provide a comprehensive, detailed technical explanation with specific examples and thorough analysis.\n\n{prompt}"
        
        # Add instruction to encourage longer responses for technical content
        if any(term in prompt.lower() for term in ["explain", "describe", "analyze", "discuss"]):
            prompt = prompt + "\n\nProvide a detailed, comprehensive explanation with examples and thorough analysis."
            
        return prompt

    def _generate_with_fallback(self, inputs, max_new_tokens, temperature, top_p):
        """Generate with multiple fallback strategies"""
        # Strategy 1: Standard generation
        response = self._single_generation(inputs, max_new_tokens, temperature, top_p)
        
        if not response or len(response.strip()) < 10:
            logger.warning("First generation attempt failed, trying with lower temperature")
            # Strategy 2: Lower temperature
            response = self._single_generation(inputs, max_new_tokens, 0.3, 0.8)
            
        if not response or len(response.strip()) < 10:
            logger.warning("Second generation attempt failed, trying with greedy decoding")
            # Strategy 3: Greedy decoding
            response = self._single_generation(inputs, max_new_tokens, 0.0, 1.0)
            
        return response

    def _single_generation(self, inputs, max_new_tokens, temperature, top_p):
        """Single generation attempt"""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    early_stopping=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up common generation artifacts
            response = self._clean_response(response)
            
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Clean up common generation artifacts"""
        # Remove thinking tags that sometimes appear
        response = response.replace("</think>", "").replace("<think>", "")
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
            
        return response.strip()

    def _postprocess_response(self, response: str, original_prompt: str) -> str:
        """Post-process response to ensure proper JSON format when needed"""
        # Check if JSON format was requested
        if '"response":' in original_prompt or '{ "response":' in original_prompt:
            # If response already looks like valid JSON, return as-is
            if response.strip().startswith('{"response":') and response.strip().endswith('}'):
                return response
                
            # If response doesn't look like JSON, wrap it
            if not (response.strip().startswith('{') and response.strip().endswith('}')):
                # Clean the response and wrap in JSON
                clean_response = response.replace('"', '\\"').replace('\n', '\\n').replace('\r', '').replace('\t', ' ')
                
                # Allow much longer responses - increased limits significantly
                if len(clean_response) > 2000:  # Increased from 600
                    # Find a good breaking point (end of sentence)
                    truncate_point = 2000
                    last_period = clean_response.rfind('.', 0, truncate_point)
                    if last_period > 1000:  # Only truncate at period if it's not too early
                        clean_response = clean_response[:last_period + 1]
                    else:
                        clean_response = clean_response[:truncate_point] + "..."
                
                response = f'{{"response": "{clean_response}"}}'
        
        return response

class DeepSeekModel:
    """DeepSeek-R1-Distill-Qwen-7B model implementation for CVDP benchmark"""
    def __init__(self, model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading {model}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Error loading tokenizer for {model}: {e}")
            fallback_model = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
            logger.info(f"Attempting fallback to {fallback_model}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
                self.model_name = fallback_model
                model = fallback_model
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raise RuntimeError(f"Failed to load any DeepSeek model: {e2}")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    trust_remote_code=True
                ).to(self.device)
            
            logger.info(f"DeepSeek model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load DeepSeek model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            start_time = time.time()
            
            # Enhanced prompt preprocessing for complex prompts
            processed_prompt = self._preprocess_prompt(prompt)
            
            inputs = self.tokenizer(processed_prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            # Ensure we generate much longer responses for DeepSeek
            max_new_tokens = min(max(150, max_length - input_length), 3072)  # Increased significantly
            
            if max_new_tokens <= 0:
                logger.warning(f"Input too long, no room for generation. Input length: {input_length}, Max length: {max_length}")
                return "Input prompt too long for generation."
            
            # Try multiple generation strategies for complex prompts
            response = self._generate_with_fallback(inputs, max_new_tokens, temperature, top_p)
            
            # Post-process response for JSON format if needed
            response = self._postprocess_response(response, prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"DeepSeek generation completed in {generation_time:.2f}s")
            
            return response if response else "Unable to generate response."
        except Exception as e:
            logger.error(f"DeepSeek text generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt to improve generation quality for complex prompts"""
        # Check if prompt requires JSON format
        if '"response":' in prompt or '{ "response":' in prompt:
            # Add clear instruction for JSON format
            prompt = f"You must respond in valid JSON format exactly as requested. {prompt}\n\nJSON Response:"
        
        # Handle technical prompts by adding focus instruction
        if len(prompt) > 300 and any(term in prompt.lower() for term in ["barrel shifter", "testing", "circular shift", "verification"]):
            prompt = f"Provide a clear, technical explanation focusing on the key concepts.\n\n{prompt}"
            
        return prompt

    def _generate_with_fallback(self, inputs, max_new_tokens, temperature, top_p):
        """Generate with multiple fallback strategies"""
        # Strategy 1: Standard generation with repetition penalty
        response = self._single_generation(inputs, max_new_tokens, temperature, top_p, 1.2)
        
        if not response or len(response.strip()) < 10:
            logger.warning("First generation attempt failed, trying with lower temperature")
            # Strategy 2: Lower temperature, less repetition penalty
            response = self._single_generation(inputs, max_new_tokens, 0.4, 0.85, 1.1)
            
        if not response or len(response.strip()) < 10:
            logger.warning("Second generation attempt failed, trying with beam search")
            # Strategy 3: Beam search
            response = self._beam_generation(inputs, max_new_tokens)
            
        return response

    def _single_generation(self, inputs, max_new_tokens, temperature, top_p, repetition_penalty=1.1):
        """Single generation attempt with specified parameters"""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    early_stopping=True,
                    length_penalty=1.0
                )
            
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up common generation artifacts
            response = self._clean_response(response)
            
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def _beam_generation(self, inputs, max_new_tokens):
        """Beam search generation for better quality"""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    length_penalty=1.2
                )
            
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            return self._clean_response(response)
        except Exception as e:
            logger.error(f"Beam generation failed: {e}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Clean up common generation artifacts"""
        # Remove thinking tags and artifacts
        response = response.replace("</think>", "").replace("<think>", "")
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        
        # Remove incomplete sentences at the end
        if '. ' in response:
            sentences = response.split('. ')
            if len(sentences) > 1 and len(sentences[-1].strip()) < 15:
                response = '. '.join(sentences[:-1]) + '.'
                
        return response.strip()

    def _postprocess_response(self, response: str, original_prompt: str) -> str:
        """Post-process response to ensure proper JSON format when needed"""
        # Check if JSON format was requested
        if '"response":' in original_prompt or '{ "response":' in original_prompt:
            # If response already looks like valid JSON, return as-is
            if response.strip().startswith('{"response":') and response.strip().endswith('}'):
                return response
                
            # If response doesn't look like JSON, wrap it
            if not (response.strip().startswith('{') and response.strip().endswith('}')):
                # Clean the response and wrap in JSON
                clean_response = response.replace('"', '\\"').replace('\n', '\\n').replace('\r', '').replace('\t', ' ')
                
                # Allow very long responses for DeepSeek - increased limits significantly  
                if len(clean_response) > 3000:  # Increased from 800
                    # Find a good breaking point (end of sentence)
                    truncate_point = 3000
                    last_period = clean_response.rfind('.', 0, truncate_point)
                    if last_period > 1500:  # Only truncate at period if it's not too early
                        clean_response = clean_response[:last_period + 1]
                    else:
                        clean_response = clean_response[:truncate_point] + "..."
                
                response = f'{{"response": "{clean_response}"}}'
        
        return response

# Global model instances
smollm_generator = None
deepseek_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and clean up on shutdown"""
    global smollm_generator, deepseek_generator
    
    logger.info("Starting SLM API server...")
    
    try:
        logger.info("Loading SmolLM model...")
        smollm_generator = SmolLMModel()
        logger.info("SmolLM model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load SmolLM model: {e}")
        smollm_generator = None
    
    try:
        logger.info("Loading DeepSeek model...")
        deepseek_generator = DeepSeekModel()
        logger.info("DeepSeek model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load DeepSeek model: {e}")
        deepseek_generator = None
    
    if smollm_generator is None and deepseek_generator is None:
        logger.error("Failed to load any models!")
        raise RuntimeError("No models could be loaded")
    
    logger.info("SLM API server ready!")
    yield
    
    # Cleanup
    logger.info("Shutting down SLM API server...")

# Initialize the FastAPI app with lifespan
app = FastAPI(
    title="SLM API for CVDP Benchmark",
    description="Small Language Model API server supporting SmolLM2 and DeepSeek models",
    version="1.0.0",
    lifespan=lifespan
)

# Define the request data structure using Pydantic
class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for text generation")
    max_length: Optional[int] = Field(default=None, ge=1, le=4096, description="Maximum length of generated text")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096, description="Maximum tokens to generate (alias for max_length)")
    model: Optional[str] = Field(default="smollm", description="Model to use: 'smollm' or 'deepseek'")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    
    def get_max_length(self) -> int:
        """Get the maximum length, supporting both max_length and max_tokens parameters"""
        if self.max_tokens is not None:
            return self.max_tokens
        elif self.max_length is not None:
            return self.max_length
        else:
            return 1024  # Increased default value for longer responses

class GenerationResponse(BaseModel):
    response: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model used for generation")
    generation_time: Optional[float] = Field(None, description="Time taken for generation in seconds")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    model: Optional[str] = Field(None, description="Model that encountered the error")
    detail: Optional[str] = Field(None, description="Detailed error information")

@app.get("/", response_model=Dict[str, Any])
def read_root():
    """Get API status and available models"""
    models = []
    
    if smollm_generator:
        models.append({
            "name": smollm_generator.model_name,
            "type": "SmolLM2-1.7B-Instruct",
            "device": smollm_generator.device,
            "status": "available"
        })
    
    if deepseek_generator:
        models.append({
            "name": deepseek_generator.model_name,
            "type": "DeepSeek-R1-Distill-Qwen-7B",
            "device": deepseek_generator.device,
            "status": "available"
        })
    
    return {
        "status": "SLM API is running",
        "version": "1.0.0",
        "models": models,
        "endpoints": ["/", "/model_info", "/generate", "/health"]
    }

@app.get("/model_info", response_model=Dict[str, Any])
def get_model_info():
    """Get detailed model information"""
    info = {}
    
    if smollm_generator:
        info["smollm"] = {
            "model_name": smollm_generator.model_name,
            "device": smollm_generator.device,
            "model_type": "SmolLM2-1.7B-Instruct",
            "status": "available"
        }
    else:
        info["smollm"] = {"status": "unavailable", "error": "Model failed to load"}
    
    if deepseek_generator:
        info["deepseek"] = {
            "model_name": deepseek_generator.model_name,
            "device": deepseek_generator.device,
            "model_type": "DeepSeek-R1-Distill-Qwen-7B",
            "status": "available"
        }
    else:
        info["deepseek"] = {"status": "unavailable", "error": "Model failed to load"}
    
    return info

@app.get("/health")
def health_check():
    """Health check endpoint"""
    available_models = []
    if smollm_generator:
        available_models.append("smollm")
    if deepseek_generator:
        available_models.append("deepseek")
    
    return {
        "status": "healthy" if available_models else "unhealthy",
        "available_models": available_models,
        "timestamp": time.time()
    }

@app.post("/generate", response_model=GenerationResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"},
    503: {"model": ErrorResponse, "description": "Service Unavailable"}
})
def generate_text(request: PromptRequest):
    """
    Generate text using the specified model.
    
    - **prompt**: The input text prompt
    - **max_length**: Maximum length of generated response (1-4096)
    - **max_tokens**: Alternative parameter name for max_length (1-4096)
    - **model**: Model to use ('smollm' or 'deepseek')
    - **temperature**: Sampling temperature (0.0-2.0)
    - **top_p**: Top-p sampling parameter (0.0-1.0)
    
    Note: Either max_length or max_tokens can be used (max_tokens takes precedence)
    """
    start_time = time.time()
    
    try:
        # Validate model availability
        if request.model == "deepseek":
            if deepseek_generator is None:
                raise HTTPException(
                    status_code=503,
                    detail="DeepSeek model is not available"
                )
            generator = deepseek_generator
        elif request.model == "smollm":
            if smollm_generator is None:
                raise HTTPException(
                    status_code=503,
                    detail="SmolLM model is not available"
                )
            generator = smollm_generator
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Use 'smollm' or 'deepseek'"
            )
        
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        # Log the request
        max_length_to_use = request.get_max_length()
        logger.info(f"Generation request - Model: {request.model}, Max length: {max_length_to_use}, "
                   f"Temperature: {request.temperature}, Top-p: {request.top_p}")
        logger.debug(f"Prompt preview: {request.prompt[:100]}...")
        
        # Generate text
        generated_text = generator.generate_text(
            prompt=request.prompt,
            max_length=max_length_to_use,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        generation_time = time.time() - start_time
        
        # Count generated tokens (approximate)
        tokens_generated = len(generated_text.split()) if generated_text else 0
        
        logger.info(f"Generation successful - Time: {generation_time:.2f}s, "
                   f"Tokens: {tokens_generated}, Model: {request.model}")
        
        return GenerationResponse(
            response=generated_text,
            model=request.model,
            generation_time=generation_time,
            tokens_generated=tokens_generated
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log detailed error information
        logger.error(f"Generation failed for model {request.model}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return structured error response
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )

# Add a simple test endpoint for debugging
@app.post("/test")
def test_generation():
    """Simple test endpoint for debugging"""
    try:
        test_request = PromptRequest(
            prompt="Hello, how are you?",
            max_length=50,
            model="smollm"
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

@app.post("/test_json")
def test_json_generation():
    """Test endpoint for JSON format responses"""
    try:
        test_request = PromptRequest(
            prompt='Answer this question in JSON format: { "response": "your answer" } Why is testing circular shifts important?',
            max_length=200,
            model="deepseek",
            temperature=0.4
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"JSON test failed: {str(e)}"}

@app.post("/test_complex")
def test_complex_prompt():
    """Test endpoint for complex technical prompts"""
    try:
        complex_prompt = '''You are solving a 'Question & Answer on Testbench' problem. Provide the response in JSON format: { "response": "<response>" }

Question: Explain in four sentences why testing circular shifts with shift_bits = DATA_WIDTH is critical for ensuring the barrel shifter correctly handles edge cases without introducing unintended behavior or corrupting data integrity.'''
        
        test_request = PromptRequest(
            prompt=complex_prompt,
            max_length=800,  # Increased from 300
            model="deepseek",
            temperature=0.3
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Complex test failed: {str(e)}"}

@app.post("/test_long")
def test_long_response():
    """Test endpoint for very long detailed responses"""
    try:
        long_prompt = '''Provide a comprehensive, detailed technical explanation in JSON format: { "response": "<detailed_response>" }

Question: Explain comprehensively why testing circular shifts with shift_bits = DATA_WIDTH is critical for barrel shifter validation. Include specific examples, edge cases, technical details about hardware implementation, potential failure modes, and verification strategies.'''
        
        test_request = PromptRequest(
            prompt=long_prompt,
            max_length=1500,  # Much larger
            model="deepseek",
            temperature=0.4
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Long response test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    logger.info("Starting SLM API server...")
    uvicorn.run(
        "slm_api_code:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )