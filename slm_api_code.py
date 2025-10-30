from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, Dict, Any
import logging
import time
import traceback
from contextlib import asynccontextmanager

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
            
            inputs = self.tokenizer(processed_prompt, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(self.device)  # 4x increase: was 2048, now 8192
            input_length = inputs['input_ids'].shape[1]
            
            # Ensure we generate much longer responses - 10x enhancement
            max_new_tokens = min(max(200, max_length - input_length), 20480)  # 10x increased: was 2048, now 20480
            
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
        """Preprocess prompt to improve generation quality with 10x enhancement for complex prompts"""
        # Check if prompt requires JSON format - enhanced instruction
        if '"response":' in prompt or '{ "response":' in prompt:
            # Add comprehensive instruction for JSON format with detailed response
            if not prompt.startswith("Generate a detailed JSON response"):
                prompt = f"Generate a comprehensive, detailed JSON response with thorough technical explanation, multiple examples, and complete analysis. Provide extensive detail in your response. {prompt}\n\nProvide a very thorough, comprehensive, detailed response with extensive technical analysis:"
        
        # Handle technical prompts by adding comprehensive instruction - enhanced
        if len(prompt) > 300 and any(term in prompt.lower() for term in ["barrel shifter", "testing", "circular shift", "verification", "lfsr", "hardware"]):
            prompt = f"Provide a very comprehensive, detailed technical explanation with multiple specific examples, step-by-step analysis, thorough coverage of all concepts, and extensive technical details.\n\n{prompt}\n\nProvide extensive detail and comprehensive analysis."
        
        # Enhanced instruction to encourage much longer responses for technical content
        if any(term in prompt.lower() for term in ["explain", "describe", "analyze", "discuss", "why", "how"]):
            prompt = prompt + "\n\nProvide a very detailed, comprehensive, thorough explanation with multiple sentences, specific examples, extensive technical analysis, and complete coverage of all relevant aspects."
            
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
        """Single generation attempt with Phi-3.5 compatibility"""
        try:
            with torch.no_grad():
                # Use legacy cache for Phi-3.5 compatibility
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
                    use_cache=True,
                    # Remove early_stopping for compatibility
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
                
                # Allow much longer responses - 10x enhancement
                if len(clean_response) > 20000:  # 10x increased: was 2000, now 20000
                    # Find a good breaking point (end of sentence)
                    truncate_point = 20000
                    last_period = clean_response.rfind('.', 0, truncate_point)
                    if last_period > 10000:  # Only truncate at period if it's not too early - 10x increased
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
            
            # Enhanced prompt preprocessing with 10x improvements for DeepSeek
            processed_prompt = self._preprocess_prompt(prompt)
            
            inputs = self.tokenizer(processed_prompt, return_tensors="pt", padding=True, truncation=True, max_length=16384).to(self.device)  # 4x increase: was 4096, now 16384
            input_length = inputs['input_ids'].shape[1]
            
            # Ensure we generate much longer responses for DeepSeek - 10x enhancement
            max_new_tokens = min(max(300, max_length - input_length), 30720)  # 10x increased: was 3072, now 30720
            
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
        """Preprocess prompt with 10x enhancement to improve generation quality for complex prompts"""
        # Check if prompt requires JSON format - enhanced instruction
        if '"response":' in prompt or '{ "response":' in prompt:
            # Add comprehensive instruction for JSON format with extensive detail
            prompt = f"You must respond in valid JSON format exactly as requested with comprehensive, detailed, thorough technical explanation. Provide extensive detail and complete analysis. {prompt}\n\nProvide comprehensive JSON Response with extensive technical detail:"
        
        # Handle technical prompts by adding comprehensive instruction - enhanced
        if len(prompt) > 200 and any(term in prompt.lower() for term in ["barrel shifter", "testing", "circular shift", "verification", "lfsr", "hardware", "explain", "describe"]):
            prompt = f"Provide a comprehensive, detailed technical explanation with extensive coverage, multiple examples, step-by-step analysis, and thorough examination of all concepts.\n\n{prompt}\n\nProvide very detailed, comprehensive technical analysis."
            
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
        """Beam search generation with Phi-3.5 compatibility"""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    length_penalty=1.2,
                    use_cache=True,
                    # Remove early_stopping for compatibility
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
                
                # Allow very long responses for DeepSeek - 10x enhancement
                if len(clean_response) > 30000:  # 10x increased: was 3000, now 30000
                    # Find a good breaking point (end of sentence)
                    truncate_point = 30000
                    last_period = clean_response.rfind('.', 0, truncate_point)
                    if last_period > 15000:  # Only truncate at period if it's not too early - 10x increased
                        clean_response = clean_response[:last_period + 1]
                    else:
                        clean_response = clean_response[:truncate_point] + "..."
                
                response = f'{{"response": "{clean_response}"}}'
        
        return response

class Phi35MiniModel:
    """Microsoft Phi-3-mini-4k-instruct model implementation for efficient SLM API use"""
    def __init__(self, model: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading {model}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="eager"  # Force eager attention to avoid flash_attn dependency
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    trust_remote_code=True
                ).to(self.device)
            
            logger.info(f"Phi-3.5-mini model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Phi-3.5-mini model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            start_time = time.time()
            
            # Enhanced prompt preprocessing for Phi-3.5 format
            processed_prompt = self._preprocess_prompt(prompt)
            
            inputs = self.tokenizer(processed_prompt, return_tensors="pt", padding=True, truncation=True, max_length=3584).to(self.device)  # Reduced to fit within Phi-3.5's 4k context
            input_length = inputs['input_ids'].shape[1]
            
            # Phi-3.5 optimized for 4k context - balanced enhancement
            max_new_tokens = min(max(100, max_length - input_length), 2048)  # Reduced from 20480 to 2048 for speed
            
            if max_new_tokens <= 0:
                logger.warning(f"Input too long, no room for generation. Input length: {input_length}, Max length: {max_length}")
                return "Input prompt too long for generation."
            
            # Try multiple generation strategies for complex prompts
            response = self._generate_with_fallback(inputs, max_new_tokens, temperature, top_p)
            
            # Post-process response for JSON format if needed
            response = self._postprocess_response(response, prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"Phi-3.5 generation completed in {generation_time:.2f}s")
            
            return response if response else "Unable to generate response."
        except Exception as e:
            logger.error(f"Phi-3.5 text generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt to improve generation quality for Phi-3.5 format"""
        # Phi-3.5 uses a specific chat template format
        if not prompt.startswith("<|user|>"):
            # Check if prompt requires JSON format
            if '"response":' in prompt or '{ "response":' in prompt:
                # Format for JSON response with Phi-3.5 chat template
                formatted_prompt = f"<|user|>\nGenerate a JSON response as requested. {prompt}<|end|>\n<|assistant|>\n"
            else:
                # Standard chat format
                formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            
            return formatted_prompt
        
        return prompt

    def _generate_with_fallback(self, inputs, max_new_tokens, temperature, top_p):
        """Generate with multiple fallback strategies for Phi-3.5"""
        # Strategy 1: Standard generation with good parameters for Phi-3.5
        response = self._single_generation(inputs, max_new_tokens, temperature, top_p, 1.1)
        
        if not response or len(response.strip()) < 10:
            logger.warning("First generation attempt failed, trying with lower temperature")
            # Strategy 2: Lower temperature for more focused output
            response = self._single_generation(inputs, max_new_tokens, 0.4, 0.85, 1.05)
            
        if not response or len(response.strip()) < 10:
            logger.warning("Second generation attempt failed, trying with beam search")
            # Strategy 3: Beam search for quality
            response = self._beam_generation(inputs, max_new_tokens)
            
        return response

    def _single_generation(self, inputs, max_new_tokens, temperature, top_p, repetition_penalty=1.1):
        """Single generation attempt optimized for Phi-3.5"""
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
                    use_cache=True,  # Explicitly enable cache
                    length_penalty=1.0
                )
            
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up common generation artifacts
            response = self._clean_response(response)
            
            return response
        except Exception as e:
            logger.error(f"Phi-3.5 generation failed: {e}")
            return ""

    def _beam_generation(self, inputs, max_new_tokens):
        """Beam search generation for Phi-3.5"""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    length_penalty=1.2,
                    repetition_penalty=1.05,
                    use_cache=True  # Explicitly enable cache
                )
            
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            return self._clean_response(response)
        except Exception as e:
            logger.error(f"Phi-3.5 beam generation failed: {e}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Clean up common generation artifacts for Phi-3.5"""
        # Remove special tokens and artifacts
        response = response.replace("<|end|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
        response = response.replace("</think>", "").replace("<think>", "")
        
        # Remove incomplete sentences at the end
        if '. ' in response:
            sentences = response.split('. ')
            if len(sentences) > 1 and len(sentences[-1].strip()) < 15:
                response = '. '.join(sentences[:-1]) + '.'
                
        return response.strip()

    def _postprocess_response(self, response: str, original_prompt: str) -> str:
        """Post-process response to ensure proper JSON format when needed for Phi-3.5"""
        # Check if JSON format was requested
        if '"response":' in original_prompt or '{ "response":' in original_prompt:
            # If response already looks like valid JSON, return as-is
            if response.strip().startswith('{"response":') and response.strip().endswith('}'):
                return response
                
            # If response doesn't look like JSON, wrap it
            if not (response.strip().startswith('{') and response.strip().endswith('}')):
                # Clean the response and wrap in JSON
                clean_response = response.replace('"', '\\"').replace('\n', '\\n').replace('\r', '').replace('\t', ' ')
                
                # Truncate very long responses to prevent JSON issues - optimized for speed
                if len(clean_response) > 2000:  # Reduced from 8000 to 2000 for faster generation
                    # Find a good breaking point (end of sentence)
                    truncate_point = 2000
                    last_period = clean_response.rfind('.', 0, truncate_point)
                    if last_period > 1000:  # Only truncate at period if it's not too early
                        clean_response = clean_response[:last_period + 1]
                    else:
                        clean_response = clean_response[:truncate_point] + "..."
                
                response = f'{{"response": "{clean_response}"}}'
        
        return response

class GPTOSSModel:
    """GPT OSS model implementation - open source GPT model for API use"""
    def __init__(self, model: str = "microsoft/DialoGPT-medium"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading GPT OSS model: {model}...")
        
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
            
            logger.info(f"GPT OSS model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load GPT OSS model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            start_time = time.time()
            
            # Enhanced prompt preprocessing for complex prompts
            if len(prompt) > 3584:  # Context limit for GPT OSS
                logger.warning(f"Input prompt ({len(prompt)} chars) exceeds context limit, truncating...")
                # Keep the last part of the prompt which is usually more important
                prompt = "..." + prompt[-3584:]
            
            # Apply chat template formatting for better responses
            formatted_prompt = f"### User: {prompt}\n### Assistant:"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3584,  # Context window limit
                padding=False
            ).to(self.device)
            
            # Enhanced generation parameters for GPT OSS
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_length, 2048),  # Limit output length
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    use_cache=True
                )
            
            # Decode and clean the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the input prompt)
            response = full_response[len(formatted_prompt):].strip()
            
            generation_time = time.time() - start_time
            logger.info(f"GPT OSS generation completed in {generation_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in GPT OSS generation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"

class NemotronModel:
    """NVIDIA Nemotron model implementation for API use"""
    def __init__(self, model: str = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading Nemotron model: {model}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
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
            
            logger.info(f"Nemotron model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Nemotron model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            start_time = time.time()
            
            # Enhanced prompt preprocessing for Nemotron
            processed_prompt = self._preprocess_prompt(prompt)
            
            # Nemotron supports large context windows
            inputs = self.tokenizer(
                processed_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192,  # Nemotron supports large context
                padding=False
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            max_new_tokens = min(max(200, max_length - input_length), 4096)
            
            if max_new_tokens <= 0:
                logger.warning(f"Input too long, no room for generation. Input length: {input_length}")
                return "Input prompt too long for generation."
            
            # Generate with Nemotron-optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    use_cache=True
                )
            
            # Decode and extract only the generated part
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if processed_prompt in full_response:
                response = full_response[len(processed_prompt):].strip()
            else:
                # Fallback: just decode the new tokens
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Post-process for JSON format if needed
            response = self._postprocess_response(response, prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"Nemotron generation completed in {generation_time:.2f}s")
            
            return response if response else "Unable to generate response."
            
        except Exception as e:
            logger.error(f"Error in Nemotron generation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt for Nemotron format"""
        # Nemotron uses Llama-style chat format
        if not prompt.startswith("<|user|>") and not prompt.startswith("[INST]"):
            # Check if JSON format is requested
            if '"response":' in prompt or '{ "response":' in prompt:
                formatted_prompt = f"<|user|>\n{prompt}\nProvide your response in the requested JSON format.<|end|>\n<|assistant|>\n"
            else:
                formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            return formatted_prompt
        return prompt

    def _clean_response(self, response: str) -> str:
        """Clean up Nemotron response artifacts"""
        # Remove special tokens
        response = response.replace("<|end|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
        response = response.replace("[/INST]", "").replace("[INST]", "")
        
        # Remove incomplete sentences at the end
        if '. ' in response:
            sentences = response.split('. ')
            if len(sentences) > 1 and len(sentences[-1].strip()) < 15:
                response = '. '.join(sentences[:-1]) + '.'
                
        return response.strip()

    def _postprocess_response(self, response: str, original_prompt: str) -> str:
        """Post-process response for JSON format if needed"""
        # Check if JSON format was requested
        if '"response":' in original_prompt or '{ "response":' in original_prompt:
            # If response already looks like valid JSON, return as-is
            if response.strip().startswith('{"response":') and response.strip().endswith('}'):
                return response
                
            # If response doesn't look like JSON, wrap it
            if not (response.strip().startswith('{') and response.strip().endswith('}')):
                # Clean the response and wrap in JSON
                clean_response = response.replace('"', '\\"').replace('\n', '\\n').replace('\r', '').replace('\t', ' ')
                
                # Truncate very long responses
                if len(clean_response) > 4000:
                    truncate_point = 4000
                    last_period = clean_response.rfind('.', 0, truncate_point)
                    if last_period > 2000:
                        clean_response = clean_response[:last_period + 1]
                    else:
                        clean_response = clean_response[:truncate_point] + "..."
                
                response = f'{{"response": "{clean_response}"}}'
        
        return response

class JambaModel:
    """AI21 Jamba Reasoning 3B model implementation for API use"""
    def __init__(self, model: str = "ai21labs/AI21-Jamba-Reasoning-3B"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading Jamba Reasoning model: {model}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
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
            
            logger.info(f"Jamba Reasoning model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Jamba Reasoning model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            start_time = time.time()
            
            # Enhanced prompt preprocessing for Jamba
            processed_prompt = self._preprocess_prompt(prompt)
            
            # Jamba supports reasonably large context windows
            inputs = self.tokenizer(
                processed_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,  # Jamba context window
                padding=False
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            max_new_tokens = min(max(200, max_length - input_length), 2048)
            
            if max_new_tokens <= 0:
                logger.warning(f"Input too long, no room for generation. Input length: {input_length}")
                return "Input prompt too long for generation."
            
            # Generate with Jamba-optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    use_cache=True
                )
            
            # Decode and extract only the generated part
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if processed_prompt in full_response:
                response = full_response[len(processed_prompt):].strip()
            else:
                # Fallback: just decode the new tokens
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Post-process for JSON format if needed
            response = self._postprocess_response(response, prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"Jamba Reasoning generation completed in {generation_time:.2f}s")
            
            return response if response else "Unable to generate response."
            
        except Exception as e:
            logger.error(f"Error in Jamba Reasoning generation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"

    def _preprocess_prompt(self, prompt: str) -> str:
        """Preprocess prompt for Jamba Reasoning format"""
        # Check if JSON format is requested
        if '"response":' in prompt or '{ "response":' in prompt:
            formatted_prompt = f"### User Request:\n{prompt}\nProvide your response in the requested JSON format.\n\n### Response:\n"
        else:
            formatted_prompt = f"### User Request:\n{prompt}\n\n### Response:\n"
        return formatted_prompt

    def _clean_response(self, response: str) -> str:
        """Clean up Jamba Reasoning response artifacts"""
        # Remove any header artifacts
        response = response.replace("### Response:", "").replace("### User Request:", "")
        
        # Remove incomplete sentences at the end
        if '. ' in response:
            sentences = response.split('. ')
            if len(sentences) > 1 and len(sentences[-1].strip()) < 15:
                response = '. '.join(sentences[:-1]) + '.'
                
        return response.strip()

    def _postprocess_response(self, response: str, original_prompt: str) -> str:
        """Post-process response for JSON format if needed"""
        # Check if JSON format was requested
        if '"response":' in original_prompt or '{ "response":' in original_prompt:
            # If response already looks like valid JSON, return as-is
            if response.strip().startswith('{"response":') and response.strip().endswith('}'):
                return response
                
            # If response doesn't look like JSON, wrap it
            if not (response.strip().startswith('{') and response.strip().endswith('}')):
                # Clean the response and wrap in JSON
                clean_response = response.replace('"', '\\"').replace('\n', '\\n').replace('\r', '').replace('\t', ' ')
                
                # Truncate very long responses
                if len(clean_response) > 4000:
                    truncate_point = 4000
                    last_period = clean_response.rfind('.', 0, truncate_point)
                    if last_period > 2000:
                        clean_response = clean_response[:last_period + 1]
                    else:
                        clean_response = clean_response[:truncate_point] + "..."
                
                response = f'{{"response": "{clean_response}"}}'
        
        return response

# Global model instances
smollm_generator = None
deepseek_generator = None
phi35_generator = None
gptoss_generator = None
nemotron_generator = None
jamba_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and clean up on shutdown"""
    global smollm_generator, deepseek_generator, phi35_generator, gptoss_generator, nemotron_generator, jamba_generator
    
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
    
    try:
        logger.info("Loading Phi-3.5-mini model...")
        phi35_generator = Phi35MiniModel()
        logger.info("Phi-3.5-mini model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Phi-3.5-mini model: {e}")
        phi35_generator = None
    
    try:
        logger.info("Loading GPT OSS model...")
        gptoss_generator = GPTOSSModel()
        logger.info("GPT OSS model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load GPT OSS model: {e}")
        gptoss_generator = None
    
    try:
        logger.info("Loading Nemotron model...")
        nemotron_generator = NemotronModel()
        logger.info("Nemotron model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Nemotron model: {e}")
        nemotron_generator = None
    
    try:
        logger.info("Loading Jamba Reasoning model...")
        jamba_generator = JambaModel()
        logger.info("Jamba Reasoning model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Jamba Reasoning model: {e}")
        jamba_generator = None
    
    if smollm_generator is None and deepseek_generator is None and phi35_generator is None and gptoss_generator is None and nemotron_generator is None and jamba_generator is None:
        logger.error("Failed to load any models!")
        raise RuntimeError("No models could be loaded")
    
    logger.info("SLM API server ready!")
    yield
    
    # Cleanup
    logger.info("Shutting down SLM API server...")

# Initialize the FastAPI app with lifespan
app = FastAPI(
    title="SLM API for CVDP Benchmark",
    description="Small Language Model API server supporting SmolLM2, DeepSeek, Phi-3.5, GPT OSS, Nemotron, and Jamba models",
    version="1.0.0",
    lifespan=lifespan
)

# Define the request data structure using Pydantic
class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for text generation")
    max_length: Optional[int] = Field(default=None, ge=1, le=32768, description="Maximum length of generated text - 10x enhanced")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32768, description="Maximum tokens to generate (alias for max_length) - 10x enhanced")
    model: Optional[str] = Field(default="smollm", description="Model to use: 'smollm', 'deepseek', 'phi35', 'gptoss', 'nemotron', or 'jamba'")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    
    def get_max_length(self) -> int:
        """Get the maximum length, supporting both max_length and max_tokens parameters"""
        if self.max_tokens is not None:
            return self.max_tokens
        elif self.max_length is not None:
            return self.max_length
        else:
            return 10240  # 10x enhanced default: was 1024, now 10240

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
    
    if phi35_generator:
        models.append({
            "name": phi35_generator.model_name,
            "type": "Phi-3-mini-4k-instruct",
            "device": phi35_generator.device,
            "status": "available"
        })
    
    if gptoss_generator:
        models.append({
            "name": gptoss_generator.model_name,
            "type": "GPT-OSS",
            "device": gptoss_generator.device,
            "status": "available"
        })
    
    if nemotron_generator:
        models.append({
            "name": nemotron_generator.model_name,
            "type": "NVIDIA-Nemotron",
            "device": nemotron_generator.device,
            "status": "available"
        })
    
    if jamba_generator:
        models.append({
            "name": jamba_generator.model_name,
            "type": "AI21-Jamba-Reasoning-3B",
            "device": jamba_generator.device,
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
    
    if phi35_generator:
        info["phi35"] = {
            "model_name": phi35_generator.model_name,
            "device": phi35_generator.device,
            "model_type": "Phi-3-mini-4k-instruct",
            "status": "available"
        }
    else:
        info["phi35"] = {"status": "unavailable", "error": "Model failed to load"}
    
    if gptoss_generator:
        info["gptoss"] = {
            "model_name": gptoss_generator.model_name,
            "device": gptoss_generator.device,
            "model_type": "GPT-OSS",
            "status": "available"
        }
    else:
        info["gptoss"] = {"status": "unavailable", "error": "Model failed to load"}
    
    if nemotron_generator:
        info["nemotron"] = {
            "model_name": nemotron_generator.model_name,
            "device": nemotron_generator.device,
            "model_type": "NVIDIA-Nemotron",
            "status": "available"
        }
    else:
        info["nemotron"] = {"status": "unavailable", "error": "Model failed to load"}
    
    if jamba_generator:
        info["jamba"] = {
            "model_name": jamba_generator.model_name,
            "device": jamba_generator.device,
            "model_type": "AI21-Jamba-Reasoning-3B",
            "status": "available"
        }
    else:
        info["jamba"] = {"status": "unavailable", "error": "Model failed to load"}
    
    return info

@app.get("/health")
def health_check():
    """Health check endpoint"""
    available_models = []
    if smollm_generator:
        available_models.append("smollm")
    if deepseek_generator:
        available_models.append("deepseek")
    if phi35_generator:
        available_models.append("phi35")
    if gptoss_generator:
        available_models.append("gptoss")
    if nemotron_generator:
        available_models.append("nemotron")
    if jamba_generator:
        available_models.append("jamba")
    
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
    - **max_length**: Maximum length of generated response (1-40960)
    - **max_tokens**: Alternative parameter name for max_length (1-40960)
    - **model**: Model to use ('smollm', 'deepseek', 'phi35', 'gptoss', 'nemotron', or 'jamba')
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
        elif request.model == "phi35":
            if phi35_generator is None:
                raise HTTPException(
                    status_code=503,
                    detail="Phi-3.5-mini model is not available"
                )
            generator = phi35_generator
        elif request.model == "gptoss":
            if gptoss_generator is None:
                raise HTTPException(
                    status_code=503,
                    detail="GPT OSS model is not available"
                )
            generator = gptoss_generator
        elif request.model == "nemotron":
            if nemotron_generator is None:
                raise HTTPException(
                    status_code=503,
                    detail="Nemotron model is not available"
                )
            generator = nemotron_generator
        elif request.model == "jamba":
            if jamba_generator is None:
                raise HTTPException(
                    status_code=503,
                    detail="Jamba Reasoning model is not available"
                )
            generator = jamba_generator
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {request.model}. Use 'smollm', 'deepseek', 'phi35', 'gptoss', 'nemotron', or 'jamba'"
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
            max_length=8000,  # 10x enhanced: was 800, now 8000
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
            max_length=15000,  # 10x enhanced: was 1500, now 15000
            model="deepseek",
            temperature=0.4
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Long response test failed: {str(e)}"}

@app.post("/test_phi35")
def test_phi35_generation():
    """Test endpoint specifically for Phi-3.5-mini model"""
    try:
        test_request = PromptRequest(
            prompt="Explain the key advantages of Microsoft's Phi-3.5-mini model for edge deployment.",
            max_length=300,
            model="phi35",
            temperature=0.6
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Phi-3.5 test failed: {str(e)}"}

@app.post("/test_phi35_json")
def test_phi35_json_generation():
    """Test endpoint for Phi-3.5 JSON format responses"""
    try:
        test_request = PromptRequest(
            prompt='Answer this question in JSON format: { "response": "your answer" } What are the main features of the Phi-3.5-mini model architecture?',
            max_length=400,
            model="phi35",
            temperature=0.4
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Phi-3.5 JSON test failed: {str(e)}"}

@app.post("/gptoss/generate")
def generate_gptoss_text(request: PromptRequest):
    """Generate text specifically using GPT OSS model"""
    request.model = "gptoss"
    return generate_text(request)

@app.post("/test_gptoss")
def test_gptoss_generation():
    """Test endpoint for GPT OSS model"""
    try:
        test_request = PromptRequest(
            prompt="Explain the concept of artificial intelligence in simple terms.",
            max_length=200,
            model="gptoss",
            temperature=0.7
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"GPT OSS test failed: {str(e)}"}

@app.post("/test_gptoss_json")
def test_gptoss_json_generation():
    """Test endpoint for GPT OSS JSON format responses"""
    try:
        test_request = PromptRequest(
            prompt='Answer this question in JSON format: { "response": "your answer" } What are the advantages of open-source language models?',
            max_length=400,
            model="gptoss",
            temperature=0.4
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"GPT OSS JSON test failed: {str(e)}"}

@app.post("/nemotron/generate")
def generate_nemotron_text(request: PromptRequest):
    """Generate text specifically using Nemotron model"""
    request.model = "nemotron"
    return generate_text(request)

@app.post("/test_nemotron")
def test_nemotron_generation():
    """Test endpoint for Nemotron model"""
    try:
        test_request = PromptRequest(
            prompt="Explain the key features and advantages of NVIDIA Nemotron models for enterprise AI applications.",
            max_length=300,
            model="nemotron",
            temperature=0.6
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Nemotron test failed: {str(e)}"}

@app.post("/test_nemotron_json")
def test_nemotron_json_generation():
    """Test endpoint for Nemotron JSON format responses"""
    try:
        test_request = PromptRequest(
            prompt='Answer this question in JSON format: { "response": "your answer" } What are the main capabilities of NVIDIA Nemotron models?',
            max_length=500,
            model="nemotron",
            temperature=0.4
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Nemotron JSON test failed: {str(e)}"}

@app.post("/jamba/generate")
def generate_jamba_text(request: PromptRequest):
    """Generate text specifically using Jamba Reasoning model"""
    request.model = "jamba"
    return generate_text(request)

@app.post("/test_jamba")
def test_jamba_generation():
    """Test endpoint for Jamba Reasoning model"""
    try:
        test_request = PromptRequest(
            prompt="Explain step-by-step how to solve this problem: If a barrel shifter has 8-bit input and shifts by 3 positions to the right, what is the output?",
            max_length=400,
            model="jamba",
            temperature=0.6
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Jamba Reasoning test failed: {str(e)}"}

@app.post("/test_jamba_json")
def test_jamba_json_generation():
    """Test endpoint for Jamba Reasoning JSON format responses"""
    try:
        test_request = PromptRequest(
            prompt='Answer this question in JSON format: { "response": "your answer" } What are the key features of AI21 Jamba Reasoning models and why are they effective for reasoning tasks?',
            max_length=600,
            model="jamba",
            temperature=0.4
        )
        return generate_text(test_request)
    except Exception as e:
        return {"error": f"Jamba Reasoning JSON test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    logger.info("Starting SLM API server...")
    uvicorn.run(
        "main:app",  # Fixed: was "slm_api_code:app", now "main:app"
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )