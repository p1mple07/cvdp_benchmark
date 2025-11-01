#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Iterative Refinement Agent for CVDP Benchmark

This agent uses an iterative approach:
1. Generate initial code using SLM API
2. Run tests (Verilator/Icarus) to check for errors
3. If errors, send code + errors back to SLM for refinement
4. Repeat up to max_iterations (default: 3)
5. Exit early if tests pass
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# Configure logging - force output to both stdout and file
class TeeLogger:
    """Logger that writes to both stdout and file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', buffering=1)  # Line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create log file
log_file = '/code/rundir/agent_detailed.log'
tee = TeeLogger(log_file)

# Configure logging to write to our tee
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=tee,
    force=True
)

logger = logging.getLogger(__name__)

# Also redirect stdout to tee for print statements
sys.stdout = tee


class IterativeRefinementAgent:
    """Agent that iteratively refines HDL code using SLM API and test feedback"""
    
    def __init__(self, max_iterations: int = 3):
        """
        Initialize the agent.
        
        Args:
            max_iterations: Maximum number of refinement iterations
        """
        self.max_iterations = max_iterations
        self.slm_api_url = os.getenv("SLM_API_URL", "http://host.docker.internal:8000")
        self.slm_model = os.getenv("SLM_MODEL", "phi35")
        self.slm_max_length = int(os.getenv("SLM_MAX_LENGTH", "8192"))
        self.slm_timeout = int(os.getenv("SLM_TIMEOUT", "300"))
        
        logger.info(f"Initialized IterativeRefinementAgent")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  SLM API URL: {self.slm_api_url}")
        logger.info(f"  SLM Model: {self.slm_model}")
        logger.info(f"  Max length: {self.slm_max_length}")
        
    def read_prompt(self) -> str:
        """Read the task from prompt.json"""
        try:
            with open("/code/prompt.json", "r") as f:
                data = json.load(f)
                prompt = data.get("prompt", "")
                logger.info(f"Read prompt: {prompt[:200]}...")
                return prompt
        except Exception as e:
            logger.error(f"Error reading prompt.json: {e}")
            return ""
    
    def gather_context(self) -> Dict[str, str]:
        """Gather all existing files as context"""
        context = {}
        
        for dir_name in ["docs", "rtl", "verif", "rundir"]:
            dir_path = Path(f"/code/{dir_name}")
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                relative_path = file_path.relative_to("/code")
                                content = f.read()
                                context[str(relative_path)] = content
                                logger.info(f"  Loaded: {relative_path} ({len(content)} bytes)")
                        except Exception as e:
                            logger.warning(f"  Could not read {file_path}: {e}")
        
        logger.info(f"Gathered context from {len(context)} files")
        return context
    
    def format_context_for_prompt(self, context: Dict[str, str], max_files: int = 10) -> str:
        """Format context files for inclusion in prompt"""
        lines = []
        file_count = 0
        
        # Prioritize docs, then rtl, then verif
        priority_order = ["docs/", "rtl/", "verif/"]
        
        for prefix in priority_order:
            for file_path, content in context.items():
                if file_path.startswith(prefix) and file_count < max_files:
                    lines.append(f"\n### File: {file_path}\n```\n{content}\n```")
                    file_count += 1
        
        return "\n".join(lines)
    
    def call_slm_api(self, prompt: str) -> Optional[str]:
        """
        Call the SLM API to generate code.
        
        Args:
            prompt: The prompt to send to the SLM
            
        Returns:
            Generated code or None if failed
        """
        try:
            import requests
            
            payload = {
                "prompt": prompt,
                "max_length": self.slm_max_length,
                "model": self.slm_model
            }
            
            logger.info(f"Calling SLM API at {self.slm_api_url}/generate")
            logger.info(f"  Model: {self.slm_model}, Max length: {self.slm_max_length}")
            
            response = requests.post(
                f"{self.slm_api_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.slm_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                # Try different response field names
                for field in ['generated_text', 'text', 'response', 'output', 'result']:
                    if field in data:
                        result = data[field]
                        logger.info(f"Received {len(result)} bytes from SLM")
                        return result
                
                # If no known field, return the whole response as string
                result = str(data)
                logger.info(f"Received response (unknown format): {len(result)} bytes")
                return result
            else:
                logger.error(f"SLM API returned status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling SLM API: {e}")
            return None
    
    def extract_verilog_code(self, response: str) -> str:
        """Extract Verilog/SystemVerilog code from SLM response"""
        # Try to extract code from markdown code blocks
        if "```verilog" in response or "```systemverilog" in response:
            # Extract from markdown code block
            start_markers = ["```verilog", "```systemverilog", "```sv"]
            for marker in start_markers:
                if marker in response:
                    start = response.find(marker) + len(marker)
                    end = response.find("```", start)
                    if end != -1:
                        code = response[start:end].strip()
                        logger.info(f"Extracted code from markdown block: {len(code)} bytes")
                        return code
        
        # If no markdown, check if it starts with module
        if "module " in response:
            # Try to extract just the module definition
            lines = response.split("\n")
            code_lines = []
            in_module = False
            for line in lines:
                if "module " in line:
                    in_module = True
                if in_module:
                    code_lines.append(line)
                if "endmodule" in line:
                    break
            
            if code_lines:
                code = "\n".join(code_lines)
                logger.info(f"Extracted module definition: {len(code)} bytes")
                return code
        
        # Otherwise return as-is
        logger.info(f"Using response as-is: {len(response)} bytes")
        return response
    
    def find_target_file(self) -> Optional[Path]:
        """Find the file that needs to be created/modified"""
        # Check rtl directory for files that don't exist or are empty
        rtl_dir = Path("/code/rtl")
        if rtl_dir.exists():
            for file_path in rtl_dir.iterdir():
                if file_path.is_file():
                    if file_path.stat().st_size == 0:
                        logger.info(f"Found empty target file: {file_path}")
                        return file_path
        
        # If no empty files, look for common top-level names
        common_names = ["top.sv", "top.v", "top_module.sv", "top_module.v"]
        for name in common_names:
            path = rtl_dir / name
            if not path.exists():
                logger.info(f"Will create target file: {path}")
                return path
        
        # Default to first .sv or .v file in rtl
        if rtl_dir.exists():
            for file_path in rtl_dir.glob("*.sv"):
                logger.info(f"Using existing file: {file_path}")
                return file_path
            for file_path in rtl_dir.glob("*.v"):
                logger.info(f"Using existing file: {file_path}")
                return file_path
        
        return None
    
    def write_code(self, file_path: Path, code: str) -> bool:
        """Write code to file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(code)
            logger.info(f"Wrote {len(code)} bytes to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {e}")
            return False
    
    def run_tests(self) -> Tuple[bool, str]:
        """
        Run tests using available tools (Verilator or Icarus Verilog).
        
        Returns:
            Tuple of (success, error_message)
        """
        # First try to find test harness
        rundir = Path("/code/rundir")
        
        # Check if there's a pytest-based test
        if (rundir / "../harness").exists() or Path("/src/test_runner.py").exists():
            return self.run_cocotb_tests()
        
        # Otherwise try verilator or icarus
        return self.run_lint_checks()
    
    def run_cocotb_tests(self) -> Tuple[bool, str]:
        """Run CocoTB-based tests if available"""
        try:
            logger.info("Running CocoTB tests...")
            
            # Try to run pytest
            result = subprocess.run(
                ["pytest", "-v", "-s", "/src/test_runner.py"],
                cwd="/code/rundir",
                capture_output=True,
                text=True,
                timeout=120
            )
            
            output = result.stdout + "\n" + result.stderr
            logger.info(f"Test output:\n{output[:1000]}")
            
            if result.returncode == 0:
                logger.info("✅ CocoTB tests PASSED")
                return True, ""
            else:
                logger.warning(f"❌ CocoTB tests FAILED (exit code: {result.returncode})")
                # Extract relevant errors (last 50 lines)
                error_lines = output.split("\n")[-50:]
                errors = "\n".join(error_lines)
                return False, errors
                
        except subprocess.TimeoutExpired:
            logger.error("CocoTB tests timed out")
            return False, "Tests timed out after 120 seconds"
        except FileNotFoundError:
            logger.warning("CocoTB tests not available, falling back to lint checks")
            return self.run_lint_checks()
        except Exception as e:
            logger.error(f"Error running CocoTB tests: {e}")
            return False, str(e)
    
    def run_lint_checks(self) -> Tuple[bool, str]:
        """Run Verilator or Icarus lint checks"""
        rtl_files = list(Path("/code/rtl").glob("*.v")) + list(Path("/code/rtl").glob("*.sv"))
        
        if not rtl_files:
            logger.warning("No RTL files found to check")
            return False, "No RTL files found"
        
        # Try Verilator first
        try:
            logger.info("Running Verilator lint checks...")
            result = subprocess.run(
                ["verilator", "--lint-only", "-Wall"] + [str(f) for f in rtl_files],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stderr + result.stdout
            
            if result.returncode == 0:
                logger.info("✅ Verilator lint checks PASSED")
                return True, ""
            else:
                logger.warning(f"❌ Verilator lint checks FAILED")
                # Extract first 50 lines of errors
                error_lines = output.split("\n")[:50]
                errors = "\n".join(error_lines)
                return False, errors
                
        except FileNotFoundError:
            logger.info("Verilator not found, trying Icarus Verilog...")
        except Exception as e:
            logger.warning(f"Verilator failed: {e}")
        
        # Try Icarus Verilog
        try:
            logger.info("Running Icarus Verilog checks...")
            result = subprocess.run(
                ["iverilog", "-tnull", "-Wall"] + [str(f) for f in rtl_files],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stderr + result.stdout
            
            if result.returncode == 0 and "error" not in output.lower():
                logger.info("✅ Icarus Verilog checks PASSED")
                return True, ""
            else:
                logger.warning(f"❌ Icarus Verilog checks FAILED")
                error_lines = output.split("\n")[:50]
                errors = "\n".join(error_lines)
                return False, errors
                
        except Exception as e:
            logger.error(f"Icarus Verilog failed: {e}")
            return False, f"No suitable verification tool available: {e}"
    
    def build_initial_prompt(self, task: str, context: Dict[str, str]) -> str:
        """Build the initial prompt for code generation"""
        context_str = self.format_context_for_prompt(context)
        
        prompt = f"""You are an expert Verilog/SystemVerilog RTL designer. Generate clean, synthesizable code.

Task:
{task}

Context Files:
{context_str}

Generate ONLY the Verilog/SystemVerilog code needed to complete the task. Do not include explanations.
Start with 'module' and end with 'endmodule'.
"""
        return prompt
    
    def build_refinement_prompt(self, task: str, previous_code: str, errors: str, iteration: int) -> str:
        """Build a refinement prompt with previous code and errors"""
        prompt = f"""You are an expert Verilog/SystemVerilog RTL designer fixing compilation/test errors.

Original Task:
{task}

Previous Code (Iteration {iteration-1}):
```verilog
{previous_code}
```

Test Errors:
```
{errors}
```

Instructions:
Fix ONLY the specific errors shown above. Generate the complete corrected Verilog/SystemVerilog code.
Do not include explanations. Start with 'module' and end with 'endmodule'.
"""
        return prompt
    
    def run(self) -> int:
        """Main agent execution loop"""
        try:
            logger.info("="*80)
            logger.info("ITERATIVE REFINEMENT AGENT STARTING")
            logger.info("="*80)
            
            # Step 1: Read task
            task = self.read_prompt()
            if not task:
                logger.error("No task found in prompt.json")
                return 0
            
            # Step 2: Gather context
            context = self.gather_context()
            
            # Step 3: Find target file
            target_file = self.find_target_file()
            if not target_file:
                logger.error("Could not determine target file")
                return 0
            
            logger.info(f"Target file: {target_file}")
            
            # Step 4: Iterative refinement loop
            code = None
            errors = None
            
            for iteration in range(1, self.max_iterations + 1):
                logger.info("\n" + "="*80)
                logger.info(f"ITERATION {iteration}/{self.max_iterations}")
                logger.info("="*80)
                
                # Build prompt
                if iteration == 1:
                    prompt = self.build_initial_prompt(task, context)
                else:
                    prompt = self.build_refinement_prompt(task, code, errors, iteration)
                
                logger.info(f"Prompt length: {len(prompt)} characters")
                
                # Print the full prompt for debugging
                print("\n" + "="*80)
                print(f"PROMPT SENT TO SLM (Iteration {iteration}):")
                print("="*80)
                print(prompt)
                print("="*80 + "\n")
                
                # Call SLM API
                logger.info("Calling SLM API...")
                response = self.call_slm_api(prompt)
                
                if response is None:
                    logger.error("Failed to get response from SLM API")
                    if code:  # Keep previous code if we have it
                        logger.info("Keeping previous code")
                        continue
                    else:
                        return 0
                
                # Print the full response for debugging
                print("\n" + "="*80)
                print(f"RESPONSE FROM SLM (Iteration {iteration}):")
                print("="*80)
                print(response)
                print("="*80 + "\n")
                
                # Extract code
                code = self.extract_verilog_code(response)
                
                # Print extracted code for debugging
                print("\n" + "="*80)
                print(f"EXTRACTED CODE (Iteration {iteration}):")
                print("="*80)
                print(code)
                print("="*80 + "\n")
                
                # Write code
                if not self.write_code(target_file, code):
                    logger.error("Failed to write code")
                    return 0
                
                # Run tests
                logger.info("Running tests...")
                success, errors = self.run_tests()
                
                if success:
                    logger.info("="*80)
                    logger.info(f"✅ SUCCESS ON ITERATION {iteration}!")
                    logger.info("="*80)
                    return 0
                else:
                    logger.warning("="*80)
                    logger.warning(f"❌ TESTS FAILED ON ITERATION {iteration}")
                    logger.warning("="*80)
                    logger.warning(f"Errors (first 500 chars):\n{errors[:500]}")
                    
                    # Small delay before next iteration
                    time.sleep(2)
            
            # Max iterations reached
            logger.warning("="*80)
            logger.warning(f"⚠️ MAX ITERATIONS ({self.max_iterations}) REACHED")
            logger.warning("="*80)
            logger.warning("Tests did not pass, but exiting cleanly")
            
            return 0  # Always exit cleanly for benchmark
            
        except Exception as e:
            logger.error(f"Agent failed with exception: {e}", exc_info=True)
            return 0  # Exit cleanly even on error


if __name__ == "__main__":
    agent = IterativeRefinementAgent(max_iterations=3)
    exit_code = agent.run()
    sys.exit(exit_code)
