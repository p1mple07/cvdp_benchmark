# Iterative Refinement Agent

An intelligent agent for the CVDP Benchmark that uses iterative refinement with your SLM API.

## How It Works

1. **Read Task**: Loads the problem description from `/code/prompt.json`
2. **Gather Context**: Reads existing files (docs, rtl, verif) for context
3. **Generate Code**: Calls your SLM API to generate initial Verilog/SystemVerilog code
4. **Run Tests**: Executes CocoTB tests or Verilator/Icarus lint checks
5. **Iterate**: If tests fail, sends code + errors back to SLM for refinement
6. **Repeat**: Up to 3 iterations (configurable)
7. **Exit Early**: Stops immediately if tests pass

## Features

- ✅ **Iterative Refinement**: Learns from test failures
- ✅ **Multiple Test Frameworks**: Supports CocoTB, Verilator, Icarus Verilog
- ✅ **Early Stopping**: Exits on first success
- ✅ **Detailed Logging**: Full execution trace in agent.log
- ✅ **Configurable**: Adjust iterations, model, API settings
- ✅ **Model Agnostic**: Works with any SLM API (DeepSeek, GPT-OSS, etc.)

## Building the Agent

```bash
cd iterative-agent
./build_agent.sh
```

This creates a Docker image named `iterative-agent`.

## Configuration

Set these environment variables in your `.env` file:

```bash
# SLM API Configuration
SLM_API_URL=http://localhost:8000
SLM_MODEL=deepseek
SLM_MAX_LENGTH=8192
SLM_TIMEOUT=300
```

**Note**: The agent uses `host.docker.internal` by default to access your host machine's API from inside Docker.

## Running the Agent

### Test on Example Dataset

```bash
# Single problem (for debugging)
python run_benchmark.py \
  -f example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl \
  -i cvdp_agentic_fixed_arbiter_0001 \
  -l -g iterative-agent \
  -p work_debug

# Check results
cat work_debug/report.txt

# View agent logs
cat work_debug/cvdp_agentic_fixed_arbiter_0001/harness/1/code/rundir/agent.log
```

### Test on Full Example Dataset

```bash
# Run all problems in example dataset
python run_benchmark.py \
  -f example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl \
  -l -g iterative-agent \
  -p work_example

# Check results
cat work_example/report.txt
```

### Test on Downloaded Dataset (92 Problems)

```bash
# Run on the full no-commercial dataset
python run_benchmark.py \
  -f cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl \
  -l -g iterative-agent \
  -p work_v1.0.2

# Check results
cat work_v1.0.2/report.txt
```

### Multi-Sample Evaluation (Recommended)

```bash
# Run 5 samples for statistical reliability
python run_samples.py \
  -f cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl \
  -l -g iterative-agent \
  -n 5 -k 1 \
  -p work_multi

# Check composite results
cat work_multi/composite_report.txt
```

### Parallel Execution

```bash
# Run with 4 parallel threads (faster!)
python run_benchmark.py \
  -f cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl \
  -l -g iterative-agent \
  -t 4 \
  -p work_parallel

# Or use multi-sampling with parallelization
python run_samples.py \
  -f cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl \
  -l -g iterative-agent \
  -n 5 -k 1 -t 4 \
  -p work_fast
```

## Debugging

### View Agent Execution

```bash
# Run single problem in debug mode
python run_benchmark.py \
  -f cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl \
  -i cvdp_agentic_64b66b_codec_0001 \
  -l -g iterative-agent \
  -p work_debug

# Navigate to work directory
cd work_debug/cvdp_agentic_64b66b_codec_0001/harness/1/

# Check what the agent did
cat agent_changes.patch

# View agent logs
cat code/rundir/agent.log

# Manually run the agent in interactive mode
./run_docker_agent.sh -d
```

### Inside the Agent Container

```bash
# When in debug mode (-d flag)
cd /code

# Check the task
cat prompt.json

# Check available files
ls -la docs/ rtl/ verif/

# Run agent manually
python3 /app/agent.py

# Run tests manually
cd rundir
pytest -v /src/test_runner.py
```

## Customizing the Agent

### Adjust Iterations

Edit `agent.py`:
```python
agent = IterativeRefinementAgent(max_iterations=5)  # Default: 3
```

### Change SLM Model

Set environment variable:
```bash
export SLM_MODEL=gptoss  # or phi35, nemotron, etc.
```

### Adjust Token Limits

```bash
export SLM_MAX_LENGTH=16384  # For longer responses
```

## Expected Performance

Based on similar iterative approaches in literature:

- **Pass@1**: ~40-60% (vs ~20-30% single-shot)
- **Pass@3 iterations**: ~60-75%
- **Pass@5 samples**: ~70-80%

Actual performance depends on:
- SLM model quality (DeepSeek, GPT-OSS, etc.)
- Problem complexity
- Prompt engineering
- Max token length

## Troubleshooting

### Agent can't connect to SLM API

- Check your API is running: `curl http://localhost:8000/generate`
- Verify `SLM_API_URL` is set correctly
- On macOS/Windows: Use `host.docker.internal`
- On Linux: May need to use host IP instead

### Tests always fail

- Check if test harness files exist
- Verify Verilator/Icarus are available: `docker run iterative-agent which verilator`
- Look at agent logs for detailed error messages

### Agent times out

- Increase `SLM_TIMEOUT` for slower models
- Reduce `SLM_MAX_LENGTH` if generating too much
- Check if your API is responding slowly

## Files Generated

After running, check these locations:

```
work_<prefix>/
├── report.json              # Benchmark results
├── report.txt               # Human-readable report
├── raw_result.json          # Detailed results
└── <problem_id>/
    └── harness/1/
        ├── agent_changes.patch    # What agent changed
        └── code/
            ├── rtl/               # Modified RTL files
            └── rundir/
                └── agent.log      # Agent execution log
```

## Tips for Best Results

1. **Start Small**: Test on example dataset first
2. **Debug Single Problems**: Use `-i` flag to focus on one problem
3. **Check Logs**: Always review `agent.log` for insights
4. **Tune Prompts**: Edit prompt templates in `agent.py`
5. **Increase Iterations**: If close to passing, try 5 iterations
6. **Use Multi-Sampling**: Run 5+ samples for reliable statistics

## Advanced Usage

### Custom Prompt Engineering

Edit the `build_initial_prompt()` and `build_refinement_prompt()` methods in `agent.py` to customize how prompts are constructed.

### Different Test Strategies

Modify the `run_tests()` method to:
- Use different verification tools
- Add custom test scripts
- Implement formal verification
- Add coverage checks

### Model Ensemble

Run multiple models and compare results:
```bash
# Run with DeepSeek
SLM_MODEL=deepseek python run_benchmark.py -f dataset.jsonl -l -g iterative-agent -p work_deepseek

# Run with GPT-OSS
SLM_MODEL=gptoss python run_benchmark.py -f dataset.jsonl -l -g iterative-agent -p work_gptoss

# Compare results
python run_reporter.py work_deepseek/report.json
python run_reporter.py work_gptoss/report.json
```

---

**Happy Testing!** 🚀
