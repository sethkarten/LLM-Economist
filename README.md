# LLM Economist

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

A comprehensive framework for economic simulations using Large Language Models (LLMs). The LLM Economist leverages state-of-the-art language models to create realistic, dynamic economic simulations with diverse agent populations for studying tax policy optimization and mechanism design.

## 🚀 Features

- **Multi-LLM Support**: Compatible with OpenAI GPT, Google Gemini, Anthropic Claude, Meta Llama, and more
- **Multiple Deployment Options**: Local (vLLM, Ollama), cloud APIs (OpenAI, OpenRouter), and Google AI
- **Diverse Economic Scenarios**: Rational agents, bounded rationality, and democratic voting mechanisms
- **Realistic Agent Personas**: LLM-generated personas based on real demographic and occupational data
- **Scalable Architecture**: Support for 3-1000+ agents with efficient parallel processing
- **Comprehensive Testing**: Full test suite with real API integration testing
- **Reproducible Research**: Standardized experiment scripts and configuration management

## 📖 Overview

The LLM Economist framework models economic systems as a two-level multi-agent reinforcement learning problem, implemented as a Stackelberg game where:

1. **Tax Planner (Leader)**: Sets tax policies to maximize social welfare
2. **Workers (Followers)**: Optimize labor allocation based on tax policies and individual utility functions

Key innovations include:
- **In-context optimization** for rational utility functions
- **Synthetic demographic data** for realistic agent diversity using real occupation, age, and gender statistics
- **LLM-generated personas** that create unique, realistic economic agents
- **Mechanism design** for positive societal influence

## 🛠️ Installation

### Initialize Conda Environment

```bash
# Create and activate the environment
conda create -n LLMEconomist python=3.11 -y
conda activate LLMEconomist
```

### Quick Install

```bash
pip install llm-economist
```

### Development Install

```bash
git clone https://github.com/sethkarten/LLMEconomist.git
cd LLMEconomist
pip install -e .
```

### Dependencies

The framework supports multiple LLM providers. Install additional dependencies as needed:

```bash
# For local LLM serving
pip install vllm ollama

# For Google Gemini
pip install google-generativeai

# For development
pip install -e .[dev]
```

## 🚦 Quick Start

### 1. Set up API Keys

Choose your preferred LLM provider and set the corresponding API key:

```bash
# OpenAI
export OPENAI_API_KEY="your_openai_key"

# OpenRouter (for multiple models)
export OPENROUTER_API_KEY="your_openrouter_key"

# Google Gemini
export GOOGLE_API_KEY="your_google_key"
```

### 2. Run Your First Simulation

```bash
# Simple rational agents simulation
python -m llm_economist.main --scenario rational --num-agents 5 --max-timesteps 500

# Bounded rationality simulation (note: currently uses 100% egotistical agents with personas)
python -m llm_economist.main --scenario bounded --num-agents 10 --percent-ego 100

# Democratic voting simulation
python -m llm_economist.main --scenario democratic --num-agents 15 --two-timescale 50
```

### 3. Try Different LLM Models

```bash
# OpenAI GPT-4
python -m llm_economist.main --llm gpt-4o --scenario rational

# Local Llama via vLLM (requires local server)
python -m llm_economist.main --llm meta-llama/Llama-3.1-8B-Instruct --service vllm --port 8000

# Claude via OpenRouter
python -m llm_economist.main --llm anthropic/claude-3.5-sonnet --use-openrouter

# Google Gemini
python -m llm_economist.main --llm gemini-1.5-flash
```

## 🏗️ Project Structure

```
LLMEconomist/
├── llm_economist/              # Main package
│   ├── agents/                 # Agent implementations
│   │   ├── worker.py          # Worker agent logic
│   │   ├── planner.py         # Tax planner logic
│   │   └── llm_agent.py       # Base LLM agent class
│   ├── models/                 # LLM model integrations
│   │   ├── openai_model.py    # OpenAI GPT models
│   │   ├── gemini_model.py    # Google Gemini models
│   │   ├── vllm_model.py      # Local vLLM/Ollama models
│   │   ├── openrouter_model.py # OpenRouter API
│   │   └── base.py            # Base model interface
│   ├── utils/                  # Utility functions
│   │   ├── common.py          # Common utilities
│   │   └── bracket.py         # Tax bracket utilities
│   ├── data/                   # Demographic data files
│   └── main.py                 # Main entry point
├── experiments/                # Experiment scripts
├── examples/                   # Usage examples
│   ├── quick_start.py         # Basic functionality tests
│   └── advanced_usage.py      # Simulation scenario tests
├── tests/                      # Test suite
└── README.md                   # This file
```

## 🔧 Configuration Options

### Simulation Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--scenario` | Economic scenario | `rational` | `rational`, `bounded`, `democratic` |
| `--num-agents` | Number of worker agents | `5` | `1-1000+` |
| `--max-timesteps` | Simulation length | `1000` | Any positive integer |
| `--two-timescale` | Steps between tax updates | `25` | Any positive integer |

### LLM Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--llm` | LLM model to use | `gpt-4o-mini` | See supported models below |
| `--prompt-algo` | Prompting strategy | `io` | `io`, `cot` |
| `--service` | Local LLM service | `vllm` | `vllm`, `ollama` |
| `--port` | Local server port | `8000` | Any valid port |

### Agent Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--worker-type` | Worker agent type | `LLM` | `LLM`, `FIXED` |
| `--planner-type` | Planner agent type | `LLM` | `LLM`, `US_FED`, `UNIFORM` |
| `--percent-ego` | % egotistical agents | `100` | `0-100` |
| `--percent-alt` | % altruistic agents | `0` | `0-100` |
| `--percent-adv` | % adversarial agents | `0` | `0-100` |

**Note**: Currently, personas (used in `bounded` and `democratic` scenarios) only support egotistical utility types, so mixed utility types are only available with default personas.

## 🤖 Supported LLM Models

### Cloud APIs

**OpenAI Models:**
- `gpt-4o` - Most capable, highest cost
- `gpt-4o-mini` - Fast and cost-effective (recommended)

**Via OpenRouter (requires OPENROUTER_API_KEY):**
- `meta-llama/llama-3.1-8b-instruct` - Open source, good performance
- `meta-llama/llama-3.1-70b-instruct` - Larger Llama model
- `anthropic/claude-3.5-sonnet` - Excellent reasoning
- `google/gemini-flash-1.5` - Fast Google model

**Google Gemini (requires GOOGLE_API_KEY):**
- `gemini-1.5-pro` - Most capable Gemini model
- `gemini-1.5-flash` - Fast and efficient (recommended)

### Local Deployment

**vLLM (Recommended for local deployment):**
```bash
# Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Use in simulation
python -m llm_economist.main --llm meta-llama/Llama-3.1-8B-Instruct --service vllm --port 8000
```

**Ollama (Easy local setup):**
```bash
# Install and start Ollama
ollama pull llama3.1:8b
ollama serve

# Use in simulation
python -m llm_economist.main --llm llama3.1:8b --service ollama --port 11434
```

## 📊 Experiment Scripts

### Pre-configured Experiments

Run the experiments from the paper:

```bash
# All experiments
python experiments/run_experiments.py --experiment all

# Specific experiments
python experiments/run_experiments.py --experiment rational
python experiments/run_experiments.py --experiment bounded
python experiments/run_experiments.py --experiment democratic
python experiments/run_experiments.py --experiment llm_comparison
python experiments/run_experiments.py --experiment scalability
```

### Custom Experiments

```bash
# Chain of thought prompting
python -m llm_economist.main --prompt-algo cot --llm gpt-4o

# Input-output prompting (default)
python -m llm_economist.main --prompt-algo io --llm gpt-4o-mini

# Large scale simulation
python -m llm_economist.main --num-agents 100 --max-timesteps 2000
```

## 📈 Examples

The framework provides two types of examples:

### Basic Functionality Tests

For quick validation of imports, setup, and basic functionality:

```bash
# Test all basic functionality
python examples/quick_start.py

# Run specific basic tests
python examples/quick_start.py --help
```

The quick start script validates:
- Package imports and dependencies
- Argument parser configuration
- API key detection
- Basic Args object creation
- Service configurations

### Advanced Usage Examples

For actual simulation testing with 20-timestep runs:

```bash
# Run all simulation scenarios
python examples/advanced_usage.py

# Test specific scenarios
python examples/advanced_usage.py rational          # OpenAI GPT-4o-mini
python examples/advanced_usage.py bounded           # Bounded rationality with personas
python examples/advanced_usage.py democratic        # Democratic voting mechanism
python examples/advanced_usage.py fixed             # Fixed workers with LLM planner

# Test different LLM providers
python examples/advanced_usage.py openrouter        # OpenRouter API
python examples/advanced_usage.py vllm              # Local vLLM server
python examples/advanced_usage.py ollama            # Local Ollama
python examples/advanced_usage.py gemini            # Google Gemini

# Show available scenarios
python examples/advanced_usage.py --help
```

All advanced examples use 20 timesteps for thorough testing while remaining fast for development.

### Example Organization

The examples are organized to provide clear separation of concerns:

- **`quick_start.py`**: Lightweight validation of basic functionality without running simulations
  - Tests imports and dependencies
  - Validates configuration setup
  - Checks API key availability
  - Fast execution (< 10 seconds)

- **`advanced_usage.py`**: Full simulation testing with real LLM APIs
  - 20-timestep economic simulations
  - All scenarios: rational, bounded, democratic, fixed workers
  - Multiple LLM providers: OpenAI, OpenRouter, vLLM, Ollama, Gemini
  - Realistic testing (2-10 minutes per scenario)

## 🧪 Testing

The framework includes comprehensive tests organized into three categories:

### Basic Functionality Tests

```bash
# Test basic functionality (imports, setup, configuration)
pytest tests/test_quickstart.py -v

# Test individual components
python examples/quick_start.py  # Direct basic functionality validation
```

### Integration Tests

```bash
# Test LLM model integrations
pytest tests/test_models.py -v

# Test simulation logic with mocking
pytest tests/test_simulation.py -v

# Test advanced usage scenarios (requires API keys)
pytest tests/test_advanced_usage.py -v
```

### End-to-End Tests

```bash
# Test actual simulations with real APIs
python examples/advanced_usage.py           # All scenarios
python examples/advanced_usage.py rational  # Specific scenario
```

### Full Test Suite

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=llm_economist --cov-report=html
```

### Test Requirements

- **API Keys**: Advanced usage and integration tests require API keys:
  - `OPENAI_API_KEY` or `ECON_OPENAI` (required for most tests)
  - `OPENROUTER_API_KEY` (optional, for OpenRouter tests)
  - `GOOGLE_API_KEY` (optional, for Gemini tests)
- **Real Integration**: Advanced tests use actual LLM APIs to ensure end-to-end functionality
- **Fast Execution**: All tests use 20 timesteps or less for quick validation
- **Local Servers**: vLLM and Ollama tests require running local servers (will skip if not available)

## 🎭 Agent Personas

The framework generates realistic agent personas using:

1. **Demographic Sampling**: Real occupation, age, and gender statistics from census data
2. **LLM Generation**: Each persona is uniquely generated based on sampled demographics
3. **Economic Realism**: Personas include realistic income levels, risk tolerance, and life circumstances

Example generated personas:
- *"You are a 55-year-old female working as a licensed practical nurse... With over 30 years of experience, you prioritize savings for retirement and healthcare needs."*
- *"You are a 53-year-old male working as a welding worker... concerns about retirement savings keep you financially cautious."*

## 📚 Research Reproduction

To reproduce the experiments from the LLM Economist paper:

### Setup

1. **Environment Setup:**
   ```bash
   git clone https://github.com/sethkarten/LLMEconomist.git
   cd LLMEconomist
   pip install -e .
   export WANDB_API_KEY="your_wandb_key"  # For experiment tracking
   ```

2. **LLM Setup (choose one):**
   
   **Option A: OpenAI (easiest):**
   ```bash
   export OPENAI_API_KEY="your_key"
   ```
   
   **Option B: Local vLLM (most cost-effective):**
   ```bash
   # Start vLLM server with Llama 3.1 8B
   vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 1 --port 8000
   ```

### Main Experiments

```bash
# Rational agents 
python experiments/run_experiments.py --experiment rational --wandb

# Bounded rationality 
python experiments/run_experiments.py --experiment bounded --wandb

# Democratic voting 
python experiments/run_experiments.py --experiment democratic --wandb

# LLM comparison 
python experiments/run_experiments.py --experiment llm_comparison --wandb

# Scalability analysis 
python experiments/run_experiments.py --experiment scalability --wandb


```

## 🚀 Advanced Features

### Custom Agent Types

Extend the framework with custom agent behaviors:

```python
from llm_economist.agents.worker import Worker

class CustomWorker(Worker):
    def compute_utility(self, income, rebate):
        # Custom utility function
        return your_custom_utility_logic(income, rebate)
```

### Custom LLM Models

Add support for new LLM providers:

```python
from llm_economist.models.base import BaseLLMModel

class CustomLLMModel(BaseLLMModel):
    def send_msg(self, system_prompt, user_prompt, temperature=None, json_format=False):
        # Implement your model's API
        return response, is_json
```

### Experiment Tracking

Enable detailed experiment tracking with Weights & Biases:

```bash
python -m llm_economist.main --wandb --scenario bounded --num-agents 20
```

## 🐛 Troubleshooting

### Common Issues

**API Key Errors:**
```bash
# Make sure your API keys are set correctly
echo $OPENAI_API_KEY
echo $OPENROUTER_API_KEY
echo $GOOGLE_API_KEY
```

**Local Model Connection:**
```bash
# Check if vLLM server is running
curl http://localhost:8000/health

# Check Ollama status
ollama list
```

**Memory Issues:**
- Reduce `--num-agents` for large simulations
- Use `gpt-4o-mini` instead of `gpt-4o` for cost efficiency
- Adjust `--history-len` to reduce memory usage

**Rate Limiting:**
- Add delays between API calls
- Use local models (vLLM/Ollama) for unrestricted access
- Switch to OpenRouter for higher rate limits

**Test Failures:**
- Ensure API keys are set for quickstart tests
- Check network connectivity for cloud API tests
- Verify local model servers are running for local tests

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{karten2025llm,
  title={LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra},
  author={Karten, Seth and Li, Wenzhe and Ding, Zihan and Kleiner, Samuel and Bai, Yu and Jin, Chi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
