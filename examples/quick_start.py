"""
Quick start examples for the LLM Economist framework.

This module provides basic functionality tests and setup validation.
For actual simulation examples, see advanced_usage.py.
"""

import os
import sys
from llm_economist.main import run_simulation, create_argument_parser, generate_experiment_name


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from llm_economist.main import run_simulation, create_argument_parser
        from llm_economist.agents.worker import Worker
        from llm_economist.agents.planner import TaxPlanner
        from llm_economist.agents.llm_agent import TestAgent
        from llm_economist.utils.common import distribute_agents
        from llm_economist.agents.worker import distribute_personas
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_argument_parser():
    """Test that argument parser works correctly."""
    print("Testing argument parser...")
    
    try:
        parser = create_argument_parser()
        # Test with minimal arguments
        args = parser.parse_args([
            "--scenario", "rational",
            "--num-agents", "3",
            "--max-timesteps", "5",
            "--worker-type", "LLM",
            "--planner-type", "LLM",
            "--llm", "gpt-4o-mini"
        ])
        
        assert args.scenario == "rational"
        assert args.num_agents == 3
        assert args.max_timesteps == 5
        assert args.worker_type == "LLM"
        assert args.planner_type == "LLM"
        assert args.llm == "gpt-4o-mini"
        
        print("✓ Argument parser working correctly")
        return True
    except Exception as e:
        print(f"✗ Argument parser error: {e}")
        return False


def test_experiment_name_generation():
    """Test experiment name generation."""
    print("Testing experiment name generation...")
    
    try:
        class Args:
            scenario = "rational"
            num_agents = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            two_timescale = 25
            history_len = 50
            max_timesteps = 100
            bracket_setting = "two"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            platforms = False
        
        args = Args()
        name = generate_experiment_name(args)
        
        # Check that name contains expected components
        expected_parts = ["rational", "a5", "w-LLM", "p-LLM", "llm-g"]
        for part in expected_parts:
            assert part in name, f"Expected '{part}' in experiment name '{name}'"
        
        print(f"✓ Experiment name generation working: {name}")
        return True
    except Exception as e:
        print(f"✗ Experiment name generation error: {e}")
        return False


def test_api_key_detection():
    """Test API key detection for different services."""
    print("Testing API key detection...")
    
    api_keys = {
        "OpenAI": os.getenv('OPENAI_API_KEY') or os.getenv('ECON_OPENAI'),
        "OpenRouter": os.getenv('OPENROUTER_API_KEY'),
        "Gemini": os.getenv('GEMINI_API_KEY'),
    }
    
    found_keys = []
    for service, key in api_keys.items():
        if key:
            found_keys.append(service)
            print(f"✓ {service} API key found")
        else:
            print(f"- {service} API key not found")
    
    if found_keys:
        print(f"✓ Found API keys for: {', '.join(found_keys)}")
        return True
    else:
        print("- No API keys found (this is okay for testing)")
        return True


def test_basic_args_creation():
    """Test creating basic argument objects."""
    print("Testing basic Args object creation...")
    
    try:
        class Args:
            scenario = "rational"
            num_agents = 3
            max_timesteps = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            port = 8000
            service = "vllm"
            use_openrouter = False
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 10
            agent_mix = "us_income"
            bracket_setting = "three"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            tax_type = "US_FED"
            warmup = 0
            wandb = False
            debug = False
            use_multithreading = False
            platforms = False
            name = ""
            log_dir = "logs"
            elasticity = [0.4]
            seed = 42
        
        args = Args()
        
        # Verify all required attributes exist
        required_attrs = [
            'scenario', 'num_agents', 'max_timesteps', 'worker_type', 
            'planner_type', 'llm', 'agent_mix', 'bracket_setting',
            'percent_ego', 'percent_alt', 'percent_adv', 'tax_type'
        ]
        
        for attr in required_attrs:
            assert hasattr(args, attr), f"Missing required attribute: {attr}"
        
        print("✓ Basic Args object creation successful")
        return True
    except Exception as e:
        print(f"✗ Args object creation error: {e}")
        return False


def test_service_configurations():
    """Test different service configurations."""
    print("Testing service configurations...")
    
    configurations = [
        {"service": "vllm", "port": 8000, "use_openrouter": False},
        {"service": "ollama", "port": 11434, "use_openrouter": False},
        {"service": "vllm", "port": 8000, "use_openrouter": True},
    ]
    
    for config in configurations:
        try:
            class Args:
                scenario = "rational"
                num_agents = 3
                max_timesteps = 5
                worker_type = "LLM"
                planner_type = "LLM"
                llm = "gpt-4o-mini"
                port = config["port"]
                service = config["service"]
                use_openrouter = config["use_openrouter"]
                prompt_algo = "io"
                history_len = 20
                timeout = 10
                two_timescale = 10
                agent_mix = "us_income"
                bracket_setting = "three"
                percent_ego = 100
                percent_alt = 0
                percent_adv = 0
                tax_type = "US_FED"
                warmup = 0
                wandb = False
                debug = False
                use_multithreading = False
                platforms = False
                name = ""
                log_dir = "logs"
                elasticity = [0.4]
                seed = 42
            
            args = Args()
            print(f"✓ Configuration valid: {config['service']} on port {config['port']}")
        except Exception as e:
            print(f"✗ Configuration error for {config}: {e}")
            return False
    
    print("✓ All service configurations valid")
    return True


def run_all_tests():
    """Run all basic functionality tests."""
    print("="*50)
    print("Running LLM Economist Quick Start Tests")
    print("="*50)
    
    tests = [
        test_imports,
        test_argument_parser,
        test_experiment_name_generation,
        test_api_key_detection,
        test_basic_args_creation,
        test_service_configurations,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n{test.__name__}:")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*50)
    
    if failed == 0:
        print("🎉 All basic functionality tests passed!")
        print("For actual simulation examples, run: python examples/advanced_usage.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return failed == 0


def main():
    """Main entry point for quick start tests."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        print("\nUsage:")
        print("  python examples/quick_start.py          # Run all basic tests")
        print("  python examples/quick_start.py --help   # Show this help")
        print("\nFor actual simulation examples:")
        print("  python examples/advanced_usage.py --help")
        return
    
    success = run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 