"""
Advanced usage examples for LLM Economist.

This script demonstrates actual simulation runs with different scenarios.
All simulations run for 20 timesteps for testing purposes.
"""

import os
import sys
from llm_economist.main import run_simulation


def test_rational_openai():
    """Test rational scenario with OpenAI GPT-4o-mini."""
    print("Running rational scenario with OpenAI GPT-4o-mini...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
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
    
    # Make sure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    try:
        run_simulation(args)
        print("✓ Rational scenario simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Rational scenario simulation failed: {e}")
        return False


def test_bounded_rationality():
    """Test bounded rationality scenario."""
    print("Running bounded rationality scenario...")
    
    class Args:
        scenario = "bounded"
        num_agents = 3
        max_timesteps = 20
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
        percent_ego = 100  # Use 100% egotistical since personas only support this
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
    
    # Make sure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("Please set OPENAI_API_KEY environment variable")
        return False
    
    try:
        run_simulation(args)
        print("✓ Bounded rationality simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Bounded rationality simulation failed: {e}")
        return False


def test_democratic_scenario():
    """Test democratic voting scenario."""
    print("Running democratic voting scenario...")
    
    class Args:
        scenario = "democratic"
        num_agents = 3
        max_timesteps = 20
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
        percent_ego = 100  # Use 100% egotistical since personas only support this
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
    
    # Make sure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("Please set OPENAI_API_KEY environment variable")
        return False
    
    try:
        run_simulation(args)
        print("✓ Democratic scenario simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Democratic scenario simulation failed: {e}")
        return False


def test_fixed_workers():
    """Test fixed workers scenario."""
    print("Running fixed workers scenario...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "FIXED"
        planner_type = "FIXED"
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
    
    # Make sure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ECON_OPENAI'):
        print("Please set OPENAI_API_KEY environment variable")
        return False
    
    try:
        run_simulation(args)
        print("✓ Fixed workers simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Fixed workers simulation failed: {e}")
        return False


def test_openrouter_rational():
    """Test rational scenario with OpenRouter."""
    print("Running rational scenario with OpenRouter...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "meta-llama/llama-3.1-8b-instruct"
        port = 8000
        service = "vllm"
        use_openrouter = True
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
    
    # Make sure OpenRouter API key is set
    if not os.getenv('OPENROUTER_API_KEY'):
        print("Please set OPENROUTER_API_KEY environment variable")
        print("Example: export OPENROUTER_API_KEY=your_api_key_here")
        return False
    
    try:
        run_simulation(args)
        print("✓ OpenRouter rational simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ OpenRouter rational simulation failed: {e}")
        return False


def test_vllm_rational():
    """Test rational scenario with local vLLM server."""
    print("Running rational scenario with local vLLM...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "meta-llama/Llama-3.1-8B-Instruct"
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
    
    print("Make sure you have a vLLM server running on port 8000")
    print("Example: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000")
    
    try:
        run_simulation(args)
        print("✓ vLLM rational simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ vLLM rational simulation failed: {e}")
        return False


def test_ollama_rational():
    """Test rational scenario with Ollama."""
    print("Running rational scenario with Ollama...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "llama3.1:8b"
        port = 11434
        service = "ollama"
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
    
    print("Make sure you have Ollama running with llama3.1:8b model")
    print("Example: ollama run llama3.1:8b")
    
    try:
        run_simulation(args)
        print("✓ Ollama rational simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Ollama rational simulation failed: {e}")
        return False


def test_gemini_rational():
    """Test rational scenario with Google Gemini."""
    print("Running rational scenario with Google Gemini...")
    
    class Args:
        scenario = "rational"
        num_agents = 3
        max_timesteps = 20
        worker_type = "LLM"
        planner_type = "LLM"
        llm = "gemini-1.5-flash"
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
    
    # Make sure Google AI API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("Please set GOOGLE_API_KEY environment variable")
        print("Example: export GOOGLE_API_KEY=your_api_key_here")
        return False
    
    try:
        run_simulation(args)
        print("✓ Gemini rational simulation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Gemini rational simulation failed: {e}")
        return False


def run_all_scenario_tests():
    """Run all scenario tests."""
    print("="*60)
    print("Running LLM Economist Advanced Usage Tests")
    print("All simulations run for 20 timesteps")
    print("="*60)
    
    # Core scenario tests (require OpenAI API key)
    core_tests = [
        ("Rational Scenario", test_rational_openai),
        ("Bounded Rationality", test_bounded_rationality),
        ("Democratic Voting", test_democratic_scenario),
        ("Fixed Workers", test_fixed_workers),
    ]
    
    # Additional service tests (require respective API keys)
    service_tests = [
        ("OpenRouter", test_openrouter_rational),
        ("vLLM", test_vllm_rational),
        ("Ollama", test_ollama_rational),
        ("Gemini", test_gemini_rational),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("\n" + "="*40)
    print("CORE SCENARIO TESTS")
    print("="*40)
    
    for test_name, test_func in core_tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*40)
    print("ADDITIONAL SERVICE TESTS")
    print("="*40)
    
    for test_name, test_func in service_tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                skipped += 1  # Count as skipped if API key missing
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60)
    
    if failed == 0:
        print("🎉 All available tests passed!")
        if skipped > 0:
            print(f"Note: {skipped} tests were skipped due to missing API keys")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return failed == 0


def main():
    """Main entry point for advanced usage examples."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--help" or command == "-h":
            print(__doc__)
            print("\nUsage:")
            print("  python examples/advanced_usage.py                    # Run all tests")
            print("  python examples/advanced_usage.py rational          # Test rational scenario")
            print("  python examples/advanced_usage.py bounded           # Test bounded rationality")
            print("  python examples/advanced_usage.py democratic        # Test democratic voting")
            print("  python examples/advanced_usage.py fixed             # Test fixed workers")
            print("  python examples/advanced_usage.py openrouter        # Test OpenRouter")
            print("  python examples/advanced_usage.py vllm              # Test vLLM")
            print("  python examples/advanced_usage.py ollama            # Test Ollama")
            print("  python examples/advanced_usage.py gemini            # Test Gemini")
            print("  python examples/advanced_usage.py --help            # Show this help")
            print("\nAll simulations run for 20 timesteps for testing purposes.")
            return
        
        # Run individual tests
        test_map = {
            "rational": test_rational_openai,
            "bounded": test_bounded_rationality,
            "democratic": test_democratic_scenario,
            "fixed": test_fixed_workers,
            "openrouter": test_openrouter_rational,
            # "vllm": test_vllm_rational,
            # "ollama": test_ollama_rational,
            # "gemini": test_gemini_rational,
        }
        
        if command in test_map:
            success = test_map[command]()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown command: {command}")
            print("Use --help to see available commands")
            sys.exit(1)
    
    # Run all tests
    success = run_all_scenario_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 