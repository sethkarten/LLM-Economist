"""
Integration tests for advanced usage scenarios.

These tests run actual simulations with real API calls to validate
the complete LLM Economist functionality.
"""

import pytest
import os
from unittest.mock import patch


class TestAdvancedUsageIntegration:
    """Integration tests for advanced usage scenarios."""
    
    def test_advanced_usage_imports(self):
        """Test that advanced usage functions can be imported."""
        from examples.advanced_usage import (
            test_rational_openai,
            test_openrouter_rational,
            test_vllm_rational,
            test_ollama_rational,
            test_gemini_rational,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers,
            main
        )
        
        # Verify they're callable
        assert callable(test_rational_openai)
        assert callable(test_openrouter_rational)
        assert callable(test_vllm_rational)
        assert callable(test_ollama_rational)
        assert callable(test_gemini_rational)
        assert callable(test_bounded_rationality)
        assert callable(test_democratic_scenario)
        assert callable(test_fixed_workers)
        assert callable(main)
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    def test_rational_openai_simulation(self):
        """Test rational scenario with OpenAI (requires API key)."""
        from examples.advanced_usage import test_rational_openai
        
        # This should run without errors if API key is available
        try:
            test_rational_openai()
        except Exception as e:
            # Allow for API rate limits or temporary failures
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise
    
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OpenRouter API key not available")
    def test_rational_openrouter_simulation(self):
        """Test rational scenario with OpenRouter (requires API key)."""
        from examples.advanced_usage import test_openrouter_rational
        
        # This should run without errors if API key is available
        try:
            test_openrouter_rational()
        except Exception as e:
            # Allow for API rate limits or temporary failures
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise
    
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Gemini API key not available")
    def test_rational_gemini_simulation(self):
        """Test rational scenario with Gemini (requires API key)."""
        from examples.advanced_usage import test_gemini_rational
        
        # This should run without errors if API key is available
        try:
            test_gemini_rational()
        except Exception as e:
            # Allow for API rate limits or temporary failures
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    def test_bounded_rationality_simulation(self):
        """Test bounded rationality scenario (requires API key)."""
        from examples.advanced_usage import test_bounded_rationality
        
        # This should run without errors if API key is available
        try:
            test_bounded_rationality()
        except Exception as e:
            # Allow for API rate limits or temporary failures
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    def test_democratic_scenario_simulation(self):
        """Test democratic scenario (requires API key)."""
        from examples.advanced_usage import test_democratic_scenario
        
        # This should run without errors if API key is available
        try:
            test_democratic_scenario()
        except Exception as e:
            # Allow for API rate limits or temporary failures
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                pytest.skip(f"API rate limit or quota exceeded: {e}")
            else:
                raise
    
    def test_fixed_workers_simulation(self):
        """Test fixed workers scenario (no API key required)."""
        from examples.advanced_usage import test_fixed_workers
        
        # This should always work as it uses fixed agents
        test_fixed_workers()
    
    def test_vllm_simulation_requires_server(self):
        """Test that vLLM scenario properly handles server connection."""
        from examples.advanced_usage import test_vllm_rational
        
        # This should fail gracefully if no vLLM server is running
        with pytest.raises(Exception) as exc_info:
            test_vllm_rational()
        
        # Should get connection error, not other types of errors
        assert any(keyword in str(exc_info.value).lower() for keyword in 
                  ["connection", "refused", "unreachable", "timeout"])
    
    def test_ollama_simulation_requires_server(self):
        """Test that Ollama scenario properly handles server connection."""
        from examples.advanced_usage import test_ollama_rational
        
        # This should fail gracefully if no Ollama server is running
        with pytest.raises(Exception) as exc_info:
            test_ollama_rational()
        
        # Should get connection error, not other types of errors
        assert any(keyword in str(exc_info.value).lower() for keyword in 
                  ["connection", "refused", "unreachable", "timeout"])


class TestAdvancedUsageCommandLine:
    """Test command line interface for advanced usage."""
    
    def test_help_command(self):
        """Test that help command works."""
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, "examples/advanced_usage.py", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "scenarios" in result.stdout.lower() or "commands" in result.stdout.lower()
    
    def test_invalid_scenario(self):
        """Test that invalid scenario is handled properly."""
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, "examples/advanced_usage.py", "invalid_scenario"
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "invalid scenario" in result.stderr.lower() or "unknown command" in result.stdout.lower()
    
    def test_list_scenarios(self):
        """Test that all scenarios are listed in help."""
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, "examples/advanced_usage.py", "--help"
        ], capture_output=True, text=True)
        
        expected_scenarios = [
            "rational", "openrouter", "vllm", "ollama", 
            "gemini", "bounded", "democratic", "fixed"
        ]
        
        for scenario in expected_scenarios:
            assert scenario in result.stdout.lower()


class TestAdvancedUsageConfiguration:
    """Test configuration and setup for advanced usage scenarios."""
    
    def test_all_scenarios_have_20_timesteps(self):
        """Test that all scenarios are configured for 20 timesteps."""
        from examples.advanced_usage import (
            test_rational_openai,
            test_openrouter_rational,
            test_vllm_rational,
            test_ollama_rational,
            test_gemini_rational,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers
        )
        
        # Use reflection to check Args classes in each function
        import inspect
        
        functions = [
            test_rational_openai,
            test_openrouter_rational,
            test_vllm_rational,
            test_ollama_rational,
            test_gemini_rational,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers
        ]
        
        for func in functions:
            source = inspect.getsource(func)
            # Check that max_timesteps = 20 is in the source
            assert "max_timesteps = 20" in source, f"Function {func.__name__} doesn't have max_timesteps = 20"
    
    def test_scenario_diversity(self):
        """Test that different scenarios have different configurations."""
        from examples.advanced_usage import (
            test_rational_openai,
            test_bounded_rationality,
            test_democratic_scenario,
            test_fixed_workers
        )
        
        # Check that different scenarios have different configurations
        # This is a basic sanity check
        import inspect
        
        rational_source = inspect.getsource(test_rational_openai)
        bounded_source = inspect.getsource(test_bounded_rationality)
        democratic_source = inspect.getsource(test_democratic_scenario)
        fixed_source = inspect.getsource(test_fixed_workers)
        
        # Rational should have scenario = "rational"
        assert 'scenario = "rational"' in rational_source
        
        # Bounded should have scenario = "bounded"
        assert 'scenario = "bounded"' in bounded_source
        
        # Democratic should have scenario = "democratic"
        assert 'scenario = "democratic"' in democratic_source
        
        # Fixed should have worker_type = "FIXED"
        assert 'worker_type = "FIXED"' in fixed_source
        assert 'planner_type = "FIXED"' in fixed_source 