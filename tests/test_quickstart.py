"""
Test the quickstart functionality and advanced usage examples.
"""

import pytest
import os
from unittest.mock import patch


class TestQuickStart:
    """Test basic quickstart functionality."""
    
    def test_quickstart_imports(self):
        """Test that quickstart functions can be imported."""
        from examples.quick_start import (
            test_imports,
            test_argument_parser,
            test_experiment_name_generation,
            test_api_key_detection,
            test_basic_args_creation,
            test_service_configurations,
            run_all_tests,
            main
        )
        
        # Verify they're callable
        assert callable(test_imports)
        assert callable(test_argument_parser)
        assert callable(test_experiment_name_generation)
        assert callable(test_api_key_detection)
        assert callable(test_basic_args_creation)
        assert callable(test_service_configurations)
        assert callable(run_all_tests)
        assert callable(main)
    
    def test_basic_functionality(self):
        """Test that basic functionality tests work."""
        from examples.quick_start import (
            test_imports,
            test_argument_parser,
            test_experiment_name_generation,
            test_basic_args_creation,
            test_service_configurations
        )
        
        # Run each test
        assert test_imports() == True
        assert test_argument_parser() == True
        assert test_experiment_name_generation() == True
        assert test_basic_args_creation() == True
        assert test_service_configurations() == True
    
    def test_api_key_detection(self):
        """Test API key detection (should always pass)."""
        from examples.quick_start import test_api_key_detection
        
        # This should always return True (even if no keys found)
        assert test_api_key_detection() == True

 