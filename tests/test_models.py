"""
Tests for LLM model implementations.
"""

import pytest
import os
from unittest.mock import Mock, patch
from llm_economist.models.base import BaseLLMModel
from llm_economist.models.openai_model import OpenAIModel
from llm_economist.models.vllm_model import VLLMModel, OllamaModel
from llm_economist.models.openrouter_model import OpenRouterModel
from llm_economist.models.gemini_model import GeminiModel


class TestBaseLLMModel:
    """Test the base LLM model class."""
    
    def test_base_model_abstract(self):
        """Test that BaseLLMModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMModel("test-model")
    
    def test_json_extraction(self):
        """Test JSON extraction functionality."""
        class TestModel(BaseLLMModel):
            def send_msg(self, system_prompt, user_prompt, temperature=None, json_format=False):
                return "test", False
        
        model = TestModel("test-model")
        
        # Test valid JSON
        valid_json = '{"key": "value", "number": 42}'
        result, is_valid = model._extract_json(f"Some text {valid_json} more text")
        assert is_valid
        assert result == valid_json
        
        # Test invalid JSON
        invalid_json = '{"key": "value"'
        result, is_valid = model._extract_json(f"Some text {invalid_json} more text")
        assert not is_valid
        
        # Test no JSON
        no_json = "This is just regular text"
        result, is_valid = model._extract_json(no_json)
        assert not is_valid


class TestOpenAIModel:
    """Test OpenAI model implementation."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel()
            assert model.model_name == "gpt-4o-mini"
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIModel()
    
    @patch('llm_economist.models.openai_model.OpenAI')
    def test_send_msg_success(self, mock_openai):
        """Test successful message sending."""
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel()
            response, is_json = model.send_msg("System prompt", "User prompt")
            
            assert response == "Test response"
            assert not is_json
            mock_client.chat.completions.create.assert_called_once()
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = OpenAIModel.get_available_models()
        assert isinstance(models, list)
        assert "gpt-4o-mini" in models


class TestVLLMModel:
    """Test vLLM model implementation."""
    
    @patch('llm_economist.models.vllm_model.OpenAI')
    def test_init(self, mock_openai):
        """Test vLLM model initialization."""
        model = VLLMModel()
        assert model.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert model.base_url == "http://localhost:8000"
    
    def test_model_mapping(self):
        """Test model name mapping."""
        with patch('llm_economist.models.vllm_model.OpenAI'):
            model = VLLMModel(model_name="llama3:8b")
            assert model.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    
    @patch('llm_economist.models.vllm_model.requests.get')
    def test_health_check(self, mock_get):
        """Test health check functionality."""
        with patch('llm_economist.models.vllm_model.OpenAI'):
            model = VLLMModel()
            
            # Test healthy server
            mock_get.return_value.status_code = 200
            assert model.check_health()
            
            # Test unhealthy server
            mock_get.side_effect = Exception("Connection error")
            assert not model.check_health()


class TestOllamaModel:
    """Test Ollama model implementation."""
    
    def test_init_missing_ollama(self):
        """Test initialization when ollama package is missing."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(ImportError, match="Please install ollama"):
                OllamaModel()


class TestOpenRouterModel:
    """Test OpenRouter model implementation."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            model = OpenRouterModel()
            assert model.api_key == "test-key"
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key not found"):
                OpenRouterModel()
    
    def test_get_popular_models(self):
        """Test getting popular models."""
        models = OpenRouterModel.get_popular_models()
        assert isinstance(models, list)
        assert "meta-llama/llama-3.1-8b-instruct" in models
    
    @patch('llm_economist.models.openrouter_model.requests.post')
    def test_send_msg_success(self, mock_post):
        """Test successful message sending."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            model = OpenRouterModel()
            response, is_json = model.send_msg("System prompt", "User prompt")
            
            assert response == "Test response"
            assert not is_json


class TestGeminiModel:
    """Test Gemini model implementation."""
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key not found"):
                GeminiModel()
    
    def test_init_missing_google_ai(self):
        """Test initialization when google.generativeai package is missing."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(ImportError, match="Please install Google AI SDK"):
                    GeminiModel()
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = GeminiModel.get_available_models()
        assert isinstance(models, list)
        assert "gemini-1.5-flash" in models


class TestModelIntegration:
    """Integration tests for model switching."""
    
    def test_model_factory_pattern(self):
        """Test that models can be created through a factory pattern."""
        # This simulates how the LLMAgent creates models
        
        def create_model(model_type, **kwargs):
            if "gpt" in model_type.lower():
                return "OpenAI"
            elif "llama" in model_type.lower():
                return "vLLM"
            elif "claude" in model_type.lower():
                return "OpenRouter"
            elif "gemini" in model_type.lower():
                return "Gemini"
            else:
                return "Unknown"
        
        assert create_model("gpt-4o-mini") == "OpenAI"
        assert create_model("llama3:8b") == "vLLM"
        assert create_model("claude-3.5-sonnet") == "OpenRouter"
        assert create_model("gemini-1.5-flash") == "Gemini"


# Fixtures for testing
@pytest.fixture
def mock_openai_env():
    """Fixture that provides OpenAI API key in environment."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_openrouter_env():
    """Fixture that provides OpenRouter API key in environment."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_google_env():
    """Fixture that provides Google API key in environment."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        yield 