"""
Simple test cases for the generic ReplayProvider implementation.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llama_stack.providers.utils.replay_provider import (
    ReplayMode,
    create_replay_provider_class,
)


# Mock Protocol for testing
class MockInferenceProvider:
    """Mock protocol similar to InferenceProvider for testing."""

    async def completion(self, model_id: str, content: str) -> dict:
        """Mock completion method."""
        pass

    # Add __webmethod__ attribute to simulate real protocol
    completion.__webmethod__ = type("WebMethod", (), {"route": "/completion", "method": "POST"})()


class MockRealProvider:
    """Mock real provider implementation."""

    def __init__(self):
        self.call_count = 0

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def completion(self, model_id: str, content: str) -> dict:
        """Mock completion that returns predictable responses."""
        self.call_count += 1
        return {"content": f"Response to: {content}", "model": model_id, "call_number": self.call_count}


@pytest.fixture
def temp_cache_dir():
    """Provide a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_real_provider():
    """Provide a mock real provider."""
    return MockRealProvider()


@pytest.fixture
async def replay_provider_class():
    """Create ReplayProvider class for MockInferenceProvider."""
    return create_replay_provider_class(MockInferenceProvider)


class TestReplayProvider:
    """Test cases for ReplayProvider functionality."""

    async def test_cache_miss_with_fallback(self, replay_provider_class, mock_real_provider, temp_cache_dir):
        """Test cache miss calls real provider and caches result."""
        replay_provider = replay_provider_class(
            real_provider=mock_real_provider, mode=ReplayMode.CACHE_WITH_FALLBACK, cache_dir=temp_cache_dir
        )

        # First call should be cache miss, call real provider
        response1 = await replay_provider.completion(model_id="test-model", content="Hello world")

        assert response1["content"] == "Response to: Hello world"
        assert response1["model"] == "test-model"
        assert response1["call_number"] == 1
        assert mock_real_provider.call_count == 1

        # Verify cache file was created
        cache_files = list(Path(temp_cache_dir).rglob("*.json"))
        assert len(cache_files) == 1

    async def test_cache_hit(self, replay_provider_class, mock_real_provider, temp_cache_dir):
        """Test cache hit returns cached response without calling real provider."""
        replay_provider = replay_provider_class(
            real_provider=mock_real_provider, mode=ReplayMode.CACHE_WITH_FALLBACK, cache_dir=temp_cache_dir
        )

        # First call - cache miss
        response1 = await replay_provider.completion(model_id="test-model", content="Hello world")
        assert mock_real_provider.call_count == 1

        # Second call with same parameters - should be cache hit
        response2 = await replay_provider.completion(model_id="test-model", content="Hello world")

        # Should return same response without calling real provider again
        assert response2 == response1
        assert mock_real_provider.call_count == 1  # No additional calls

    async def test_cache_only_mode_with_miss(self, replay_provider_class, mock_real_provider, temp_cache_dir):
        """Test CACHE_ONLY mode raises error on cache miss."""
        replay_provider = replay_provider_class(
            real_provider=mock_real_provider, mode=ReplayMode.CACHE_ONLY, cache_dir=temp_cache_dir
        )

        # Should raise error since cache is empty and mode is CACHE_ONLY
        with pytest.raises(RuntimeError, match="Cache miss .* and mode is CACHE_ONLY"):
            await replay_provider.completion(model_id="test-model", content="Hello world")

        # Real provider should never be called
        assert mock_real_provider.call_count == 0

    async def test_different_parameters_different_cache(
        self, replay_provider_class, mock_real_provider, temp_cache_dir
    ):
        """Test different parameters result in different cache entries."""
        replay_provider = replay_provider_class(
            real_provider=mock_real_provider, mode=ReplayMode.CACHE_WITH_FALLBACK, cache_dir=temp_cache_dir
        )

        # Call with first set of parameters
        response1 = await replay_provider.completion(model_id="test-model", content="First message")

        # Call with different parameters
        response2 = await replay_provider.completion(model_id="test-model", content="Second message")

        # Both should call real provider (different cache keys)
        assert mock_real_provider.call_count == 2
        assert response1["content"] == "Response to: First message"
        assert response2["content"] == "Response to: Second message"

        # Should have created 2 cache files
        cache_files = list(Path(temp_cache_dir).rglob("*.json"))
        assert len(cache_files) == 2


# Simple integration test to verify the pattern works
async def test_basic_replay_provider_creation():
    """Test that we can create a ReplayProvider for any protocol."""
    # This is the key test - can we dynamically create providers?
    InferenceReplayProvider = create_replay_provider_class(MockInferenceProvider)

    # Verify the class was created with correct name
    assert InferenceReplayProvider.__name__ == "MockInferenceProviderReplayProvider"

    # Verify it has the expected methods
    assert hasattr(InferenceReplayProvider, "completion")
    assert hasattr(InferenceReplayProvider, "__init__")
    assert hasattr(InferenceReplayProvider, "initialize")
    assert hasattr(InferenceReplayProvider, "shutdown")


if __name__ == "__main__":
    # Simple test runner for development
    async def run_basic_test():
        await test_basic_replay_provider_creation()
        print("✓ Basic ReplayProvider creation works!")

    asyncio.run(run_basic_test())
