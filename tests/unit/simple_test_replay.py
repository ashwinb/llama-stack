#!/usr/bin/env python3
"""
Simple synchronous test for ReplayProvider to validate basic functionality.
"""

import asyncio
import tempfile
from pathlib import Path

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


async def test_basic_functionality():
    """Test basic ReplayProvider functionality."""
    print("🧪 Testing ReplayProvider Basic Functionality")
    print("=" * 50)

    # Create ReplayProvider class
    InferenceReplayProvider = create_replay_provider_class(MockInferenceProvider)
    print(f"✓ Created class: {InferenceReplayProvider.__name__}")

    # Create mock real provider
    mock_real_provider = MockRealProvider()

    # Create temp cache directory
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        print(f"✓ Using temp cache: {temp_cache_dir}")

        # Create replay provider
        replay_provider = InferenceReplayProvider(
            real_provider=mock_real_provider, mode=ReplayMode.CACHE_WITH_FALLBACK, cache_dir=temp_cache_dir
        )
        print("✓ Created ReplayProvider instance")

        # Test 1: Cache miss -> real provider call
        print("\n📋 Test 1: Cache miss with fallback")
        response1 = await replay_provider.completion(model_id="test-model", content="Hello world")
        print(f"  Response: {response1}")
        print(f"  Real provider call count: {mock_real_provider.call_count}")
        assert response1["content"] == "Response to: Hello world"
        assert response1["model"] == "test-model"
        assert mock_real_provider.call_count == 1
        print("  ✅ Cache miss test passed")

        # Verify cache file was created
        cache_files = list(Path(temp_cache_dir).rglob("*.json"))
        print(f"  Cache files created: {len(cache_files)}")
        assert len(cache_files) == 1
        print("  ✅ Cache file creation verified")

        # Test 2: Cache hit -> no real provider call
        print("\n📋 Test 2: Cache hit")
        response2 = await replay_provider.completion(model_id="test-model", content="Hello world")
        print(f"  Response: {response2}")
        print(f"  Real provider call count: {mock_real_provider.call_count}")
        assert response2 == response1
        assert mock_real_provider.call_count == 1  # No additional calls
        print("  ✅ Cache hit test passed")

        # Test 3: Different parameters -> different cache
        print("\n📋 Test 3: Different parameters")
        response3 = await replay_provider.completion(model_id="test-model", content="Different message")
        print(f"  Response: {response3}")
        print(f"  Real provider call count: {mock_real_provider.call_count}")
        assert response3["content"] == "Response to: Different message"
        assert mock_real_provider.call_count == 2
        print("  ✅ Different parameters test passed")

        # Verify 2 cache files now exist
        cache_files = list(Path(temp_cache_dir).rglob("*.json"))
        print(f"  Total cache files: {len(cache_files)}")
        assert len(cache_files) == 2
        print("  ✅ Multiple cache files verified")


async def test_cache_only_mode():
    """Test CACHE_ONLY mode."""
    print("\n🧪 Testing CACHE_ONLY Mode")
    print("=" * 30)

    InferenceReplayProvider = create_replay_provider_class(MockInferenceProvider)
    mock_real_provider = MockRealProvider()

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        replay_provider = InferenceReplayProvider(
            real_provider=mock_real_provider, mode=ReplayMode.CACHE_ONLY, cache_dir=temp_cache_dir
        )

        # Should raise error on cache miss
        try:
            await replay_provider.completion(model_id="test-model", content="Hello world")
            assert False, "Expected RuntimeError was not raised"
        except RuntimeError as e:
            print(f"  ✅ Expected error: {e}")
            assert "Cache miss" in str(e)
            assert "CACHE_ONLY" in str(e)

        # Real provider should never be called
        assert mock_real_provider.call_count == 0
        print("  ✅ Real provider was not called")


async def main():
    """Run all tests."""
    try:
        await test_basic_functionality()
        await test_cache_only_mode()

        print("\n🎉 All tests passed!")
        print("\n✨ ReplayProvider is working correctly:")
        print("  - Dynamic protocol implementation ✓")
        print("  - Cache miss/hit behavior ✓")
        print("  - Multiple cache entries ✓")
        print("  - CACHE_ONLY mode ✓")
        print("  - File-based caching ✓")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🚀 Ready for next phase of development!")
    else:
        exit(1)
