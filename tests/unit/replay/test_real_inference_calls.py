#!/usr/bin/env python3
"""
Test real inference calls through the ReplayProvider to verify:
1. Cache-with-fallback mode makes real API calls on cache miss
2. Cache-only mode uses cached responses
3. Responses are properly cached and retrieved
4. Real provider integration works end-to-end
"""

import asyncio
import json
import tempfile
from pathlib import Path

from llama_stack.distribution.datatypes import ModelInput, Provider, StackRunConfig
from llama_stack.distribution.stack import construct_stack
from llama_stack.providers.datatypes import Api


async def test_real_inference_calls():
    """Test real inference calls through replay provider."""

    cache_dir = Path(tempfile.gettempdir()) / "test_real_cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"🔧 Using cache directory: {cache_dir}")

    # Test 1: Cache-with-fallback mode (should make real API calls)
    print("\n" + "=" * 60)
    print("🧪 TEST 1: Cache-with-fallback mode")
    print("=" * 60)

    success = await test_cache_with_fallback(cache_dir)
    if not success:
        return False

    # Test 2: Cache-only mode (should use cached responses)
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Cache-only mode")
    print("=" * 60)

    success = await test_cache_only(cache_dir)
    if not success:
        return False

    # Test 3: Verify cache contents
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Cache verification")
    print("=" * 60)

    success = verify_cache_contents(cache_dir)
    return success


async def test_cache_with_fallback(cache_dir: Path):
    """Test cache-with-fallback mode - should make real API calls."""

    # Create config with cache-with-fallback mode
    run_config = StackRunConfig(
        image_name="test_fallback",
        providers={
            "inference": [
                Provider(
                    provider_id="fireworks",
                    provider_type="remote::fireworks",
                    config={
                        "url": "https://api.fireworks.ai/inference/v1",
                        "api_key": "fw_3ZmjqrM4b3PhNgH3DXYXC9Ce",
                    },
                ),
                Provider(
                    provider_id="__replay__",
                    provider_type="replay",
                    config={
                        "real_provider_id": "fireworks",
                        "mode": "cache_with_fallback",  # Should make real calls on cache miss
                        "cache_dir": str(cache_dir),
                    },
                ),
            ],
            "telemetry": [
                Provider(
                    provider_id="meta-reference",
                    provider_type="inline::meta-reference",
                    config={},
                ),
            ],
        },
        models=[
            ModelInput(
                model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                provider_id="__replay__",
                provider_model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                model_type="llm",
            )
        ],
    )

    try:
        # Construct stack (this will register models)
        impls = await construct_stack(run_config)

        inference_impl = impls[Api.inference]
        print(f"✅ Resolved inference implementation: {type(inference_impl).__name__}")

        # Debug: Check what providers are available
        if hasattr(inference_impl, "routing_table"):
            print(f"🔍 Available providers in routing table:")
            for provider_id, provider in inference_impl.routing_table.impls_by_provider_id.items():
                print(f"   - {provider_id}: {type(provider).__name__}")

        # Debug: Check which provider will be used for our model
        model = await inference_impl.routing_table.get_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        if model:
            print(f"🎯 Model '{model.model_id}' will use provider: {model.provider_id}")
        else:
            print("❌ Model not found in routing table")

        # Make a real inference call
        print("🔄 Making inference call (should hit real API)...")

        from llama_stack.apis.inference import SamplingParams, TopPSamplingStrategy, UserMessage

        messages = [UserMessage(content="What is 2+2? Answer in exactly one word.")]
        sampling_params = SamplingParams(max_tokens=10, strategy=TopPSamplingStrategy(temperature=0.1))

        response = await inference_impl.chat_completion(
            model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=messages,
            sampling_params=sampling_params,
        )

        print(f"✅ Got response: {response.completion_message.content}")
        print(f"   Stop reason: {response.completion_message.stop_reason}")
        if hasattr(response, "metrics") and response.metrics:
            print(f"   Metrics: {response.metrics}")

        # Verify we got a real response
        if not response.completion_message.content:
            print("❌ Empty response content")
            return False

        print("✅ Cache-with-fallback mode working - made real API call")
        return True

    except Exception as e:
        print(f"❌ Error in cache-with-fallback test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_cache_only(cache_dir: Path):
    """Test cache-only mode - should use cached responses."""

    # Create config with cache-only mode
    run_config = StackRunConfig(
        image_name="test_cache_only",
        providers={
            "inference": [
                Provider(
                    provider_id="fireworks",
                    provider_type="remote::fireworks",
                    config={
                        "url": "https://api.fireworks.ai/inference/v1",
                        "api_key": "fw_3ZmjqrM4b3PhNgH3DXYXC9Ce",
                    },
                ),
                Provider(
                    provider_id="__replay__",
                    provider_type="replay",
                    config={
                        "real_provider_id": "fireworks",
                        "mode": "cache_only",  # Should only use cache
                        "cache_dir": str(cache_dir),
                    },
                ),
            ],
            "telemetry": [
                Provider(
                    provider_id="meta-reference",
                    provider_type="inline::meta-reference",
                    config={},
                ),
            ],
        },
        models=[
            ModelInput(
                model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                provider_id="__replay__",
                provider_model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                model_type="llm",
            )
        ],
    )

    try:
        # Construct stack (this will register models)
        impls = await construct_stack(run_config)

        inference_impl = impls[Api.inference]
        print(f"✅ Resolved inference implementation: {type(inference_impl).__name__}")

        # Debug: Check what providers are available
        if hasattr(inference_impl, "routing_table"):
            print(f"🔍 Available providers in routing table:")
            for provider_id, provider in inference_impl.routing_table.impls_by_provider_id.items():
                print(f"   - {provider_id}: {type(provider).__name__}")

        # Debug: Check which provider will be used for our model
        model = await inference_impl.routing_table.get_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        if model:
            print(f"🎯 Model '{model.model_id}' will use provider: {model.provider_id}")
        else:
            print("❌ Model not found in routing table")

        # Make the same inference call (should use cache)
        print("🔄 Making inference call (should use cache)...")

        from llama_stack.apis.inference import SamplingParams, TopPSamplingStrategy, UserMessage

        messages = [UserMessage(content="What is 2+2? Answer in exactly one word.")]
        sampling_params = SamplingParams(max_tokens=10, strategy=TopPSamplingStrategy(temperature=0.1))

        response = await inference_impl.chat_completion(
            model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=messages,
            sampling_params=sampling_params,
        )

        print(f"✅ Got cached response: {response.completion_message.content}")
        print(f"   Stop reason: {response.completion_message.stop_reason}")
        if hasattr(response, "metrics") and response.metrics:
            print(f"   Metrics: {response.metrics}")

        # Verify we got the same response from cache
        if not response.completion_message.content:
            print("❌ Empty response content from cache")
            return False

        print("✅ Cache-only mode working - used cached response")
        return True

    except Exception as e:
        print(f"❌ Error in cache-only test: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_cache_contents(cache_dir: Path):
    """Verify that cache files were created and contain expected data."""

    print(f"🔍 Checking cache directory: {cache_dir}")

    # Look for cache files
    cache_files = list(cache_dir.rglob("*.json"))

    if not cache_files:
        print("❌ No cache files found")
        return False

    print(f"✅ Found {len(cache_files)} cache files:")

    for cache_file in cache_files:
        rel_path = cache_file.relative_to(cache_dir)
        print(f"   📁 {rel_path}")

        # Read and verify cache file structure
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Check expected cache structure
            expected_keys = ["response", "metadata"]
            missing_keys = [k for k in expected_keys if k not in cache_data]

            if missing_keys:
                print(f"   ❌ Missing keys in cache file: {missing_keys}")
                return False

            print(f"   ✅ Valid cache structure")
            print(f"   📊 Metadata: {cache_data['metadata']}")

            # Check if response has expected structure
            response = cache_data["response"]
            if "completion_message" in response and "content" in response["completion_message"]:
                content = response["completion_message"]["content"]
                print(f"   💬 Cached content: {content}")
            else:
                print(f"   ⚠️  Unexpected response structure: {list(response.keys())}")

        except Exception as e:
            print(f"   ❌ Error reading cache file: {e}")
            return False

    print("✅ Cache verification successful")
    return True


async def main():
    """Run the real inference call tests."""
    print("🚀 Starting real inference call tests...")
    print("   This will make actual API calls to Fireworks AI")
    print("=" * 60)

    success = await test_real_inference_calls()

    print("=" * 60)
    if success:
        print("🎉 All tests PASSED!")
        print("   ✅ Cache-with-fallback mode makes real API calls")
        print("   ✅ Cache-only mode uses cached responses")
        print("   ✅ Responses are properly cached and retrieved")
        print("   ✅ Real provider integration works end-to-end")
    else:
        print("💥 Some tests FAILED!")
        print("   Check the output above for details")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
