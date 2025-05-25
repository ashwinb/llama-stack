#!/usr/bin/env python3
"""
Integration test for ReplayProvider with Stack resolver system.
Tests that replay providers can be declaratively configured in run.yaml
and properly instantiated by the resolver.
"""

import asyncio
import tempfile
from pathlib import Path

from llama_stack.distribution.datatypes import Provider, StackRunConfig
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.resolver import resolve_impls
from llama_stack.distribution.store.registry import create_dist_registry
from llama_stack.providers.datatypes import Api


async def test_replay_provider_integration():
    """Test that replay provider can be configured and instantiated."""

    # Create a StackRunConfig with both real provider and replay provider
    run_config = StackRunConfig(
        image_name="test_replay",
        providers={
            "inference": [
                # First, add the real fireworks provider
                Provider(
                    provider_id="fireworks",
                    provider_type="remote::fireworks",
                    config={
                        "url": "https://api.fireworks.ai/inference/v1",
                        "api_key": "fw_3ZmjqrM4b3PhNgH3DXYXC9Ce",  # real key
                    },
                ),
                # Then add the replay provider that wraps it
                Provider(
                    provider_id="__replay__",
                    provider_type="replay",
                    config={
                        "real_provider_id": "fireworks",
                        "mode": "cache_only",
                        "cache_dir": str(Path(tempfile.gettempdir()) / "test_cache"),
                    },
                ),
            ]
        },
    )

    print("✅ Created StackRunConfig with replay provider")
    print(f"   Provider ID: __replay__")
    print(f"   Provider Type: replay")
    print(f"   Config: {run_config.providers['inference'][0].config}")

    # Get provider registry
    provider_registry = get_provider_registry(None)
    print(f"✅ Loaded provider registry with {len(provider_registry)} APIs")

    # Check that replay provider is in the inference registry
    if "replay" in provider_registry[Api.inference]:
        print("✅ Replay provider found in inference registry")
        replay_spec = provider_registry[Api.inference]["replay"]
        print(f"   Spec type: {type(replay_spec).__name__}")
        print(f"   Module: {replay_spec.module}")
        print(f"   Config class: {replay_spec.config_class}")
    else:
        print("❌ Replay provider NOT found in inference registry")
        print(f"   Available providers: {list(provider_registry[Api.inference].keys())}")
        return False

    # Create a distribution registry
    dist_registry, _ = await create_dist_registry(None, "test_replay")

    try:
        # Attempt to resolve implementations
        print("🔄 Attempting to resolve providers...")
        impls = await resolve_impls(run_config, provider_registry, dist_registry)

        print(f"✅ Successfully resolved {len(impls)} implementations")

        # Check if inference API was resolved
        if Api.inference in impls:
            inference_impl = impls[Api.inference]
            print(f"✅ Inference implementation resolved: {type(inference_impl).__name__}")

            # Check if it has the expected provider attributes
            if hasattr(inference_impl, "__provider_id__"):
                print(f"   Provider ID: {inference_impl.__provider_id__}")
            if hasattr(inference_impl, "__provider_spec__"):
                print(f"   Provider Spec: {type(inference_impl.__provider_spec__).__name__}")

            # Check if it's an auto-routed provider with our replay provider as inner impl
            if hasattr(inference_impl, "routing_table") and hasattr(
                inference_impl.routing_table, "impls_by_provider_id"
            ):
                inner_impls = inference_impl.routing_table.impls_by_provider_id
                print(f"   Inner implementations: {list(inner_impls.keys())}")

                # Check that both providers are present
                expected_providers = ["fireworks", "__replay__"]
                missing_providers = [p for p in expected_providers if p not in inner_impls]

                if missing_providers:
                    print(f"❌ Missing providers in inner implementations: {missing_providers}")
                    return False

                # Check the fireworks provider
                fireworks_impl = inner_impls["fireworks"]
                print(f"✅ Fireworks provider found: {type(fireworks_impl).__name__}")

                # Check the replay provider
                replay_impl = inner_impls["__replay__"]
                print(f"✅ Replay provider found as inner implementation: {type(replay_impl).__name__}")

                # Check if it's our ReplayProvider
                if hasattr(replay_impl, "__provider_id__"):
                    print(f"   Replay Provider ID: {replay_impl.__provider_id__}")
                if hasattr(replay_impl, "__provider_spec__"):
                    print(f"   Replay Provider Spec: {type(replay_impl.__provider_spec__).__name__}")

                # Check if the replay provider has access to the real provider
                if hasattr(replay_impl, "real_provider"):
                    if replay_impl.real_provider is not None:
                        print(f"✅ Replay provider has real provider: {type(replay_impl.real_provider).__name__}")
                    else:
                        print("⚠️  Replay provider has None real_provider (cache-only mode)")

                return True
            else:
                print("❌ No routing table found in inference implementation")
                return False
        else:
            print("❌ Inference API not found in resolved implementations")
            return False

    except Exception as e:
        print(f"❌ Failed to resolve providers: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the integration test."""
    print("🚀 Starting ReplayProvider integration test...")
    print("=" * 60)

    success = await test_replay_provider_integration()

    print("=" * 60)
    if success:
        print("🎉 Integration test PASSED!")
        print("   Replay provider can be configured and instantiated via StackRunConfig")
    else:
        print("💥 Integration test FAILED!")
        print("   Replay provider integration needs debugging")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
