# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field

from llama_stack.distribution.resolver import api_protocol_map
from llama_stack.providers.datatypes import Api
from llama_stack.providers.utils.replay_provider import ReplayMode, ReplayProvider


class ReplayProviderConfig(BaseModel):
    """Configuration for the replay provider that wraps another provider with caching."""

    real_provider_id: str = Field(..., description="The provider ID of the real provider to wrap with caching")
    mode: str = Field(default="cache_with_fallback", description="Replay mode: 'cache_only' or 'cache_with_fallback'")
    cache_dir: str = Field(default="~/.llama_stack_cache", description="Directory to store cached responses")

    @classmethod
    def sample_run_config(cls) -> Dict[str, Any]:
        return {"real_provider_id": "fireworks", "mode": "cache_only", "cache_dir": "~/.llama_stack_cache"}


async def get_replay_provider_impl(
    api: Api, config: ReplayProviderConfig, inner_impls: Dict[str, Any], deps: Dict[Api, Any]
) -> Any:
    """
    Creates a replay provider that wraps the real provider specified in config.

    Args:
        api: The API this provider implements
        config: ReplayProviderConfig with real_provider_id and caching settings
        inner_impls: Dictionary of inner provider implementations by provider_id (unused for now)
        deps: Dictionary of API dependencies

    Returns:
        ReplayProvider instance that wraps the real provider
    """
    # Get the real provider from inner_impls
    if config.real_provider_id not in inner_impls:
        if config.mode == "cache_only":
            # For cache-only mode, we can work without a real provider
            print(f"Warning: Real provider '{config.real_provider_id}' not found, but using cache-only mode")
            real_provider = None
        else:
            raise ValueError(f"Real provider '{config.real_provider_id}' not found in inner implementations")
    else:
        real_provider = inner_impls[config.real_provider_id]
        print(f"Found real provider '{config.real_provider_id}': {type(real_provider).__name__}")

    # Get the protocol for this API
    protocols = api_protocol_map()
    if api not in protocols:
        raise ValueError(f"Unknown API: {api}")

    protocol = protocols[api]

    # Convert string mode to enum
    mode = ReplayMode.CACHE_ONLY if config.mode == "cache_only" else ReplayMode.CACHE_WITH_FALLBACK

    # Create and return the replay provider
    return ReplayProvider(real_provider=real_provider, protocol=protocol, mode=mode, cache_dir=config.cache_dir)
