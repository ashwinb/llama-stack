"""
Generic ReplayProvider that can wrap any API provider with caching capability.
Inspired by create_api_client_class pattern for dynamic protocol implementation.
"""

import hashlib
import inspect
import json
from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path
from typing import Any


class ReplayMode(Enum):
    """Operating modes for ReplayProvider."""

    CACHE_ONLY = "cache_only"  # Never call real provider, error on cache miss
    CACHE_WITH_FALLBACK = "fallback"  # Cache miss -> real provider -> update cache


def create_replay_provider_class(protocol) -> type:
    """
    Creates a ReplayProvider class that implements the given protocol with caching.
    Similar to create_api_client_class but adds caching layer to all webmethod calls.

    Args:
        protocol: The protocol class (e.g., InferenceProvider, Agents, etc.)

    Returns:
        A ReplayProvider class that implements the protocol with caching
    """

    class ReplayProvider:
        def __init__(
            self,
            real_provider: Any,
            mode: ReplayMode = ReplayMode.CACHE_WITH_FALLBACK,
            cache_dir: str = "~/.llama_stack_cache",
        ):
            self.real_provider = real_provider
            self.mode = mode
            self.cache_dir = Path(cache_dir).expanduser()
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Store routes for this protocol (similar to APIClient)
            self.routes = {}
            for name, method in inspect.getmembers(protocol):
                if hasattr(method, "__webmethod__"):
                    sig = inspect.signature(method)
                    self.routes[name] = (method.__webmethod__, sig)

        async def initialize(self):
            """Initialize the real provider."""
            if hasattr(self.real_provider, "initialize"):
                await self.real_provider.initialize()

        async def shutdown(self):
            """Shutdown the real provider."""
            if hasattr(self.real_provider, "shutdown"):
                await self.real_provider.shutdown()

        async def register_model(self, model):
            """Register model with the real provider."""
            if hasattr(self.real_provider, "register_model"):
                return await self.real_provider.register_model(model)
            return model

        async def unregister_model(self, model_id: str):
            """Unregister model with the real provider."""
            if hasattr(self.real_provider, "unregister_model"):
                await self.real_provider.unregister_model(model_id)

        def _generate_cache_key(self, method_name: str, *args, **kwargs) -> str:
            """Generate deterministic cache key for method call."""
            # TODO: Implement proper parameter hashing
            # For now, simple hash of method name and str representation
            param_str = f"{args}_{kwargs}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

            provider_type = type(self.real_provider).__name__ if self.real_provider else "NoProvider"
            api_name = protocol.__name__.lower().replace("provider", "")

            return f"{api_name}_{provider_type}_{method_name}_{param_hash}"

        def _get_cache_path(self, cache_key: str) -> Path:
            """Get cache file path for a given cache key."""
            provider_type = type(self.real_provider).__name__ if self.real_provider else "NoProvider"
            api_name = protocol.__name__.lower().replace("provider", "")

            cache_path = self.cache_dir / api_name / provider_type / f"{cache_key}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            return cache_path

        async def _load_from_cache(self, cache_key: str) -> Any:
            """Load response from cache if available."""
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                print(f"📁 Loading from cache: {cache_path}")
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                    response_data = cached_data.get("response")
                    response_type = cached_data.get("metadata", {}).get("response_type")

                    # Try to reconstruct the proper response object
                    if response_type and response_data:
                        try:
                            # Import the response class dynamically
                            if response_type == "ChatCompletionResponse":
                                from llama_stack.apis.inference import ChatCompletionResponse

                                return ChatCompletionResponse(**response_data)
                            elif response_type == "CompletionResponse":
                                from llama_stack.apis.inference import CompletionResponse

                                return CompletionResponse(**response_data)
                            elif response_type == "EmbeddingsResponse":
                                from llama_stack.apis.inference import EmbeddingsResponse

                                return EmbeddingsResponse(**response_data)
                            # Add more response types as needed
                        except Exception as e:
                            print(f"⚠️  Failed to reconstruct {response_type}: {e}")
                            print(f"   Falling back to raw response data")

                    # Fallback to raw response data
                    return response_data
            return None

        async def _save_to_cache(self, cache_key: str, response: Any) -> None:
            """Save response to cache."""
            cache_path = self._get_cache_path(cache_key)

            # Serialize response properly (handle Pydantic models)
            if hasattr(response, "model_dump"):
                # Pydantic v2 model - use mode='json' to properly serialize enums
                serialized_response = response.model_dump(mode="json")
            elif hasattr(response, "dict"):
                # Pydantic v1 model
                serialized_response = response.dict()
            else:
                # Fallback to string representation
                serialized_response = str(response)

            cache_data = {
                "cache_key": cache_key,
                "response": serialized_response,
                "metadata": {
                    "provider_type": type(self.real_provider).__name__ if self.real_provider else "NoProvider",
                    "api_name": protocol.__name__,
                    "response_type": type(response).__name__,
                },
            }

            print(f"💾 Saving to cache: {cache_path}")
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

        async def __acall__(self, method_name: str, *args, **kwargs) -> Any:
            """Handle caching for any protocol method call."""
            if method_name not in self.routes:
                raise AttributeError(f"Unknown method: {method_name}")

            cache_key = self._generate_cache_key(method_name, *args, **kwargs)

            # Try cache first
            cached_response = await self._load_from_cache(cache_key)
            if cached_response is not None:
                print(f"Cache HIT for {cache_key}")
                return cached_response

            # Handle cache miss
            if self.mode == ReplayMode.CACHE_ONLY:
                raise RuntimeError(f"Cache miss for {cache_key} and mode is CACHE_ONLY")

            # Call real provider
            if self.real_provider is None:
                raise RuntimeError(f"Cache miss for {cache_key} and no real provider available")

            print(f"Cache MISS for {cache_key}, calling real provider")
            real_method = getattr(self.real_provider, method_name)
            response = await real_method(*args, **kwargs)

            # Save to cache
            await self._save_to_cache(cache_key, response)

            return response

    # Add protocol methods to the ReplayProvider class
    for name, method in inspect.getmembers(protocol):
        if hasattr(method, "__webmethod__"):

            async def method_impl(self, *args, method_name=name, **kwargs):
                return await self.__acall__(method_name, *args, **kwargs)

            method_impl.__name__ = name
            method_impl.__qualname__ = f"ReplayProvider.{name}"
            method_impl.__signature__ = inspect.signature(method)
            setattr(ReplayProvider, name, method_impl)

    # Name the class after the protocol
    ReplayProvider.__name__ = f"{protocol.__name__}ReplayProvider"
    return ReplayProvider


def ReplayProvider(
    real_provider: Any,
    protocol: type,
    mode: ReplayMode = ReplayMode.CACHE_WITH_FALLBACK,
    cache_dir: str = "~/.llama_stack_cache",
):
    """
    Convenience function to create a ReplayProvider instance for any protocol.

    Args:
        real_provider: The real provider to wrap (can be None for cache-only mode)
        protocol: The protocol class (e.g., InferenceProvider, Agents, etc.)
        mode: Operating mode (CACHE_ONLY or CACHE_WITH_FALLBACK)
        cache_dir: Directory to store cached responses

    Returns:
        ReplayProvider instance that implements the protocol with caching
    """
    ReplayProviderClass = create_replay_provider_class(protocol)
    return ReplayProviderClass(real_provider, mode, cache_dir)
