# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import Api

from .config import LlamaGuardConfig


async def get_provider_impl(config: LlamaGuardConfig, deps: dict[Api, Any]) -> Any:
    from .llama_guard import LlamaGuardSafetyImpl

    assert isinstance(config, LlamaGuardConfig), f"Unexpected config type: {type(config)}"

    impl = LlamaGuardSafetyImpl(config, deps)
    await impl.initialize()
    return impl
