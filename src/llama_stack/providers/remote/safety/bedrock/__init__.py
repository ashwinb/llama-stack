# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import BedrockSafetyConfig

if TYPE_CHECKING:
    from .bedrock import BedrockSafetyAdapter


async def get_adapter_impl(config: BedrockSafetyConfig, _deps: dict[str, Any]) -> BedrockSafetyAdapter:
    from .bedrock import BedrockSafetyAdapter

    impl = BedrockSafetyAdapter(config)
    await impl.initialize()
    return impl
