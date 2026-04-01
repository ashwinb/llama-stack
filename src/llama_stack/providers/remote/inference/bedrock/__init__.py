# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import BedrockConfig

if TYPE_CHECKING:
    from .bedrock import BedrockInferenceAdapter


async def get_adapter_impl(config: BedrockConfig, _deps: dict[str, Any]) -> BedrockInferenceAdapter:
    from .bedrock import BedrockInferenceAdapter

    assert isinstance(config, BedrockConfig), f"Unexpected config type: {type(config)}"

    impl = BedrockInferenceAdapter(config=config)

    await impl.initialize()

    return impl
