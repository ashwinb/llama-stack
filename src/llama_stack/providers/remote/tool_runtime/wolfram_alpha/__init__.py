# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from llama_stack_api import Api

from .config import WolframAlphaToolConfig
from .wolfram_alpha import WolframAlphaToolRuntimeImpl

__all__ = ["WolframAlphaToolConfig", "WolframAlphaToolRuntimeImpl"]


class WolframAlphaToolProviderDataValidator(BaseModel):
    wolfram_alpha_api_key: SecretStr


async def get_adapter_impl(config: WolframAlphaToolConfig, _deps: dict[Api, Any]) -> WolframAlphaToolRuntimeImpl:
    impl = WolframAlphaToolRuntimeImpl(config)
    await impl.initialize()
    return impl
