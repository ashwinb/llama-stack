# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from llama_stack_api import ToolRuntime

from .bing_search import BingSearchToolRuntimeImpl
from .config import BingSearchToolConfig

__all__ = ["BingSearchToolConfig", "BingSearchToolRuntimeImpl"]


class BingSearchToolProviderDataValidator(BaseModel):
    bing_search_api_key: SecretStr


async def get_adapter_impl(config: BingSearchToolConfig, _deps: dict[str, Any]) -> ToolRuntime:
    impl = BingSearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
