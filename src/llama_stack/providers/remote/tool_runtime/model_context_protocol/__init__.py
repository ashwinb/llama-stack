# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import Api

from .config import MCPProviderConfig
from .model_context_protocol import ModelContextProtocolToolRuntimeImpl


async def get_adapter_impl(config: MCPProviderConfig, _deps: dict[Api, Any]) -> ModelContextProtocolToolRuntimeImpl:
    impl = ModelContextProtocolToolRuntimeImpl(config, _deps)
    await impl.initialize()
    return impl
