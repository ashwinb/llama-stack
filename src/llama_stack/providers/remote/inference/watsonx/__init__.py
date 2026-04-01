# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_stack_api import Api

from .config import WatsonXConfig

if TYPE_CHECKING:
    from .watsonx import WatsonXInferenceAdapter


async def get_adapter_impl(config: WatsonXConfig, _deps: dict[Api, Any]) -> WatsonXInferenceAdapter:
    # import dynamically so the import is used only when it is needed
    from .watsonx import WatsonXInferenceAdapter

    adapter = WatsonXInferenceAdapter(config)
    return adapter
