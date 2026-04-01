# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_stack_api import Api

from .config import OCIConfig

if TYPE_CHECKING:
    from .oci import OCIInferenceAdapter


async def get_adapter_impl(config: OCIConfig, _deps: dict[Api, Any]) -> OCIInferenceAdapter:
    from .oci import OCIInferenceAdapter

    adapter = OCIInferenceAdapter(config=config)
    await adapter.initialize()
    return adapter
