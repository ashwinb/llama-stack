# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import Api

from .config import RunpodImplConfig


async def get_adapter_impl(config: RunpodImplConfig, _deps: dict[Api, Any]) -> Any:
    from .runpod import RunpodInferenceAdapter

    assert isinstance(config, RunpodImplConfig), f"Unexpected config type: {type(config)}"
    impl = RunpodInferenceAdapter(config=config)
    await impl.initialize()
    return impl
