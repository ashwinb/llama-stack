# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from .config import SambaNovaSafetyConfig
from .sambanova import SambaNovaSafetyAdapter


async def get_adapter_impl(config: SambaNovaSafetyConfig, _deps: dict[str, Any]) -> SambaNovaSafetyAdapter:
    impl = SambaNovaSafetyAdapter(config)
    await impl.initialize()
    return impl
