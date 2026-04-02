# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import SambaNovaSafetyConfig

if TYPE_CHECKING:
    from .sambanova import SambaNovaSafetyAdapter


async def get_adapter_impl(config: SambaNovaSafetyConfig, _deps: dict[str, Any]) -> SambaNovaSafetyAdapter:
    from .sambanova import SambaNovaSafetyAdapter

    impl = SambaNovaSafetyAdapter(config)
    await impl.initialize()
    return impl
