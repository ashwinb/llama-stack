# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import NVIDIASafetyConfig

if TYPE_CHECKING:
    from .nvidia import NVIDIASafetyAdapter


async def get_adapter_impl(config: NVIDIASafetyConfig, _deps: dict[str, Any]) -> NVIDIASafetyAdapter:
    from .nvidia import NVIDIASafetyAdapter

    impl = NVIDIASafetyAdapter(config)
    await impl.initialize()
    return impl
