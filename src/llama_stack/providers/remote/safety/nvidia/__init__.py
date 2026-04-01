# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from .config import NVIDIASafetyConfig
from .nvidia import NVIDIASafetyAdapter


async def get_adapter_impl(config: NVIDIASafetyConfig, _deps: dict[str, Any]) -> NVIDIASafetyAdapter:
    impl = NVIDIASafetyAdapter(config)
    await impl.initialize()
    return impl
