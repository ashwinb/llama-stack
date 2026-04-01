# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_stack.core.datatypes import AccessRule, Api

from .config import BuiltinResponsesImplConfig

if TYPE_CHECKING:
    from .impl import BuiltinResponsesImpl


async def get_provider_impl(
    config: BuiltinResponsesImplConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
) -> BuiltinResponsesImpl:
    from .impl import BuiltinResponsesImpl

    impl = BuiltinResponsesImpl(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps.get(Api.safety),
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
        deps[Api.conversations],
        deps[Api.prompts],
        deps[Api.files],
        deps[Api.connectors],
        policy,
    )
    await impl.initialize()
    return impl
