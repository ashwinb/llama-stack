# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import CodeScannerConfig

if TYPE_CHECKING:
    from .code_scanner import BuiltinCodeScannerSafetyImpl


async def get_provider_impl(config: CodeScannerConfig, deps: dict[str, Any]) -> BuiltinCodeScannerSafetyImpl:
    from .code_scanner import BuiltinCodeScannerSafetyImpl

    impl = BuiltinCodeScannerSafetyImpl(config, deps)
    await impl.initialize()
    return impl
