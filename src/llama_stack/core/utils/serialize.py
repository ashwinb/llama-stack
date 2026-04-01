# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import datetime
from enum import Enum
from typing import Any


class EnumEncoder(json.JSONEncoder):
    """JSON encoder that serializes Enum values and datetime objects."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        elif isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)
