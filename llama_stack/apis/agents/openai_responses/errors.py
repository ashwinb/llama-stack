# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from pydantic import BaseModel
from llama_stack.schema_utils import json_schema_type, register_schema

@json_schema_type
class OpenAIResponseError(BaseModel):
    """Error details for failed OpenAI response requests.

    :param code: Error code identifying the type of failure
    :param message: Human-readable error message describing the failure
    """

    code: str
    message: str
