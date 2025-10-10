# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Annotated
from pydantic import BaseModel, Field
from typing_extensions import Literal
from llama_stack.schema_utils import json_schema_type, register_schema
from .messages import OpenAIResponseMessage
from .tool_calls import OpenAIResponseInputFunctionToolCallOutput, OpenAIResponseMCPApprovalRequest, OpenAIResponseMCPApprovalResponse, OpenAIResponseOutputMessageFileSearchToolCall, OpenAIResponseOutputMessageFunctionToolCall, OpenAIResponseOutputMessageWebSearchToolCall

OpenAIResponseInput = Annotated[
    # Responses API allows output messages to be passed in as input
    OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseInputFunctionToolCallOutput
    | OpenAIResponseMCPApprovalRequest
    | OpenAIResponseMCPApprovalResponse
    |
    # Fallback to the generic message type as a last resort
    OpenAIResponseMessage,
    Field(union_mode="left_to_right"),
]

register_schema(OpenAIResponseInput, name="OpenAIResponseInput")


class ListOpenAIResponseInputItem(BaseModel):
    """List container for OpenAI response input items.

    :param data: List of input items
    :param object: Object type identifier, always "list"
    """

    data: list[OpenAIResponseInput]
    object: Literal["list"] = "list"
