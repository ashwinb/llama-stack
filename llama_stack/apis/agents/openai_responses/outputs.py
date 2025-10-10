# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Annotated, Any
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from llama_stack.schema_utils import json_schema_type, register_schema
from .messages import OpenAIResponseAnnotations, OpenAIResponseMessage
from .tool_calls import OpenAIResponseMCPApprovalRequest, OpenAIResponseOutputMessageFileSearchToolCall, OpenAIResponseOutputMessageFunctionToolCall, OpenAIResponseOutputMessageMCPCall, OpenAIResponseOutputMessageMCPListTools, OpenAIResponseOutputMessageWebSearchToolCall

OpenAIResponseOutput = Annotated[
    OpenAIResponseMessage
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools
    | OpenAIResponseMCPApprovalRequest,
    Field(discriminator="type"),
]

register_schema(OpenAIResponseOutput, name="OpenAIResponseOutput")


class OpenAIResponseTextFormat(TypedDict, total=False):
    """Configuration for Responses API text format.

    :param type: Must be "text", "json_schema", or "json_object" to identify the format type
    :param name: The name of the response format. Only used for json_schema.
    :param schema: The JSON schema the response should conform to. In a Python SDK, this is often a `pydantic` model. Only used for json_schema.
    :param description: (Optional) A description of the response format. Only used for json_schema.
    :param strict: (Optional) Whether to strictly enforce the JSON schema. If true, the response must match the schema exactly. Only used for json_schema.
    """

    type: Literal["text"] | Literal["json_schema"] | Literal["json_object"]
    name: str | None
    schema: dict[str, Any] | None
    description: str | None
    strict: bool | None


@json_schema_type
class OpenAIResponseText(BaseModel):
    """Text response configuration for OpenAI responses.

    :param format: (Optional) Text format configuration specifying output format requirements
    """

    format: OpenAIResponseTextFormat | None = None


@json_schema_type
class OpenAIResponseContentPartOutputText(BaseModel):
    """Text content within a streamed response part.

    :param type: Content part type identifier, always "output_text"
    :param text: Text emitted for this content part
    :param annotations: Structured annotations associated with the text
    :param logprobs: (Optional) Token log probability details
    """

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[OpenAIResponseAnnotations] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] | None = None


@json_schema_type
class OpenAIResponseContentPartRefusal(BaseModel):
    """Refusal content within a streamed response part.

    :param type: Content part type identifier, always "refusal"
    :param refusal: Refusal text supplied by the model
    """

    type: Literal["refusal"] = "refusal"
    refusal: str


@json_schema_type
class OpenAIResponseContentPartReasoningText(BaseModel):
    """Reasoning text emitted as part of a streamed response.

    :param type: Content part type identifier, always "reasoning_text"
    :param text: Reasoning text supplied by the model
    """

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


OpenAIResponseContentPart = Annotated[
    OpenAIResponseContentPartOutputText | OpenAIResponseContentPartRefusal | OpenAIResponseContentPartReasoningText,
    Field(discriminator="type"),
]

register_schema(OpenAIResponseContentPart, name="OpenAIResponseContentPart")
