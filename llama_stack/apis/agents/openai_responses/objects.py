# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Annotated
from pydantic import BaseModel, Field
from typing_extensions import Literal
from llama_stack.schema_utils import json_schema_type, register_schema
from .errors import OpenAIResponseError
from .inputs import OpenAIResponseInput
from .outputs import OpenAIResponseContentPart, OpenAIResponseOutput, OpenAIResponseText, OpenAIResponseTextFormat
from .tools import OpenAIResponseTool
from .usage import OpenAIResponseUsage

@json_schema_type
class OpenAIResponseObject(BaseModel):
    """Complete OpenAI response object containing generation results and metadata.

    :param created_at: Unix timestamp when the response was created
    :param error: (Optional) Error details if the response generation failed
    :param id: Unique identifier for this response
    :param model: Model identifier used for generation
    :param object: Object type identifier, always "response"
    :param output: List of generated output items (messages, tool calls, etc.)
    :param parallel_tool_calls: Whether tool calls can be executed in parallel
    :param previous_response_id: (Optional) ID of the previous response in a conversation
    :param status: Current status of the response generation
    :param temperature: (Optional) Sampling temperature used for generation
    :param text: Text formatting configuration for the response
    :param top_p: (Optional) Nucleus sampling parameter used for generation
    :param tools: (Optional) An array of tools the model may call while generating a response.
    :param truncation: (Optional) Truncation strategy applied to the response
    :param usage: (Optional) Token usage information for the response
    """

    created_at: int
    error: OpenAIResponseError | None = None
    id: str
    model: str
    object: Literal["response"] = "response"
    output: list[OpenAIResponseOutput]
    parallel_tool_calls: bool = False
    previous_response_id: str | None = None
    status: str
    temperature: float | None = None
    # Default to text format to avoid breaking the loading of old responses
    # before the field was added. New responses will have this set always.
    text: OpenAIResponseText = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text"))
    top_p: float | None = None
    tools: list[OpenAIResponseTool] | None = None
    truncation: str | None = None
    usage: OpenAIResponseUsage | None = None


@json_schema_type
class OpenAIDeleteResponseObject(BaseModel):
    """Response object confirming deletion of an OpenAI response.

    :param id: Unique identifier of the deleted response
    :param object: Object type identifier, always "response"
    :param deleted: Deletion confirmation flag, always True
    """

    id: str
    object: Literal["response"] = "response"
    deleted: bool = True


@json_schema_type
class OpenAIResponseObjectStreamResponseCreated(BaseModel):
    """Streaming event indicating a new response has been created.

    :param response: The response object that was created
    :param type: Event type identifier, always "response.created"
    """

    response: OpenAIResponseObject
    type: Literal["response.created"] = "response.created"


@json_schema_type
class OpenAIResponseObjectStreamResponseInProgress(BaseModel):
    """Streaming event indicating the response remains in progress.

    :param response: Current response state while in progress
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.in_progress"
    """

    response: OpenAIResponseObject
    sequence_number: int
    type: Literal["response.in_progress"] = "response.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseCompleted(BaseModel):
    """Streaming event indicating a response has been completed.

    :param response: Completed response object
    :param type: Event type identifier, always "response.completed"
    """

    response: OpenAIResponseObject
    type: Literal["response.completed"] = "response.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseIncomplete(BaseModel):
    """Streaming event emitted when a response ends in an incomplete state.

    :param response: Response object describing the incomplete state
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.incomplete"
    """

    response: OpenAIResponseObject
    sequence_number: int
    type: Literal["response.incomplete"] = "response.incomplete"


@json_schema_type
class OpenAIResponseObjectStreamResponseFailed(BaseModel):
    """Streaming event emitted when a response fails.

    :param response: Response object describing the failure
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.failed"
    """

    response: OpenAIResponseObject
    sequence_number: int
    type: Literal["response.failed"] = "response.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemAdded(BaseModel):
    """Streaming event for when a new output item is added to the response.

    :param response_id: Unique identifier of the response containing this output
    :param item: The output item that was added (message, tool call, etc.)
    :param output_index: Index position of this item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_item.added"
    """

    response_id: str
    item: OpenAIResponseOutput
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.added"] = "response.output_item.added"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemDone(BaseModel):
    """Streaming event for when an output item is completed.

    :param response_id: Unique identifier of the response containing this output
    :param item: The completed output item (message, tool call, etc.)
    :param output_index: Index position of this item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_item.done"
    """

    response_id: str
    item: OpenAIResponseOutput
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.done"] = "response.output_item.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDelta(BaseModel):
    """Streaming event for incremental text content updates.

    :param content_index: Index position within the text content
    :param delta: Incremental text content being added
    :param item_id: Unique identifier of the output item being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.delta"
    """

    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.delta"] = "response.output_text.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDone(BaseModel):
    """Streaming event for when text output is completed.

    :param content_index: Index position within the text content
    :param text: Final complete text content of the output item
    :param item_id: Unique identifier of the completed output item
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.output_text.done"
    """

    content_index: int
    text: str  # final text of the output item
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.done"] = "response.output_text.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta(BaseModel):
    """Streaming event for incremental function call argument updates.

    :param delta: Incremental function call arguments being added
    :param item_id: Unique identifier of the function call being updated
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.function_call_arguments.delta"
    """

    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone(BaseModel):
    """Streaming event for when function call arguments are completed.

    :param arguments: Final complete arguments JSON string for the function call
    :param item_id: Unique identifier of the completed function call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.function_call_arguments.done"
    """

    arguments: str  # final arguments of the function call
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallInProgress(BaseModel):
    """Streaming event for web search calls in progress.

    :param item_id: Unique identifier of the web search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.web_search_call.in_progress"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.in_progress"] = "response.web_search_call.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallSearching(BaseModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.searching"] = "response.web_search_call.searching"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallCompleted(BaseModel):
    """Streaming event for completed web search calls.

    :param item_id: Unique identifier of the completed web search call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.web_search_call.completed"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.completed"] = "response.web_search_call.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsInProgress(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.in_progress"] = "response.mcp_list_tools.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsFailed(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.failed"] = "response.mcp_list_tools.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsCompleted(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.completed"] = "response.mcp_list_tools.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta(BaseModel):
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.arguments.delta"] = "response.mcp_call.arguments.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDone(BaseModel):
    arguments: str  # final arguments of the MCP call
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.arguments.done"] = "response.mcp_call.arguments.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallInProgress(BaseModel):
    """Streaming event for MCP calls in progress.

    :param item_id: Unique identifier of the MCP call
    :param output_index: Index position of the item in the output list
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.in_progress"
    """

    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.in_progress"] = "response.mcp_call.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallFailed(BaseModel):
    """Streaming event for failed MCP calls.

    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.failed"
    """

    sequence_number: int
    type: Literal["response.mcp_call.failed"] = "response.mcp_call.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallCompleted(BaseModel):
    """Streaming event for completed MCP calls.

    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.mcp_call.completed"
    """

    sequence_number: int
    type: Literal["response.mcp_call.completed"] = "response.mcp_call.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseContentPartAdded(BaseModel):
    """Streaming event for when a new content part is added to a response item.

    :param content_index: Index position of the part within the content array
    :param response_id: Unique identifier of the response containing this content
    :param item_id: Unique identifier of the output item containing this content part
    :param output_index: Index position of the output item in the response
    :param part: The content part that was added
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.content_part.added"
    """

    content_index: int
    response_id: str
    item_id: str
    output_index: int
    part: OpenAIResponseContentPart
    sequence_number: int
    type: Literal["response.content_part.added"] = "response.content_part.added"


@json_schema_type
class OpenAIResponseObjectStreamResponseContentPartDone(BaseModel):
    """Streaming event for when a content part is completed.

    :param content_index: Index position of the part within the content array
    :param response_id: Unique identifier of the response containing this content
    :param item_id: Unique identifier of the output item containing this content part
    :param output_index: Index position of the output item in the response
    :param part: The completed content part
    :param sequence_number: Sequential number for ordering streaming events
    :param type: Event type identifier, always "response.content_part.done"
    """

    content_index: int
    response_id: str
    item_id: str
    output_index: int
    part: OpenAIResponseContentPart
    sequence_number: int
    type: Literal["response.content_part.done"] = "response.content_part.done"


OpenAIResponseObjectStream = Annotated[
    OpenAIResponseObjectStreamResponseCreated
    | OpenAIResponseObjectStreamResponseInProgress
    | OpenAIResponseObjectStreamResponseOutputItemAdded
    | OpenAIResponseObjectStreamResponseOutputItemDone
    | OpenAIResponseObjectStreamResponseOutputTextDelta
    | OpenAIResponseObjectStreamResponseOutputTextDone
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone
    | OpenAIResponseObjectStreamResponseWebSearchCallInProgress
    | OpenAIResponseObjectStreamResponseWebSearchCallSearching
    | OpenAIResponseObjectStreamResponseWebSearchCallCompleted
    | OpenAIResponseObjectStreamResponseMcpListToolsInProgress
    | OpenAIResponseObjectStreamResponseMcpListToolsFailed
    | OpenAIResponseObjectStreamResponseMcpListToolsCompleted
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDone
    | OpenAIResponseObjectStreamResponseMcpCallInProgress
    | OpenAIResponseObjectStreamResponseMcpCallFailed
    | OpenAIResponseObjectStreamResponseMcpCallCompleted
    | OpenAIResponseObjectStreamResponseContentPartAdded
    | OpenAIResponseObjectStreamResponseContentPartDone
    | OpenAIResponseObjectStreamResponseIncomplete
    | OpenAIResponseObjectStreamResponseFailed
    | OpenAIResponseObjectStreamResponseCompleted,
    Field(discriminator="type"),
]

register_schema(OpenAIResponseObjectStream, name="OpenAIResponseObjectStream")


@json_schema_type
class OpenAIResponseObjectWithInput(OpenAIResponseObject):
    """OpenAI response object extended with input context information.

    :param input: List of input items that led to this response
    """

    input: list[OpenAIResponseInput]

    def to_response_object(self) -> OpenAIResponseObject:
        """Convert to OpenAIResponseObject by excluding input field."""
        return OpenAIResponseObject(**{k: v for k, v in self.model_dump().items() if k != "input"})


@json_schema_type
class ListOpenAIResponseObject(BaseModel):
    """Paginated list of OpenAI response objects with navigation metadata.

    :param data: List of response objects with their input context
    :param has_more: Whether there are more results available beyond this page
    :param first_id: Identifier of the first item in this page
    :param last_id: Identifier of the last item in this page
    :param object: Object type identifier, always "list"
    """

    data: list[OpenAIResponseObjectWithInput]
    has_more: bool
    first_id: str
    last_id: str
    object: Literal["list"] = "list"
