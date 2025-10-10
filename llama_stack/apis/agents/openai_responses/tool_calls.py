# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any
from pydantic import BaseModel
from typing_extensions import Literal
from llama_stack.schema_utils import json_schema_type, register_schema

@json_schema_type
class OpenAIResponseOutputMessageWebSearchToolCall(BaseModel):
    """Web search tool call output message for OpenAI responses.

    :param id: Unique identifier for this tool call
    :param status: Current status of the web search operation
    :param type: Tool call type identifier, always "web_search_call"
    """

    id: str
    status: str
    type: Literal["web_search_call"] = "web_search_call"


class OpenAIResponseOutputMessageFileSearchToolCallResults(BaseModel):
    """Search results returned by the file search operation.

    :param attributes: (Optional) Key-value attributes associated with the file
    :param file_id: Unique identifier of the file containing the result
    :param filename: Name of the file containing the result
    :param score: Relevance score for this search result (between 0 and 1)
    :param text: Text content of the search result
    """

    attributes: dict[str, Any]
    file_id: str
    filename: str
    score: float
    text: str


@json_schema_type
class OpenAIResponseOutputMessageFileSearchToolCall(BaseModel):
    """File search tool call output message for OpenAI responses.

    :param id: Unique identifier for this tool call
    :param queries: List of search queries executed
    :param status: Current status of the file search operation
    :param type: Tool call type identifier, always "file_search_call"
    :param results: (Optional) Search results returned by the file search operation
    """

    id: str
    queries: list[str]
    status: str
    type: Literal["file_search_call"] = "file_search_call"
    results: list[OpenAIResponseOutputMessageFileSearchToolCallResults] | None = None


@json_schema_type
class OpenAIResponseOutputMessageFunctionToolCall(BaseModel):
    """Function tool call output message for OpenAI responses.

    :param call_id: Unique identifier for the function call
    :param name: Name of the function being called
    :param arguments: JSON string containing the function arguments
    :param type: Tool call type identifier, always "function_call"
    :param id: (Optional) Additional identifier for the tool call
    :param status: (Optional) Current status of the function call execution
    """

    call_id: str
    name: str
    arguments: str
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    status: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPCall(BaseModel):
    """Model Context Protocol (MCP) call output message for OpenAI responses.

    :param id: Unique identifier for this MCP call
    :param type: Tool call type identifier, always "mcp_call"
    :param arguments: JSON string containing the MCP call arguments
    :param name: Name of the MCP method being called
    :param server_label: Label identifying the MCP server handling the call
    :param error: (Optional) Error message if the MCP call failed
    :param output: (Optional) Output result from the successful MCP call
    """

    id: str
    type: Literal["mcp_call"] = "mcp_call"
    arguments: str
    name: str
    server_label: str
    error: str | None = None
    output: str | None = None


class MCPListToolsTool(BaseModel):
    """Tool definition returned by MCP list tools operation.

    :param input_schema: JSON schema defining the tool's input parameters
    :param name: Name of the tool
    :param description: (Optional) Description of what the tool does
    """

    input_schema: dict[str, Any]
    name: str
    description: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPListTools(BaseModel):
    """MCP list tools output message containing available tools from an MCP server.

    :param id: Unique identifier for this MCP list tools operation
    :param type: Tool call type identifier, always "mcp_list_tools"
    :param server_label: Label identifying the MCP server providing the tools
    :param tools: List of available tools provided by the MCP server
    """

    id: str
    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    server_label: str
    tools: list[MCPListToolsTool]


@json_schema_type
class OpenAIResponseMCPApprovalRequest(BaseModel):
    """
    A request for human approval of a tool invocation.
    """

    arguments: str
    id: str
    name: str
    server_label: str
    type: Literal["mcp_approval_request"] = "mcp_approval_request"


@json_schema_type
class OpenAIResponseMCPApprovalResponse(BaseModel):
    """
    A response to an MCP approval request.
    """

    approval_request_id: str
    approve: bool
    type: Literal["mcp_approval_response"] = "mcp_approval_response"
    id: str | None = None
    reason: str | None = None


@json_schema_type
class OpenAIResponseInputFunctionToolCallOutput(BaseModel):
    """
    This represents the output of a function call that gets passed back to the model.
    """

    call_id: str
    output: str
    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    status: str | None = None
