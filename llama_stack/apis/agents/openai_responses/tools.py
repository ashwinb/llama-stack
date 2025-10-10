# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Annotated, Any
from pydantic import BaseModel, Field
from typing_extensions import Literal
from llama_stack.schema_utils import json_schema_type, register_schema

WebSearchToolTypes = ["web_search", "web_search_preview", "web_search_preview_2025_03_11"]


@json_schema_type
class OpenAIResponseInputToolWebSearch(BaseModel):
    """Web search tool configuration for OpenAI response inputs.

    :param type: Web search tool type variant to use
    :param search_context_size: (Optional) Size of search context, must be "low", "medium", or "high"
    """

    # Must match values of WebSearchToolTypes above
    type: Literal["web_search"] | Literal["web_search_preview"] | Literal["web_search_preview_2025_03_11"] = (
        "web_search"
    )
    # TODO: actually use search_context_size somewhere...
    search_context_size: str | None = Field(default="medium", pattern="^low|medium|high$")


@json_schema_type
class OpenAIResponseInputToolFunction(BaseModel):
    """Function tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "function"
    :param name: Name of the function that can be called
    :param description: (Optional) Description of what the function does
    :param parameters: (Optional) JSON schema defining the function's parameters
    :param strict: (Optional) Whether to enforce strict parameter validation
    """

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None
    strict: bool | None = None


@json_schema_type
class OpenAIResponseInputToolFileSearch(BaseModel):
    """File search tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "file_search"
    :param vector_store_ids: List of vector store identifiers to search within
    :param filters: (Optional) Additional filters to apply to the search
    :param max_num_results: (Optional) Maximum number of search results to return (1-50)
    :param ranking_options: (Optional) Options for ranking and scoring search results
    """

    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int | None = Field(default=10, ge=1, le=50)
    ranking_options: FileSearchRankingOptions | None = None


class ApprovalFilter(BaseModel):
    """Filter configuration for MCP tool approval requirements.

    :param always: (Optional) List of tool names that always require approval
    :param never: (Optional) List of tool names that never require approval
    """

    always: list[str] | None = None
    never: list[str] | None = None


class AllowedToolsFilter(BaseModel):
    """Filter configuration for restricting which MCP tools can be used.

    :param tool_names: (Optional) List of specific tool names that are allowed
    """

    tool_names: list[str] | None = None


@json_schema_type
class OpenAIResponseInputToolMCP(BaseModel):
    """Model Context Protocol (MCP) tool configuration for OpenAI response inputs.

    :param type: Tool type identifier, always "mcp"
    :param server_label: Label to identify this MCP server
    :param server_url: URL endpoint of the MCP server
    :param headers: (Optional) HTTP headers to include when connecting to the server
    :param require_approval: Approval requirement for tool calls ("always", "never", or filter)
    :param allowed_tools: (Optional) Restriction on which tools can be used from this server
    """

    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str
    headers: dict[str, Any] | None = None

    require_approval: Literal["always"] | Literal["never"] | ApprovalFilter = "never"
    allowed_tools: list[str] | AllowedToolsFilter | None = None


OpenAIResponseInputTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseInputToolMCP,
    Field(discriminator="type"),
]

register_schema(OpenAIResponseInputTool, name="OpenAIResponseInputTool")


@json_schema_type
class OpenAIResponseToolMCP(BaseModel):
    """Model Context Protocol (MCP) tool configuration for OpenAI response object.

    :param type: Tool type identifier, always "mcp"
    :param server_label: Label to identify this MCP server
    :param allowed_tools: (Optional) Restriction on which tools can be used from this server
    """

    type: Literal["mcp"] = "mcp"
    server_label: str
    allowed_tools: list[str] | AllowedToolsFilter | None = None


OpenAIResponseTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseToolMCP,  # The only type that differes from that in the inputs is the MCP tool
    Field(discriminator="type"),
]

register_schema(OpenAIResponseTool, name="OpenAIResponseTool")
