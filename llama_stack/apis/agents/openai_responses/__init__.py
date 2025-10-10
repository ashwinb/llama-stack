# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenAI Responses schema re-exports."""

# This package mirrors the original openai_responses.py module but splits the
# definitions into focused submodules. The generator at
# scripts/generate_openai_responses.py targets this layout so the code can be
# updated automatically in manageable chunks.

from .errors import OpenAIResponseError

from .inputs import (
    OpenAIResponseInput,
    ListOpenAIResponseInputItem,
)

from .messages import (
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContent,
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseAnnotationCitation,
    OpenAIResponseAnnotationContainerFileCitation,
    OpenAIResponseAnnotationFilePath,
    OpenAIResponseAnnotations,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageContent,
    OpenAIResponseMessage,
)

from .objects import (
    OpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseObjectStreamResponseInProgress,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseIncomplete,
    OpenAIResponseObjectStreamResponseFailed,
    OpenAIResponseObjectStreamResponseOutputItemAdded,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseObjectStreamResponseOutputTextDone,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone,
    OpenAIResponseObjectStreamResponseWebSearchCallInProgress,
    OpenAIResponseObjectStreamResponseWebSearchCallSearching,
    OpenAIResponseObjectStreamResponseWebSearchCallCompleted,
    OpenAIResponseObjectStreamResponseMcpListToolsInProgress,
    OpenAIResponseObjectStreamResponseMcpListToolsFailed,
    OpenAIResponseObjectStreamResponseMcpListToolsCompleted,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDone,
    OpenAIResponseObjectStreamResponseMcpCallInProgress,
    OpenAIResponseObjectStreamResponseMcpCallFailed,
    OpenAIResponseObjectStreamResponseMcpCallCompleted,
    OpenAIResponseObjectStreamResponseContentPartAdded,
    OpenAIResponseObjectStreamResponseContentPartDone,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectWithInput,
    ListOpenAIResponseObject,
)

from .outputs import (
    OpenAIResponseOutput,
    OpenAIResponseTextFormat,
    OpenAIResponseText,
    OpenAIResponseContentPartOutputText,
    OpenAIResponseContentPartRefusal,
    OpenAIResponseContentPartReasoningText,
    OpenAIResponseContentPart,
)

from .tool_calls import (
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAIResponseOutputMessageFileSearchToolCallResults,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    MCPListToolsTool,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseInputFunctionToolCallOutput,
)

from .tools import (
    WebSearchToolTypes,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolFileSearch,
    ApprovalFilter,
    AllowedToolsFilter,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputTool,
    OpenAIResponseToolMCP,
    OpenAIResponseTool,
)

from .usage import (
    OpenAIResponseUsageOutputTokensDetails,
    OpenAIResponseUsageInputTokensDetails,
    OpenAIResponseUsage,
)

__all__ = [
    'OpenAIResponseError',
    'OpenAIResponseInput',
    'ListOpenAIResponseInputItem',
    'OpenAIResponseInputMessageContentText',
    'OpenAIResponseInputMessageContentImage',
    'OpenAIResponseInputMessageContent',
    'OpenAIResponseAnnotationFileCitation',
    'OpenAIResponseAnnotationCitation',
    'OpenAIResponseAnnotationContainerFileCitation',
    'OpenAIResponseAnnotationFilePath',
    'OpenAIResponseAnnotations',
    'OpenAIResponseOutputMessageContentOutputText',
    'OpenAIResponseOutputMessageContent',
    'OpenAIResponseMessage',
    'OpenAIResponseObject',
    'OpenAIDeleteResponseObject',
    'OpenAIResponseObjectStreamResponseCreated',
    'OpenAIResponseObjectStreamResponseInProgress',
    'OpenAIResponseObjectStreamResponseCompleted',
    'OpenAIResponseObjectStreamResponseIncomplete',
    'OpenAIResponseObjectStreamResponseFailed',
    'OpenAIResponseObjectStreamResponseOutputItemAdded',
    'OpenAIResponseObjectStreamResponseOutputItemDone',
    'OpenAIResponseObjectStreamResponseOutputTextDelta',
    'OpenAIResponseObjectStreamResponseOutputTextDone',
    'OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta',
    'OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone',
    'OpenAIResponseObjectStreamResponseWebSearchCallInProgress',
    'OpenAIResponseObjectStreamResponseWebSearchCallSearching',
    'OpenAIResponseObjectStreamResponseWebSearchCallCompleted',
    'OpenAIResponseObjectStreamResponseMcpListToolsInProgress',
    'OpenAIResponseObjectStreamResponseMcpListToolsFailed',
    'OpenAIResponseObjectStreamResponseMcpListToolsCompleted',
    'OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta',
    'OpenAIResponseObjectStreamResponseMcpCallArgumentsDone',
    'OpenAIResponseObjectStreamResponseMcpCallInProgress',
    'OpenAIResponseObjectStreamResponseMcpCallFailed',
    'OpenAIResponseObjectStreamResponseMcpCallCompleted',
    'OpenAIResponseObjectStreamResponseContentPartAdded',
    'OpenAIResponseObjectStreamResponseContentPartDone',
    'OpenAIResponseObjectStream',
    'OpenAIResponseObjectWithInput',
    'ListOpenAIResponseObject',
    'OpenAIResponseOutput',
    'OpenAIResponseTextFormat',
    'OpenAIResponseText',
    'OpenAIResponseContentPartOutputText',
    'OpenAIResponseContentPartRefusal',
    'OpenAIResponseContentPartReasoningText',
    'OpenAIResponseContentPart',
    'OpenAIResponseOutputMessageWebSearchToolCall',
    'OpenAIResponseOutputMessageFileSearchToolCallResults',
    'OpenAIResponseOutputMessageFileSearchToolCall',
    'OpenAIResponseOutputMessageFunctionToolCall',
    'OpenAIResponseOutputMessageMCPCall',
    'MCPListToolsTool',
    'OpenAIResponseOutputMessageMCPListTools',
    'OpenAIResponseMCPApprovalRequest',
    'OpenAIResponseMCPApprovalResponse',
    'OpenAIResponseInputFunctionToolCallOutput',
    'WebSearchToolTypes',
    'OpenAIResponseInputToolWebSearch',
    'OpenAIResponseInputToolFunction',
    'OpenAIResponseInputToolFileSearch',
    'ApprovalFilter',
    'AllowedToolsFilter',
    'OpenAIResponseInputToolMCP',
    'OpenAIResponseInputTool',
    'OpenAIResponseToolMCP',
    'OpenAIResponseTool',
    'OpenAIResponseUsageOutputTokensDetails',
    'OpenAIResponseUsageInputTokensDetails',
    'OpenAIResponseUsage',
]
