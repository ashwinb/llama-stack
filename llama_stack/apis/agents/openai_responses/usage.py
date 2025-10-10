# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from pydantic import BaseModel
from llama_stack.schema_utils import json_schema_type, register_schema

class OpenAIResponseUsageOutputTokensDetails(BaseModel):
    """Token details for output tokens in OpenAI response usage.

    :param reasoning_tokens: Number of tokens used for reasoning (o1/o3 models)
    """

    reasoning_tokens: int | None = None


class OpenAIResponseUsageInputTokensDetails(BaseModel):
    """Token details for input tokens in OpenAI response usage.

    :param cached_tokens: Number of tokens retrieved from cache
    """

    cached_tokens: int | None = None


@json_schema_type
class OpenAIResponseUsage(BaseModel):
    """Usage information for OpenAI response.

    :param input_tokens: Number of tokens in the input
    :param output_tokens: Number of tokens in the output
    :param total_tokens: Total tokens used (input + output)
    :param input_tokens_details: Detailed breakdown of input token usage
    :param output_tokens_details: Detailed breakdown of output token usage
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: OpenAIResponseUsageInputTokensDetails | None = None
    output_tokens_details: OpenAIResponseUsageOutputTokensDetails | None = None
