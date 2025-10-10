# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Annotated
from pydantic import BaseModel, Field
from typing_extensions import Literal
from llama_stack.schema_utils import json_schema_type, register_schema

@json_schema_type
class OpenAIResponseInputMessageContentText(BaseModel):
    """Text content for input messages in OpenAI response format.

    :param text: The text content of the input message
    :param type: Content type identifier, always "input_text"
    """

    text: str
    type: Literal["input_text"] = "input_text"


@json_schema_type
class OpenAIResponseInputMessageContentImage(BaseModel):
    """Image content for input messages in OpenAI response format.

    :param detail: Level of detail for image processing, can be "low", "high", or "auto"
    :param type: Content type identifier, always "input_image"
    :param image_url: (Optional) URL of the image content
    """

    detail: Literal["low"] | Literal["high"] | Literal["auto"] = "auto"
    type: Literal["input_image"] = "input_image"
    # TODO: handle file_id
    image_url: str | None = None


OpenAIResponseInputMessageContent = Annotated[
    OpenAIResponseInputMessageContentText | OpenAIResponseInputMessageContentImage,
    Field(discriminator="type"),
]

register_schema(OpenAIResponseInputMessageContent, name="OpenAIResponseInputMessageContent")


@json_schema_type
class OpenAIResponseAnnotationFileCitation(BaseModel):
    """File citation annotation for referencing specific files in response content.

    :param type: Annotation type identifier, always "file_citation"
    :param file_id: Unique identifier of the referenced file
    :param filename: Name of the referenced file
    :param index: Position index of the citation within the content
    """

    type: Literal["file_citation"] = "file_citation"
    file_id: str
    filename: str
    index: int


@json_schema_type
class OpenAIResponseAnnotationCitation(BaseModel):
    """URL citation annotation for referencing external web resources.

    :param type: Annotation type identifier, always "url_citation"
    :param end_index: End position of the citation span in the content
    :param start_index: Start position of the citation span in the content
    :param title: Title of the referenced web resource
    :param url: URL of the referenced web resource
    """

    type: Literal["url_citation"] = "url_citation"
    end_index: int
    start_index: int
    title: str
    url: str


@json_schema_type
class OpenAIResponseAnnotationContainerFileCitation(BaseModel):
    type: Literal["container_file_citation"] = "container_file_citation"
    container_id: str
    end_index: int
    file_id: str
    filename: str
    start_index: int


@json_schema_type
class OpenAIResponseAnnotationFilePath(BaseModel):
    type: Literal["file_path"] = "file_path"
    file_id: str
    index: int


OpenAIResponseAnnotations = Annotated[
    OpenAIResponseAnnotationFileCitation
    | OpenAIResponseAnnotationCitation
    | OpenAIResponseAnnotationContainerFileCitation
    | OpenAIResponseAnnotationFilePath,
    Field(discriminator="type"),
]

register_schema(OpenAIResponseAnnotations, name="OpenAIResponseAnnotations")


@json_schema_type
class OpenAIResponseOutputMessageContentOutputText(BaseModel):
    text: str
    type: Literal["output_text"] = "output_text"
    annotations: list[OpenAIResponseAnnotations] = Field(default_factory=list)


OpenAIResponseOutputMessageContent = Annotated[
    OpenAIResponseOutputMessageContentOutputText,
    Field(discriminator="type"),
]

register_schema(OpenAIResponseOutputMessageContent, name="OpenAIResponseOutputMessageContent")


@json_schema_type
class OpenAIResponseMessage(BaseModel):
    """
    Corresponds to the various Message types in the Responses API.
    They are all under one type because the Responses API gives them all
    the same "type" value, and there is no way to tell them apart in certain
    scenarios.
    """

    content: str | list[OpenAIResponseInputMessageContent] | list[OpenAIResponseOutputMessageContent]
    role: Literal["system"] | Literal["developer"] | Literal["user"] | Literal["assistant"]
    type: Literal["message"] = "message"

    # The fields below are not used in all scenarios, but are required in others.
    id: str | None = None
    status: str | None = None
