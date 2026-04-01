# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.routing_tables.datasets import DatasetsRoutingTable
from llama_stack.log import get_logger
from llama_stack_api import (
    AppendRowsParams,
    DatasetIO,
    DatasetPurpose,
    DataSource,
    IterRowsRequest,
    PaginatedResponse,
)
from llama_stack_api.datasets.api import RegisterDatasetRequest

logger = get_logger(name=__name__, category="core::routers")


class DatasetIORouter(DatasetIO):
    """Router that delegates DatasetIO operations to the appropriate provider via a routing table."""

    def __init__(
        self,
        routing_table: DatasetsRoutingTable,
    ) -> None:
        logger.debug("Initializing DatasetIORouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("DatasetIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("DatasetIORouter.shutdown")
        pass

    async def register_dataset(
        self,
        purpose: DatasetPurpose,
        source: DataSource,
        metadata: dict[str, Any] | None = None,
        dataset_id: str | None = None,
    ) -> None:
        logger.debug(
            "DatasetIORouter.register_dataset",
            purpose=purpose,
            source=source,
            metadata=metadata,
            dataset_id=dataset_id,
        )
        await self.routing_table.register_dataset(
            RegisterDatasetRequest(
                purpose=purpose,
                source=source,
                metadata=metadata,
                dataset_id=dataset_id,
            )
        )

    async def iterrows(self, request: IterRowsRequest) -> PaginatedResponse:
        logger.debug(
            "DatasetIORouter.iterrows: , start_index= limit",
            dataset_id=request.dataset_id,
            start_index=request.start_index,
            limit=request.limit,
        )
        provider: Any = await self.routing_table.get_provider_impl(request.dataset_id)
        return await provider.iterrows(  # ty: ignore[missing-argument,unresolved-attribute]  # provider implementations accept expanded kwargs at runtime
            dataset_id=request.dataset_id,  # ty: ignore[unknown-argument]
            start_index=request.start_index,  # ty: ignore[unknown-argument]
            limit=request.limit,  # ty: ignore[unknown-argument]
        )

    async def append_rows(self, params: AppendRowsParams) -> None:
        logger.debug("DatasetIORouter.append_rows", dataset_id=params.dataset_id, rows_count=len(params.rows))
        provider: Any = await self.routing_table.get_provider_impl(params.dataset_id)
        return await provider.append_rows(  # ty: ignore[missing-argument,unresolved-attribute]  # provider implementations accept expanded kwargs at runtime
            dataset_id=params.dataset_id,  # ty: ignore[unknown-argument]
            rows=params.rows,  # ty: ignore[unknown-argument]
        )
