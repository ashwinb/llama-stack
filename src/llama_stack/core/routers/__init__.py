# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, cast

from llama_stack.core.datatypes import (
    AccessRule,
    RoutedProtocol,
    StackConfig,
)
from llama_stack.core.routing_tables.common import CommonRoutingTableImpl
from llama_stack.core.store import DistributionRegistry
from llama_stack.providers.utils.inference.inference_store import InferenceStore
from llama_stack_api import Api


async def get_routing_table_impl(
    api: Api,
    impls_by_provider_id: dict[str, RoutedProtocol],
    _deps: dict[str, Any],
    dist_registry: DistributionRegistry,
    policy: list[AccessRule],
) -> CommonRoutingTableImpl:
    from ..routing_tables.benchmarks import BenchmarksRoutingTable
    from ..routing_tables.datasets import DatasetsRoutingTable
    from ..routing_tables.models import ModelsRoutingTable
    from ..routing_tables.scoring_functions import ScoringFunctionsRoutingTable
    from ..routing_tables.shields import ShieldsRoutingTable
    from ..routing_tables.toolgroups import ToolGroupsRoutingTable
    from ..routing_tables.vector_stores import VectorStoresRoutingTable

    api_to_tables: dict[str, type[CommonRoutingTableImpl]] = {
        "models": ModelsRoutingTable,
        "shields": ShieldsRoutingTable,
        "datasets": DatasetsRoutingTable,
        "scoring_functions": ScoringFunctionsRoutingTable,
        "benchmarks": BenchmarksRoutingTable,
        "tool_groups": ToolGroupsRoutingTable,
        "vector_stores": VectorStoresRoutingTable,
    }

    if api.value not in api_to_tables:
        raise ValueError(f"API {api.value} not found in router map")

    impl: CommonRoutingTableImpl = api_to_tables[api.value](impls_by_provider_id, dist_registry, policy)

    await impl.initialize()
    return impl


async def get_auto_router_impl(
    api: Api,
    routing_table: CommonRoutingTableImpl,
    deps: dict[Api, Any],
    run_config: StackConfig,
    policy: list[AccessRule],
) -> RoutedProtocol:
    from ..routing_tables.benchmarks import BenchmarksRoutingTable
    from ..routing_tables.datasets import DatasetsRoutingTable
    from ..routing_tables.models import ModelsRoutingTable
    from ..routing_tables.scoring_functions import ScoringFunctionsRoutingTable
    from ..routing_tables.shields import ShieldsRoutingTable
    from ..routing_tables.toolgroups import ToolGroupsRoutingTable
    from ..routing_tables.vector_stores import VectorStoresRoutingTable
    from .datasets import DatasetIORouter
    from .eval_scoring import EvalRouter, ScoringRouter
    from .inference import InferenceRouter
    from .safety import SafetyRouter
    from .tool_runtime import ToolRuntimeRouter
    from .vector_io import VectorIORouter

    api_to_dep_impl: dict[str, Any] = {}
    # TODO: move pass configs to routers instead
    if api == Api.inference:
        inference_ref = run_config.storage.stores.inference
        if not inference_ref:
            raise ValueError("storage.stores.inference must be configured in run config")

        inference_store = InferenceStore(
            reference=inference_ref,
            policy=policy,
        )
        await inference_store.initialize()
        api_to_dep_impl["store"] = inference_store
    elif api == Api.vector_io:
        api_to_dep_impl["vector_stores_config"] = run_config.vector_stores
        api_to_dep_impl["inference_api"] = deps.get(Api.inference)
    elif api == Api.safety:
        api_to_dep_impl["safety_config"] = run_config.safety

    impl: RoutedProtocol
    if api == Api.vector_io:
        impl = VectorIORouter(cast(VectorStoresRoutingTable, routing_table), **api_to_dep_impl)
    elif api == Api.inference:
        impl = InferenceRouter(cast(ModelsRoutingTable, routing_table), **api_to_dep_impl)
    elif api == Api.safety:
        impl = SafetyRouter(cast(ShieldsRoutingTable, routing_table), **api_to_dep_impl)
    elif api == Api.datasetio:
        impl = DatasetIORouter(cast(DatasetsRoutingTable, routing_table))
    elif api == Api.scoring:
        impl = ScoringRouter(cast(ScoringFunctionsRoutingTable, routing_table))
    elif api == Api.eval:
        impl = EvalRouter(cast(BenchmarksRoutingTable, routing_table))
    elif api == Api.tool_runtime:
        impl = ToolRuntimeRouter(cast(ToolGroupsRoutingTable, routing_table))
    else:
        raise ValueError(f"API {api.value} not found in router map")

    await impl.initialize()
    return impl
