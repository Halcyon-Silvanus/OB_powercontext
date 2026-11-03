# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Durable, scope-local evidence dependencies and native Memory availability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime.models import CommitConnectorCheckpoint, GetMemoryEntryRequest
from powercontext.sources.connectors import ConnectorBinding
from powercontext.sources.models import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """An evidence binding or lifecycle event violates the registered graph."""


class SourceLifecycleEvent(BaseModel):
    """One strictly typed, replayable state change of a registered Source."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    node_id: str
    reason: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle identity and reason must be nonblank")  # noqa: TRY003
        return value


class _Node(BaseModel):
    kind: Literal["source", "memory", "topic"]
    reference: SourceRef | MemoryCitation | ArtifactRef
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int = 1
    available: bool = True
    event: SourceLifecycleEvent | None = None


class _Graph(BaseModel):
    # Order is persisted explicitly: JSON object key order is not a graph contract.
    order: list[str] = Field(default_factory=list)
    nodes: dict[str, _Node] = Field(default_factory=dict)


class SourceLifecycleApplication:
    """Create evidence services sharing the Runtime's durable scope storage."""

    def __init__(self, runtime: BuiltinRuntime, *, storage_available: bool = True) -> None:
        self._runtime = runtime
        self._storage_available = storage_available

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id, storage_available=self._storage_available)


class ScopedSourceLifecycle:
    """Single-writer DAG with durable desired state and idempotent propagation."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str, *, storage_available: bool = True) -> None:
        self._runtime = runtime
        self._storage_available = storage_available
        self._binding = ConnectorBinding(
            scope_id=scope_id, binding_id="source-lifecycle", connector_name="source-lifecycle", connector_version="1"
        )
        self.scope_id = scope_id

    async def _load(self) -> tuple[JsonValue | None, _Graph]:
        state = await self._runtime.ingestion.checkpoint(self._binding)
        return state.checkpoint, _Graph() if state.checkpoint is None else _Graph.model_validate(state.checkpoint)

    async def _save(self, expected: JsonValue | None, graph: _Graph) -> None:
        await self._runtime.ingestion.commit(
            CommitConnectorCheckpoint(
                binding=self._binding, expected=expected, checkpoint=graph.model_dump(mode="json")
            )
        )

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        if type(revision) is not int or revision < 1 or not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("invalid source binding")  # noqa: TRY003
        await self._register(node_id, _Node(kind="source", reference=source_ref, revision=revision))

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: Literal["all", "any"]
    ) -> None:
        self._validate_dependencies(parents, policy)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("invalid memory citation")  # noqa: TRY003
        await self._register(node_id, _Node(kind="memory", reference=citation, parents=parents, policy=policy))

    async def register_topic(
        self, node_id: str, artifact_ref: ArtifactRef, *, parents: tuple[str, ...], policy: Literal["all", "any"]
    ) -> None:
        self._validate_dependencies(parents, policy)
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("invalid topic reference")  # noqa: TRY003
        await self._register(node_id, _Node(kind="topic", reference=artifact_ref, parents=parents, policy=policy))

    @staticmethod
    def _validate_dependencies(parents: tuple[str, ...], policy: str) -> None:
        if (
            policy not in {"all", "any"}
            or isinstance(parents, str)
            or any(not isinstance(parent, str) or not parent.strip() for parent in parents)
        ):
            raise SourceLifecycleError("invalid dependency policy or parents")  # noqa: TRY003

    async def _register(self, node_id: str, node: _Node) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise SourceLifecycleError("node_id must be nonblank")  # noqa: TRY003
        previous, graph = await self._load()
        existing = graph.nodes.get(node_id)
        if existing is not None:
            if existing.model_copy(update={"available": True, "event": None}) != node:
                raise SourceLifecycleError("node binding conflicts with existing registration")  # noqa: TRY003
            await self._propagate(graph, "registration retry")
            return
        if node.kind != "source":
            if (
                not node.parents
                or len(set(node.parents)) != len(node.parents)
                or any(parent == node_id or parent not in graph.nodes for parent in node.parents)
            ):
                raise SourceLifecycleError("parents must be distinct, previously registered nodes")  # noqa: TRY003
            node.available = self._supported(node, graph)
        await self._validate_reference(node)
        graph.order.append(node_id)
        graph.nodes[node_id] = node
        # Persist intent before any native writes, so retries finish partial propagation.
        await self._save(previous, graph)
        await self._propagate(graph, "evidence registration")

    async def _validate_reference(self, node: _Node) -> None:
        reference = node.reference
        try:
            if isinstance(reference, SourceRef):
                await self._runtime.records.for_scope(self.scope_id).get_source(
                    reference.source_type, reference.source_id
                )
            elif isinstance(reference, MemoryCitation):
                await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=reference))
            else:
                await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                    reference.family, reference.artifact_id, reference.revision
                )
        except Exception as error:
            raise SourceLifecycleError("registered reference is not readable in this scope") from error  # noqa: TRY003

    @staticmethod
    def _supported(node: _Node, graph: _Graph) -> bool:
        states = (graph.nodes[parent].available for parent in node.parents)
        return all(states) if node.policy == "all" else any(states)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        previous, graph = await self._load()
        node = graph.nodes.get(event.node_id)
        if node is None or node.kind != "source" or node.revision != event.revision:
            raise SourceLifecycleError("unknown source or lifecycle revision mismatch")  # noqa: TRY003
        if node.event is not None and (
            event.sequence < node.event.sequence or (event.sequence == node.event.sequence and event != node.event)
        ):
            raise SourceLifecycleError("event sequence conflicts with accepted history")  # noqa: TRY003
        if node.event != event:
            node.available = event.available
            node.event = event
            for node_id in graph.order:
                child = graph.nodes[node_id]
                if child.kind != "source":
                    child.available = self._supported(child, graph)
            await self._save(previous, graph)
        # A replay checks actual native state rather than trusting a completion bit.
        await self._propagate(graph, event.reason)
        return sorted(node_id for node_id, item in graph.nodes.items() if item.available)

    async def _propagate(self, graph: _Graph, reason: str) -> None:
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            service = context.artifacts.memory
            memories = [node for node in graph.nodes.values() if isinstance(node.reference, MemoryCitation)]
            if not memories:
                return
            current = await service.head(context.artifacts.memory_artifact_id)
            entries = {entry.entry_id: entry for entry in await service.entries(current)}
            for node in memories:
                citation = node.reference
                if not isinstance(citation, MemoryCitation):
                    continue
                entry = entries[citation.entry_id]
                desired = "active" if node.available else "inactive"
                state = next(item.state for item in current.content.manifest.entries if item.entry_id == entry.entry_id)
                if state == desired:
                    continue
                operation = service.reactivate if node.available else service.forget
                current = await operation(current, entries=(entry,), reason=reason)

    async def resolve(
        self, node_id: str, *, purpose: Literal["generation", "audit"]
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        _, graph = await self._load()
        node = graph.nodes.get(node_id)
        if node is None or purpose not in {"generation", "audit"} or (purpose == "generation" and not node.available):
            raise SourceLifecycleError("node is unavailable for the requested purpose")  # noqa: TRY003
        return node.reference

    async def unavailable_topics(self) -> frozenset[tuple[str, int]]:
        if not self._storage_available:
            return frozenset()
        _, graph = await self._load()
        return frozenset(
            (node.reference.artifact_id, node.reference.revision)
            for node in graph.nodes.values()
            if node.kind == "topic" and not node.available and isinstance(node.reference, ArtifactRef)
        )
