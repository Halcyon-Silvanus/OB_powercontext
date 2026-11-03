"""Recovery of a partially propagated evidence withdrawal."""

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.artifacts.memory.service import MemoryService
from powercontext.builtin.runtime import BuiltinConfig, CaptureSource, RememberMemoryRequest, open_builtin_runtime
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft


def test_partial_native_failure_can_retry_without_revising_completed_entries(monkeypatch):
    asyncio.run(_interrupted_withdrawal(monkeypatch))


async def _interrupted_withdrawal(monkeypatch):
    async with open_builtin_runtime(BuiltinConfig()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Recovery", summary="Durable evidence state", idempotency_key="recovery")
        )
        lifecycle = runtime.lifecycle.for_scope(scope.scope_id)
        source = await runtime.sources.for_scope(scope.scope_id).capture(
            CaptureSource(source_id="origin", content="origin", metadata={})
        )
        await lifecycle.register_source("root", source.source_ref, revision=1)
        memory = runtime.memory.for_scope(scope.scope_id)
        for number in range(2):
            created = await memory.remember(
                RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"fact {number}"),))
            )
            await lifecycle.register_memory(f"fact{number}", created.entry.citation, parents=("root",), policy="all")
        original_forget = MemoryService.forget
        attempts = 0

        async def interrupted(self, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError("simulated storage interruption")  # noqa: TRY003
            return await original_forget(self, *args, **kwargs)

        monkeypatch.setattr(MemoryService, "forget", interrupted)
        event = SourceLifecycleEvent(node_id="root", revision=1, sequence=1, available=False, reason="withdrawn")
        with pytest.raises(RuntimeError, match="storage interruption"):
            await lifecycle.apply(event)
        monkeypatch.setattr(MemoryService, "forget", original_forget)
        assert await lifecycle.apply(event) == []
        state = await memory.list(include_inactive=True)
        assert all(row.state == "inactive" for row in state.entries)
        head = state.memory_ref
        assert await lifecycle.apply(event) == []
        assert (await memory.list(include_inactive=True)).memory_ref == head
