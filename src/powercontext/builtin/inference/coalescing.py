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

"""Share in-flight document embeddings and retain a bounded LRU of successes."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict

from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


class CoalescingEmbeddingModel:
    """Coordinate calls on one event loop without transferring cancellation.

    Each in-flight text resolves with its batch result as well as its vector.
    Keeping that result lets every caller charge each batch once, even when it
    requested several of that batch's texts or joined after dispatch.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")  # noqa: TRY003
        if type(coalesce_ms) not in (int, float) or not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be finite and nonnegative")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._delay = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: dict[str, asyncio.Future[tuple[EmbeddingVector, EmbeddingResult]]] = {}
        self._queued: list[str] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()

    @property
    def profile(self) -> EmbeddingProfile:
        """Expose the underlying model's unchanged embedding profile."""
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Preserve input order and duplicates while sharing distinct work."""
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        loop = asyncio.get_running_loop()
        cached: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[tuple[EmbeddingVector, EmbeddingResult]]] = {}
        for text in texts:
            if text in self._cache:
                self._cache.move_to_end(text)
                cached[text] = self._cache[text]
            else:
                future = self._pending.get(text)
                if future is None:
                    future = loop.create_future()
                    # A cancelled last waiter must not leave an unobserved error.
                    future.add_done_callback(_observe_exception)
                    self._pending[text] = future
                    self._queued.append(text)
                waiting[text] = future
        if self._queued and self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush())

        resolved = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        batches: dict[int, EmbeddingResult] = {}
        for text, (vector, batch) in zip(waiting, resolved, strict=True):
            cached[text] = vector
            batches[id(batch)] = batch
        usages = [batch.usage for batch in batches.values()]
        return EmbeddingResult(
            vectors=tuple(cached[text] for text in texts),
            usage=InferenceUsage(
                requests=sum(usage.requests for usage in usages),
                input_tokens=_sum_optional([usage.input_tokens for usage in usages]),
                output_tokens=_sum_optional([usage.output_tokens for usage in usages]),
            ),
        )

    async def _flush(self) -> None:
        await asyncio.sleep(self._delay)
        queued, self._queued = self._queued, []
        self._flush_task = None
        for offset in range(0, len(queued), self._max_batch_size):
            texts = tuple(queued[offset : offset + self._max_batch_size])
            task = asyncio.create_task(self._run_batch(texts))
            self._batches.add(task)
            task.add_done_callback(self._batches.discard)

    async def _run_batch(self, texts: tuple[str, ...]) -> None:
        try:
            result = await self._model.embed(texts)
            if len(result.vectors) != len(texts):
                raise ValueError("embedding vector count does not match the batch")  # noqa: TRY003, TRY301
        except BaseException as error:
            for text in texts:
                self._pending.pop(text).set_exception(error)
        else:
            for text, vector in zip(texts, result.vectors, strict=True):
                if self._cache_size:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                self._pending.pop(text).set_result((vector, result))


def _observe_exception(future: asyncio.Future[tuple[EmbeddingVector, EmbeddingResult]]) -> None:
    if not future.cancelled():
        future.exception()


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


__all__ = ["CoalescingEmbeddingModel"]
