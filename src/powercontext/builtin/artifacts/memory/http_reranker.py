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

"""HTTP scoring with stable selection across the complete Memory candidate pool."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score bounded batches without changing original candidate identities.

    The caller owns the HTTP client, including its authentication, timeout and
    lifetime. HTTP failures propagate instead of silently using coarse ranks.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return globally ranked candidates after validating every batch."""
        if not candidates or type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("limit must be between one and the nonempty candidate count")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for offset in range(0, len(candidates), self._batch_size):
            batch = candidates[offset : offset + self._batch_size]
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": offset + index, "text": candidate.text}
                        for index, candidate in enumerate(batch, start=1)
                    ],
                },
            )
            requests += 1
            response.raise_for_status()
            scores.update(_batch_scores(response, offset + 1, len(batch)))

        selected = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(selected_ranks=tuple(selected), usage=InferenceUsage(requests=requests))


def _batch_scores(response: httpx.Response, first_rank: int, count: int) -> dict[int, float]:
    payload = response.json()
    scores: dict[int, float] = {}
    if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
        raise ValueError("response must contain only a scores array")  # noqa: TRY003

    expected = set(range(first_rank, first_rank + count))
    observed: set[int] = set()
    for item in payload["scores"]:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each score must contain only rank and score")  # noqa: TRY003
        rank, score = item["rank"], item["score"]
        if type(rank) is not int or rank not in expected or rank in observed:
            raise ValueError("scores must cover each rank in the current batch exactly once")  # noqa: TRY003
        if type(score) not in (int, float):
            raise ValueError("score must be a finite number")  # noqa: TRY003
        try:
            finite = math.isfinite(score)
        except OverflowError as error:
            raise ValueError("score must be a finite number") from error  # noqa: TRY003
        if not finite:
            raise ValueError("score must be a finite number")  # noqa: TRY003
        observed.add(rank)
        scores[rank] = score
    if observed != expected:
        raise ValueError("scores must cover each rank in the current batch exactly once")  # noqa: TRY003
    return scores


__all__ = ["HTTPMemoryReranker"]
