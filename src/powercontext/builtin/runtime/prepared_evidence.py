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

"""Query-aware selection of literal, cited sentence windows under a byte budget."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from powercontext.artifacts import ArtifactAddress
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_context import (
    MemoryEntryAddress,
    PreparedContextBuild,
    PreparedEvidenceFragment,
    PreparedMemoryCandidates,
)
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

_STOP = frozenset([
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "and",
    "or",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "their",
    "has",
    "have",
    "had",
    "can",
    "could",
    "may",
    "might",
    "will",
    "would",
    "do",
    "does",
    "did",
    "not",
    "no",
    "than",
    "into",
    "through",
    "between",
    "during",
    "more",
    "less",
    "most",
    "also",
    "both",
    "such",
    "using",
    "used",
    "use",
])


def _terms(text: str) -> list[str]:
    words = re.findall(r"[^\W_]+", text.lower(), re.UNICODE)
    result = []
    for word in words:
        if word in _STOP:
            continue
        # Conservative English normalization also preserves biomedical identifiers.
        for suffix in ("ation", "ments", "ment", "ing", "ies", "ed", "es", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
                break
        result.append(word)
    return result


@dataclass(frozen=True)
class _Document:
    address: MemoryEntryAddress
    text: str
    sentences: tuple[tuple[int, int], ...]


def _sentences(text: str) -> tuple[tuple[int, int], ...]:
    spans = []
    start = 0
    # Split only at a sentence boundary; decimal points and initials stay intact.
    for match in re.finditer(r"[.!?。！？](?:[\"')\]]*)(?=\s+|$)|\n+", text):  # noqa: RUF001
        end = match.end()
        while start < end and text[start].isspace():
            start += 1
        if start < end and text[start:end].strip():
            spans.append((start, end))
        start = end
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return tuple(spans)


def _render(fragments: Sequence[PreparedEvidenceFragment]) -> str:
    items = [
        {
            "citation": {
                "scope_id": part.address.memory.scope_id,
                "memory_ref": part.address.memory.artifact.model_dump(mode="json"),
                "entry_id": part.address.entry_id,
                "entry_version_id": part.address.entry_version_id,
            },
            "content": part.text,
            "start": part.start,
            "end": part.end,
        }
        for part in fragments
    ]
    encoded = json.dumps({"trust": "untrusted_history", "items": items}, ensure_ascii=False, separators=(",", ":"))
    return (
        TRUST_POLICY
        + "\n\nBEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1\n"
        + encoded
        + "\nEND_POWERCONTEXT_PREPARED_CONTEXT_V1"
    )


def _documents(candidates: Sequence[PreparedMemoryCandidates]) -> list[_Document]:
    documents = []
    seen = set()
    for group in candidates:
        for hit in group.hits:
            identity = (
                group.scope_id,
                hit.memory_ref.artifact_id,
                hit.memory_ref.revision,
                hit.entry_id,
                hit.entry_version_id,
            )
            if identity in seen:
                continue
            seen.add(identity)
            documents.append(
                _Document(
                    MemoryEntryAddress(
                        ArtifactAddress(scope_id=group.scope_id, artifact=hit.memory_ref),
                        hit.entry_id,
                        hit.entry_version_id,
                    ),
                    hit.text,
                    _sentences(hit.text),
                )
            )
    return documents


def _scores(documents: list[_Document], query: set[str]) -> list[list[float]]:
    counts = [Counter(_terms(document.text)) for document in documents]
    inverse = {
        term: 1 + math.log((len(documents) + 1) / (1 + sum(term in count for count in counts))) for term in query
    }
    doc_scores = [sum(inverse[term] * min(count[term], 3) / 3 for term in query) for count in counts]
    best_doc = max(doc_scores, default=1) or 1
    scores = []
    for document, doc_score in zip(documents, doc_scores, strict=True):
        rows = []
        for start, end in document.sentences:
            words = Counter(_terms(document.text[start:end]))
            overlap = sum(inverse[term] * min(words[term], 2) for term in query)
            # A matched result sentence is more useful than a repeated title alone.
            outcome = bool(
                re.search(
                    r"\b(result|conclu|found|show|demonstrat|associat|increas|decreas|reduc|inhibit|induc|significant)",
                    document.text[start:end],
                    re.I,
                )
            )
            rows.append(
                overlap
                / (1 + len(list(words.elements())) / 90)
                * (0.7 + 0.3 * doc_score / best_doc)
                * (1.12 if outcome else 1)
            )
        scores.append(rows)
    return scores


def _select(
    request: PrepareContextRequest, documents: list[_Document], scores: list[list[float]]
) -> list[PreparedEvidenceFragment]:
    selected: list[PreparedEvidenceFragment] = []
    used: set[tuple[int, int]] = set()
    while True:
        best = None
        best_value = 0.0
        for number, document in enumerate(documents):
            rows = scores[number]
            for first in range(len(rows)):
                if rows[first] <= 0 or (number, first) in used:
                    continue
                for last in range(first, min(first + 10, len(rows))):
                    if any((number, index) in used for index in range(first, last + 1)):
                        break
                    start, end = document.sentences[first][0], document.sentences[last][1]
                    fragment = PreparedEvidenceFragment(document.address, start, end, document.text[start:end])
                    trial = [*selected, fragment]
                    size = len(_render(trial).encode("utf-8"))
                    if size > request.max_bytes:
                        break
                    # Overlapping query terms and whole result sentences earn utility;
                    # zero-match boilerplate cannot improve a window's score.
                    value = sum(rows[first : last + 1])
                    if value > best_value or (value == best_value and best is not None and size < best[3]):
                        best_value = value
                        best = (number, first, last, size, fragment)
        if best is None:
            break
        number, first, last, _, fragment = best
        selected.append(fragment)
        used.update((number, index) for index in range(first, last + 1))
    return selected


def select_evidence(
    request: PrepareContextRequest, candidates: Sequence[PreparedMemoryCandidates]
) -> PreparedContextBuild:
    documents = _documents(candidates)
    query = set(_terms(request.query))
    selected = _select(request, documents, _scores(documents, query)) if documents and query else []
    if not selected:
        return PreparedContextBuild(PreparedContext(status="empty", content=None, content_bytes=0), ())
    content = _render(selected)
    return PreparedContextBuild(
        PreparedContext(status="ready", content=content, content_bytes=len(content.encode("utf-8"))),
        tuple(part.address for part in selected),
        fragments=tuple(selected),
    )
