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

"""Complete PDF page evidence and offline lexical retrieval through Memory."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.artifacts.search import analyze_fts_query
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    Source,
    SourceMaterialization,
    TextEvidence,
)


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(gt=0, strict=True)
    text: str


class PDFPageSource(Source):
    page: PDFPageInput


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    from pypdf import PdfReader

    if not value.content or b"%PDF-" not in value.content[:1024]:
        raise ValueError("expected PDF bytes")  # noqa: TRY003
    try:
        reader = PdfReader(io.BytesIO(value.content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a password")  # noqa: TRY003, TRY301
        digest = hashlib.sha256(value.content).hexdigest()
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id, document_sha256=digest, page_number=index, text=page.extract_text() or ""
            )
            for index, page in enumerate(reader.pages, start=1)
        )
        if not pages:
            raise ValueError("PDF contains no pages")  # noqa: TRY003, TRY301
    except Exception as error:
        raise ValueError("PDF could not be completely parsed") from error  # noqa: TRY003
    return pages


class _PDFAdapter:
    name = "pdf-page"
    input_class = PDFPageInput
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = f"pdf:{quote(value.document_id, safe='')}:{value.document_sha256}:{value.page_number}"
        if len(identity) > 256:
            identity = (
                "pdf:sha256:"
                + hashlib.sha256(value.document_id.encode()).hexdigest()
                + f":{value.document_sha256}:{value.page_number}"
            )
        return PDFPageSource(name=identity, materialization=SourceMaterialization.CAPTURED, page=value)

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return source.page


class _PDFProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        page = source.page
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{page.document_id}\n{page.text}",
            metadata={
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(_PDFAdapter(), projections=(_PDFProjection(),))


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Collect native lexical candidates, then rank complete pages using BM25.

    Single-term searches complement the complete query so an informative rare
    term is not lost to query-coverage admission. All returned identities come
    from the caller's scoped Memory; no independent evidence store is used.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003
    terms = list(dict.fromkeys(re.findall(r"[a-z]+|\d+", analyze_fts_query(question).lower())))
    if not terms:
        return ()
    candidates = {}
    for query in dict.fromkeys([question, *terms]):
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=max(100, limit * 5)))
        for hit in page.hits:
            # ArtifactRef may be an unfrozen Pydantic model.
            key = (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id, hit.entry_version_id)
            candidates[key] = hit
    if not candidates:
        return ()
    frequencies = {key: Counter(re.findall(r"[a-z]+|\d+", hit.text.lower())) for key, hit in candidates.items()}
    lengths = {key: sum(counts.values()) for key, counts in frequencies.items()}
    average = sum(lengths.values()) / len(lengths) or 1
    document_frequency = Counter(term for counts in frequencies.values() for term in terms if term in counts)
    inverse = {
        term: math.log(1 + (len(candidates) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in terms
    }

    def score(key):
        counts = frequencies[key]
        normalization = 1.2 * (0.25 + 0.75 * lengths[key] / average)
        return sum(
            inverse[term] * counts[term] * 2.2 / (counts[term] + normalization) for term in terms if counts[term]
        )

    ordered = sorted(candidates, key=lambda key: (-score(key), key))
    return tuple(
        MemoryCitation(memory_ref=candidates[key].memory_ref, entry_id=key[2], entry_version_id=key[3])
        for key in ordered[:limit]
    )


__all__ = ["PDF_PAGE_SOURCE_DEFINITION", "PDFDocumentInput", "PDFPageInput", "parse_pdf_document", "search_pdf_pages"]
