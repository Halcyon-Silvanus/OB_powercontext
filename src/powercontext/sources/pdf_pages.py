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

    _validate_document_id = field_validator("document_id")(PDFDocumentInput.validate_document_id.__func__)


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


_FINANCIAL_QUERY_FILLER = frozenset([
    "reported",
    "report",
    "materially",
    "important",
    "ongoing",
    "directly",
    "outlined",
    "explicitly",
    "state",
    "assuming",
    "perspective",
    "investment",
    "analyst",
    "details",
    "shown",
    "give",
    "response",
    "relying",
    "calculate",
    "calculation",
    "define",
    "answer",
    "respond",
    "question",
    "primarily",
    "quantity",
    "percentage",
    "change",
    "percent",
    "usd",
    "million",
    "millions",
    "billion",
    "billions",
    "fy",
])


def _pdf_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z]+|\d+", text.casefold())
    normalized = []
    for word in words:
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 4 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        if len(word) > 4 and word.endswith("e"):
            word = word[:-1]
        normalized.append(word)
    return normalized


def _pdf_query(question: str) -> dict[str, float]:
    content = analyze_fts_query(question)
    words = [word for word in re.findall(r"[a-z]+|\d+", content.casefold()) if word not in _FINANCIAL_QUERY_FILLER]
    weighted = {word: (0.12 if word.isdigit() else 1.0) for word in _pdf_terms(" ".join(words))}
    expansions = (
        (r"\bcapex\b|capital expenditure", "capital expenditures purchases property plant equipment"),
        (r"\bebitda\b|\bd\s*&\s*a\b", "earnings operating income depreciation amortization"),
        (r"income statement|\bp\s*&\s*l\b", "consolidated statements income earnings operations"),
        (r"cash flow", "consolidated statements cash flows operating investing financing"),
        (r"\blegal\b|litigation|lawsuit", "legal proceedings litigation contingencies claims"),
        (r"geograph|\bcountries\b|\bregions\b", "geographic international countries regions operations segments"),
        (r"\brevenue\b|\brevenues\b", "revenue revenues net sales"),
        (r"\bprofit\b", "profit income earnings"),
        (r"\bdebt\b", "debt borrowings obligations maturities"),
    )
    for pattern, expansion in expansions:
        if re.search(pattern, question, re.IGNORECASE):
            for word in _pdf_terms(expansion):
                weighted.setdefault(word, 0.45)
    return weighted


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Rank scoped, persisted pages with whole-corpus lexical statistics.

    Native search supplies initial relevance; the public Memory listing closes
    candidate truncation gaps for long questions and tables using different
    financial terminology. No secondary evidence store or external model is used.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003
    weighted = _pdf_query(question)
    if not weighted:
        return ()
    page = await memory.list()
    if not page.entries:
        return ()
    native = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=100))
    native_rank = {(hit.entry_id, hit.entry_version_id): rank for rank, hit in enumerate(native.hits)}
    bodies = [record.entry.text.partition("\n") for record in page.entries]
    frequencies = [Counter(_pdf_terms(body or identifier)) for identifier, _, body in bodies]
    lengths = [sum(counts.values()) for counts in frequencies]
    average = sum(lengths) / len(lengths) or 1
    document_frequency = Counter(term for counts in frequencies for term in weighted if term in counts)
    inverse = {
        term: math.log(1 + (len(frequencies) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in weighted
    }
    question_slug = re.sub(r"[^a-z0-9]", "", question.casefold())
    years = re.findall(r"(?:fy)?((?:19|20)\d{2})", question.casefold())
    requested_year = max(years, default="")

    def score(index: int) -> float:
        counts = frequencies[index]
        normalization = 1.2 * (0.45 + 0.55 * lengths[index] / average)
        lexical = sum(
            weighted[term] * inverse[term] * counts[term] * 2.2 / (counts[term] + normalization)
            for term in weighted
            if counts[term]
        )
        identifier = bodies[index][0].casefold()
        company = re.split(r"(?:19|20)\d{2}", identifier, maxsplit=1)[0]
        company_slug = re.sub(r"[^a-z0-9]", "", company)
        if len(company_slug) >= 2 and company_slug in question_slug:
            lexical *= 3.0
        if requested_year and requested_year in identifier:
            lexical *= 1.3
        record = page.entries[index]
        rank = native_rank.get((record.entry.entry_id, record.entry.entry_version_id))
        return lexical + (0 if rank is None else 0.1 / (rank + 1))

    order = sorted(range(len(page.entries)), key=lambda index: (-score(index), index))
    return tuple(page.entries[index].citation for index in order[:limit] if score(index) > 0)


__all__ = ["PDF_PAGE_SOURCE_DEFINITION", "PDFDocumentInput", "PDFPageInput", "parse_pdf_document", "search_pdf_pages"]
