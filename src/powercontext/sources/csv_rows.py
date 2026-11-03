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

"""Captured CSV records with stable primary-key identities and exact evidence."""

from __future__ import annotations

import csv
import io
import json
from typing import Self
from urllib.parse import quote

from pydantic import BaseModel, field_validator, model_validator

from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    Source,
    SourceMaterialization,
    TextEvidence,
)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Encode identity components independently without normalizing their values."""
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be nonempty without surrounding whitespace")  # noqa: TRY003
    if not primary_key_value.strip():
        raise ValueError("primary key value must not be blank")  # noqa: TRY003
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > 256:
        raise ValueError("CSV row identity exceeds 256 characters")  # noqa: TRY003
    return identity


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document and its exact primary-key column."""

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("document_id must be nonempty without surrounding whitespace")  # noqa: TRY003
        return value

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must be nonempty")  # noqa: TRY003
        return value


class CSVRowInput(BaseModel):
    """A validated record preserving original column and value order."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        csv_row_source_id(self.document_id, self.primary_key_value)
        if not self.columns or any(not column.strip() for column in self.columns):
            raise ValueError("CSV columns must not be empty or blank")  # noqa: TRY003
        if len(set(self.columns)) != len(self.columns) or len(self.columns) != len(self.values):
            raise ValueError("CSV columns must be unique and match the row width")  # noqa: TRY003
        if self.primary_key not in self.columns:
            raise ValueError("primary_key must exactly match a column")  # noqa: TRY003
        if self.values[self.columns.index(self.primary_key)] != self.primary_key_value:
            raise ValueError("primary_key_value does not match the record")  # noqa: TRY003
        return self


class CSVRowSource(CSVRowInput, Source):
    """One durable captured row, independently addressable from other rows."""

    source_id: str

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        identity = csv_row_source_id(self.document_id, self.primary_key_value)
        if (
            self.source_id != identity
            or self.name != identity
            or self.materialization != SourceMaterialization.CAPTURED
        ):
            raise ValueError("CSV Source must be captured under its row identity")  # noqa: TRY003
        return self


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Validate the entire document before returning any captured records."""
    try:
        reader = csv.reader(io.StringIO(value.content.decode("utf-8-sig"), newline=""), strict=True)
        columns = tuple(next(reader, ()))
        if not columns or any(not column.strip() for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV requires a nonempty unique header")  # noqa: TRY003
        if value.primary_key not in columns:
            raise ValueError("primary_key must exactly match a column")  # noqa: TRY003
        key_index = columns.index(value.primary_key)
        seen: set[str] = set()
        rows: list[CSVRowInput] = []
        for record in reader:
            if not record:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV record width does not match the header")  # noqa: TRY003
            key = record[key_index]
            if key in seen:
                raise ValueError("duplicate CSV primary key")  # noqa: TRY003
            rows.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=key,
                    columns=columns,
                    values=tuple(record),
                )
            )
            seen.add(key)
    except csv.Error as error:
        raise ValueError("malformed CSV document") from error  # noqa: TRY003
    return tuple(rows)


class _CSVRowAdapter:
    input_class = CSVRowInput
    source_class = CSVRowSource
    name = "csv-row"

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            **value.model_dump(), name=identity, source_id=identity, materialization=SourceMaterialization.CAPTURED
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput.model_validate(source.model_dump())


class _CSVTextEvidence:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(_CSVRowAdapter(), projections=(_CSVTextEvidence(),))

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
