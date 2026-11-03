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

"""Canonical, independently verifiable context packs and prefix deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """A context pack violates the wire or reconstruction contract."""


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_items(items: tuple[bytes, ...]) -> None:
    if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
        raise ContextPackError("items must be a tuple of bytes")  # noqa: TRY003
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextPackError("each item must be valid UTF-8") from error  # noqa: TRY003


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decoded(value: Any) -> tuple[bytes, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContextPackError("wire items must be an array of Base64 strings")  # noqa: TRY003
    try:
        items = tuple(base64.b64decode(item, validate=True) for item in value)
    except (ValueError, binascii.Error) as error:
        raise ContextPackError("invalid Base64 encoding") from error  # noqa: TRY003
    if _encoded(items) != value:
        raise ContextPackError("noncanonical Base64 encoding")  # noqa: TRY003
    _validate_items(items)
    return items


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContextPackError("duplicate JSON field")  # noqa: TRY003
        value[key] = item
    return value


def _read(data: bytes, schema: str, fields: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeDecodeError, AttributeError) as error:
        raise ContextPackError("invalid UTF-8 JSON") from error  # noqa: TRY003
    if not isinstance(value, dict) or set(value) != fields or value.get("schema") != schema:
        raise ContextPackError("invalid schema or fields")  # noqa: TRY003
    return value


def _validate_digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ContextPackError("invalid SHA-256 digest")  # noqa: TRY003


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete context represented by ordered UTF-8 byte chunks."""

    items: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _validate_items(self.items)

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        return cls(items)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content must be Unicode text")  # noqa: TRY003
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        chunks: list[bytes] = []
        chunk = bytearray()
        try:
            for character in content:
                encoded = character.encode("utf-8")
                if len(encoded) > chunk_bytes:
                    raise ContextPackError("chunk limit cannot hold a UTF-8 character")  # noqa: TRY003
                if len(chunk) + len(encoded) > chunk_bytes:
                    chunks.append(bytes(chunk))
                    chunk.clear()
                chunk.extend(encoded)
        except (UnicodeEncodeError, TypeError) as error:
            raise ContextPackError("content must be valid Unicode text") from error  # noqa: TRY003
        if chunk:
            chunks.append(bytes(chunk))
        return cls.build(tuple(chunks))

    @property
    def digest(self) -> str:
        unsigned = {"schema": _PACK_SCHEMA, "items": _encoded(self.items)}
        return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()

    def to_bytes(self) -> bytes:
        payload = {"schema": _PACK_SCHEMA, "items": _encoded(self.items), "digest": self.digest, "byte_count": 0}
        while True:
            encoded = _canonical(payload)
            if payload["byte_count"] == len(encoded):
                return encoded
            payload["byte_count"] = len(encoded)

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        value = _read(data, _PACK_SCHEMA, {"schema", "items", "digest", "byte_count"})
        pack = cls.build(_decoded(value["items"]))
        if (
            value["digest"] != pack.digest
            or type(value["byte_count"]) is not int
            or value["byte_count"] != pack.byte_count
        ):
            raise ContextPackError("pack digest or byte count mismatch")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A base identity, retained prefix and replacement suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.base_digest)
        _validate_digest(self.target_digest)
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix_items must be a nonnegative integer")  # noqa: TRY003
        _validate_items(self.appended)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for first, second in zip(base.items, target.items, strict=False):
            if first != second:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        return _canonical({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        value = _read(data, _DELTA_SCHEMA, {"schema", "base_digest", "target_digest", "prefix_items", "appended"})
        return cls(value["base_digest"], value["target_digest"], value["prefix_items"], _decoded(value["appended"]))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Reconstruct solely from the supplied base and validated delta."""
    if base.digest != delta.base_digest or delta.prefix_items > len(base.items):
        raise ContextPackError("base digest or prefix bound mismatch")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("target digest mismatch")  # noqa: TRY003
    return target


__all__ = ["ContextPack", "ContextPackDelta", "ContextPackError", "apply_context_pack_delta"]
