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

"""Lossless Pi v3 branch evidence and durable incremental acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    Source,
    SourceMaterialization,
    TextEvidence,
)


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    event: PiEventInput


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(session_id: str, event_id: str) -> str:
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > 256:
        identity = "pi:sha256:" + hashlib.sha256(_canonical((session_id, event_id)).encode()).hexdigest()
    return identity


def _archive(value: PiSessionInput) -> tuple[str, dict[str, dict[str, Any]], list[str]]:
    try:
        records = [json.loads(line) for line in value.content.decode("utf-8-sig").splitlines() if line.strip()]
    except (UnicodeError, ValueError) as error:
        raise ValueError("invalid Pi JSONL") from error  # noqa: TRY003
    if not records or not isinstance(records[0], dict):
        raise ValueError("missing Pi session header")  # noqa: TRY003
    header = records[0]
    session_id = header.get("id")
    if header.get("type") != "session" or type(header.get("version")) is not int or header["version"] != 3:
        raise ValueError("expected a Pi v3 session")  # noqa: TRY003
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session id must be nonempty")  # noqa: TRY003
    nodes = _validate_nodes(records[1:])
    leaf = value.leaf_id if value.leaf_id is not None else next(reversed(nodes), None)
    if leaf is not None and leaf not in nodes:
        raise ValueError("Pi leaf does not exist")  # noqa: TRY003
    path: list[str] = []
    while leaf is not None:
        path.append(leaf)
        leaf = nodes[leaf]["parentId"]
    return session_id, nodes, list(reversed(path))


def _validate_nodes(records: list[Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Pi node must be an object")  # noqa: TRY003, TRY004
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip() or event_id in nodes:
            raise ValueError("Pi node ids must be nonempty and unique")  # noqa: TRY003
        if "parentId" not in record or not isinstance(record.get("type"), str):
            raise ValueError("Pi node requires parentId and type")  # noqa: TRY003
        parent = record["parentId"]
        if parent is not None and (not isinstance(parent, str) or parent not in nodes):
            raise ValueError("Pi parent must precede its child")  # noqa: TRY003
        # Also reject nonfinite numbers, which are not JSON evidence values.
        _canonical(record)
        nodes[event_id] = record
    return nodes


def _blocks(content: object) -> str:
    if isinstance(content, str):
        return content
    blocks: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if block.get("type") == "text" and isinstance(text, str):
                blocks.append(text)
            elif block.get("type") == "toolCall":
                arguments = _canonical(block.get("arguments", {}))
                blocks.append(f"[tool_call {block.get('name', '')} {block.get('id', '')}] {arguments}")
    return "\n".join(blocks)


def _event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record["type"]
    body, role = "", ""
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message must be an object")  # noqa: TRY003
        role = message.get("role", "")
        body = _blocks(message.get("content"))
        if role == "toolResult" and body:
            body = f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}] {body}"
        elif role == "bashExecution":
            command, output = message.get("command", ""), message.get("output", "")
            body = f"$ {command}\n{output}" if command or output else ""
    elif event_type in {"compaction", "branch_summary"}:
        role, body = event_type, record.get("summary", "")
    elif event_type == "custom_message":
        role, body = "custom", _blocks(record.get("content"))
    if not body:
        return None
    if not isinstance(body, str) or not isinstance(role, str) or not role:
        raise ValueError("Pi visible event requires textual role and content")  # noqa: TRY003
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=record["parentId"],
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    session_id, nodes, path = _archive(value)
    return tuple(event for key in path if (event := _event(session_id, nodes[key])) is not None)


class _PiAdapter:
    input_class = PiEventInput
    source_class = PiEventSource
    name = "pi-event"

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=_identity(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            event=value,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return source.event


class _PiProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        event = source.event
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=event.content,
            metadata={
                key: getattr(event, key) for key in ("session_id", "event_id", "parent_id", "event_type", "role")
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(_PiAdapter(), projections=(_PiProjection(),))


class PiSessionConnector:
    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path, self._leaf = Path(path), leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        value = PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf)
        session_id, nodes, path = _archive(value)
        events = tuple(event for key in path if (event := _event(session_id, nodes[key])) is not None)
        previous = session.checkpoint
        if previous is None:
            imported: dict[str, str] = {}
        elif (
            isinstance(previous, dict)
            and previous.get("session_id") == session_id
            and isinstance(previous.get("records"), dict)
        ):
            imported = dict(previous["records"])
        else:
            raise ValueError("checkpoint does not describe this Pi session")  # noqa: TRY003
        fingerprints = {key: hashlib.sha256(_canonical(record).encode()).hexdigest() for key, record in nodes.items()}
        for key, digest in imported.items():
            if fingerprints.get(key) != digest:
                raise ValueError("previously imported Pi event was removed or rewritten")  # noqa: TRY003
        proposed = dict(imported)
        for event in events:
            if event.event_id not in imported:
                await session.submit(_identity(session_id, event.event_id), "pi-event", event)
                proposed[event.event_id] = fingerprints[event.event_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE, checkpoint={"session_id": session_id, "records": proposed}
        )


__all__ = ["PI_EVENT_SOURCE_DEFINITION", "PiEventInput", "PiSessionConnector", "PiSessionInput", "parse_pi_session"]
