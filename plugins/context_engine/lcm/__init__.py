"""Built-in LCM context-engine spike.

LCM is an opt-in, profile-local context engine.  It persists recoverable raw
conversation turns in SQLite, emits a compact checkpoint when compression fires,
and exposes explicit recall tools.  It is not activated unless a profile sets
``context.engine: lcm``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_messages_tokens_rough
from hermes_constants import get_hermes_home

from .compactor import build_checkpoint
from .redaction import sanitize_for_storage
from .storage import LCMStorage


class LCMContextEngine(ContextEngine):
    """Hermes-native, disabled-by-default LCM context engine."""

    def __init__(self, threshold_percent: float = 0.75, protect_first_n: int = 3, protect_last_n: int = 6):
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.context_length = 200_000
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.session_id = "default"
        self.platform = ""
        self.hermes_home = get_hermes_home()
        self.db_path = self.hermes_home / "context" / "lcm.sqlite3"
        self._storage: LCMStorage | None = None

    @property
    def name(self) -> str:
        return "lcm"

    def is_available(self) -> bool:
        return True

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or self.last_prompt_tokens + self.last_completion_tokens)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return int(tokens or 0) >= self.threshold_tokens

    def should_compress_preflight(self, messages: list[dict[str, Any]]) -> bool:
        return estimate_messages_tokens_rough(messages) >= self.threshold_tokens

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        return len(messages) > self.protect_first_n + self.protect_last_n + 1

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id or "default"
        self.platform = str(kwargs.get("platform") or "")
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME")
        self.hermes_home = Path(hermes_home).expanduser() if hermes_home else get_hermes_home()
        configured_db = kwargs.get("db_path") or kwargs.get("lcm_db_path")
        self.db_path = Path(configured_db).expanduser() if configured_db else self.hermes_home / "context" / "lcm.sqlite3"
        self._storage = LCMStorage(self.db_path)
        self._storage.upsert_conversation(self.session_id, self.platform)

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        if messages:
            self._persist_messages(messages)
        if self._storage is not None:
            self._storage.close()
            self._storage = None

    def on_session_reset(self) -> None:
        super().on_session_reset()

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> list[dict[str, Any]]:
        storage = self._ensure_storage()
        persisted_count, flagged_count = self._persist_messages(messages)
        checkpoint = build_checkpoint(messages, persisted_count=persisted_count, flagged_count=flagged_count)
        storage.add_summary(
            session_id=self.session_id,
            summary=checkpoint,
            source_start=0,
            source_end=max(0, len(messages) - 1),
        )
        self.compression_count += 1

        system_messages = [m for m in messages if m.get("role") == "system"][:1]
        non_system = [m for m in messages if m.get("role") != "system"]
        head = non_system[: self.protect_first_n]
        tail = non_system[-self.protect_last_n :] if self.protect_last_n else []
        compacted = [*system_messages, *head, {"role": "system", "content": checkpoint}, *tail]
        return _dedupe_messages_preserve_order(compacted)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "lcm_grep",
                "description": "Search recoverable profile-local LCM transcript context for the active session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search terms."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                        "include_flagged": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include prompt-injection-flagged turns for audit/debugging.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "lcm_describe",
                "description": "Describe LCM summaries or non-flagged matching context for the active session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional search terms."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    },
                },
            },
        ]

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs) -> str:
        try:
            if kwargs.get("messages"):
                self._persist_messages(kwargs["messages"])
            if name == "lcm_grep":
                return json.dumps(self._grep(args), ensure_ascii=False)
            if name == "lcm_describe":
                return json.dumps(self._describe(args), ensure_ascii=False)
            return json.dumps({"success": False, "error": f"Unknown LCM tool: {name}"})
        except Exception as exc:  # Keep tool calls model-safe: JSON error, not traceback.
            return json.dumps({"success": False, "error": str(exc)})

    def _ensure_storage(self) -> LCMStorage:
        if self._storage is None:
            self._storage = LCMStorage(self.db_path)
            self._storage.upsert_conversation(self.session_id, self.platform)
        return self._storage

    def _persist_messages(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        storage = self._ensure_storage()
        storage.upsert_conversation(self.session_id, self.platform)
        persisted = 0
        flagged = 0
        for index, message in enumerate(messages):
            text = _message_text(message)
            scan = sanitize_for_storage(text)
            metadata = _message_metadata(message)
            item_id = storage.append_context_item(
                session_id=self.session_id,
                message_index=index,
                role=str(message.get("role") or "unknown"),
                content=scan.content,
                metadata=metadata,
                secret_redacted=scan.secret_redacted,
                injection_flag=scan.injection_flag,
                injection_reason=scan.injection_reason,
            )
            if item_id is not None:
                persisted += 1
            if scan.injection_flag:
                flagged += 1
        return persisted, flagged

    def _grep(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        rows = self._ensure_storage().search(
            session_id=self.session_id,
            query=query,
            limit=int(args.get("limit") or 10),
            include_flagged=bool(args.get("include_flagged", False)),
        )
        return {
            "success": True,
            "query": query,
            "matches": [_row_for_tool(row) for row in rows],
        }

    def _describe(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        limit = int(args.get("limit") or 10)
        storage = self._ensure_storage()
        if query:
            rows = storage.search(session_id=self.session_id, query=query, limit=limit, include_flagged=False)
            excluded = storage.count_flagged_matches(session_id=self.session_id, query=query)
            return {
                "success": True,
                "query": query,
                "items": [_row_for_tool(row) for row in rows],
                "excluded_flagged": excluded,
            }
        summaries = storage.summaries(session_id=self.session_id, limit=limit)
        return {"success": True, "summaries": summaries, "excluded_flagged": 0}


def register(ctx) -> None:
    ctx.register_context_engine(LCMContextEngine())


def _message_text(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    content = _content_text(message.get("content"))
    tool_bits: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict):
            fn = tool_call.get("function") or {}
            tool_bits.append(f"tool_call {fn.get('name')}: {fn.get('arguments')}")
    prefix = f"[{role}]"
    if tool_bits:
        return f"{prefix} {content}\n" + "\n".join(tool_bits)
    return f"{prefix} {content}"


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if message.get("tool_call_id"):
        metadata["tool_call_id"] = message.get("tool_call_id")
    if message.get("tool_calls"):
        metadata["tool_calls"] = message.get("tool_calls")
    return metadata


def _row_for_tool(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_index": row["message_index"],
        "role": row["role"],
        "content": row["content"],
        "secret_redacted": bool(row["secret_redacted"]),
        "injection_flag": bool(row["injection_flag"]),
        "injection_reason": row.get("injection_reason"),
    }


def _dedupe_messages_preserve_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    result: list[dict[str, Any]] = []
    for msg in messages:
        ident = id(msg)
        if ident in seen:
            continue
        seen.add(ident)
        result.append(msg)
    return result
