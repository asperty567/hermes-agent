"""SQLite storage for the built-in LCM context engine."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class LCMStorage:
    """Small profile-local SQLite store for recoverable session context."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("pragma journal_mode=wal")
            self.conn.execute("pragma foreign_keys=on")
            self.create_tables()

    def create_tables(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
            create table if not exists conversations (
                session_id text primary key,
                platform text,
                created_at real not null,
                updated_at real not null
            );

            create table if not exists context_items (
                id integer primary key autoincrement,
                session_id text not null references conversations(session_id) on delete cascade,
                message_index integer not null,
                role text not null,
                content text not null,
                metadata_json text not null default '{}',
                secret_redacted integer not null default 0,
                injection_flag integer not null default 0,
                injection_reason text,
                created_at real not null,
                unique(session_id, message_index)
            );

            create virtual table if not exists lcm_fts using fts5(
                content,
                session_id unindexed,
                item_id unindexed
            );

            create table if not exists summary_messages (
                id integer primary key autoincrement,
                session_id text not null references conversations(session_id) on delete cascade,
                summary text not null,
                source_start integer not null,
                source_end integer not null,
                created_at real not null
            );

            create table if not exists summary_edges (
                parent_id integer not null references summary_messages(id) on delete cascade,
                child_id integer not null references summary_messages(id) on delete cascade,
                edge_type text not null default 'next',
                primary key(parent_id, child_id, edge_type)
            );
                """
            )
            self.conn.commit()

    def upsert_conversation(self, session_id: str, platform: str = "") -> None:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
            insert into conversations(session_id, platform, created_at, updated_at)
            values (?, ?, ?, ?)
            on conflict(session_id) do update set
              platform=excluded.platform,
              updated_at=excluded.updated_at
                """,
                (session_id, platform, now, now),
            )
            self.conn.commit()

    def append_context_item(
        self,
        *,
        session_id: str,
        message_index: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        secret_redacted: bool = False,
        injection_flag: bool = False,
        injection_reason: str | None = None,
    ) -> int | None:
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                """
            insert or ignore into context_items(
                session_id, message_index, role, content, metadata_json,
                secret_redacted, injection_flag, injection_reason, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message_index,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    1 if secret_redacted else 0,
                    1 if injection_flag else 0,
                    injection_reason,
                    now,
                ),
            )
            if cur.rowcount == 0:
                self.conn.commit()
                row = self.conn.execute(
                    "select id from context_items where session_id=? and message_index=?",
                    (session_id, message_index),
                ).fetchone()
                return int(row["id"]) if row else None

            if cur.lastrowid is None:
                self.conn.commit()
                return None
            item_id = int(cur.lastrowid)
            self.conn.execute(
                "insert into lcm_fts(rowid, content, session_id, item_id) values (?, ?, ?, ?)",
                (item_id, content, session_id, str(item_id)),
            )
            self.conn.commit()
            return item_id

    def add_summary(self, *, session_id: str, summary: str, source_start: int, source_end: int) -> int:
        now = time.time()
        with self._lock:
            previous = self.conn.execute(
                "select id from summary_messages where session_id=? order by id desc limit 1",
                (session_id,),
            ).fetchone()
            cur = self.conn.execute(
                """
            insert into summary_messages(session_id, summary, source_start, source_end, created_at)
            values (?, ?, ?, ?, ?)
                """,
                (session_id, summary, source_start, source_end, now),
            )
            if cur.lastrowid is None:
                self.conn.commit()
                raise RuntimeError("failed to insert LCM summary")
            summary_id = int(cur.lastrowid)
            if previous:
                self.conn.execute(
                    "insert or ignore into summary_edges(parent_id, child_id, edge_type) values (?, ?, 'next')",
                    (int(previous["id"]), summary_id),
                )
            self.conn.commit()
            return summary_id

    def search(self, *, session_id: str, query: str, limit: int = 10, include_flagged: bool = False) -> list[dict[str, Any]]:
        terms = _fts_terms(query)
        if not terms:
            return []
        match_expr = " AND ".join(terms)
        sql = (
            "select ci.* from lcm_fts f "
            "join context_items ci on ci.id = cast(f.item_id as integer) "
            "where f.session_id=? and lcm_fts match ?"
        )
        params: list[Any] = [session_id, match_expr]
        if not include_flagged:
            sql += " and ci.injection_flag=0"
        sql += " order by ci.message_index asc limit ?"
        params.append(max(1, min(int(limit or 10), 50)))
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def count_flagged_matches(self, *, session_id: str, query: str) -> int:
        terms = _fts_terms(query)
        if not terms:
            return 0
        with self._lock:
            row = self.conn.execute(
                """
            select count(*) as n from lcm_fts f
            join context_items ci on ci.id = cast(f.item_id as integer)
            where f.session_id=? and lcm_fts match ? and ci.injection_flag=1
                """,
                (session_id, " AND ".join(terms)),
            ).fetchone()
            return int(row["n"] if row else 0)

    def summaries(self, *, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "select * from summary_messages where session_id=? order by id desc limit ?",
                (session_id, max(1, min(int(limit or 10), 50))),
            ).fetchall()
            return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def _fts_terms(query: str) -> list[str]:
    import re

    terms = re.findall(r"[\w]{2,}", query or "", flags=re.UNICODE)
    # Keep the query simple and safe for FTS MATCH syntax.  Double quotes inside
    # tokens are impossible after the regex, but quote anyway to avoid operator
    # interpretation for words like NOT/OR.
    return [f'"{term}"' for term in terms[:8]]
