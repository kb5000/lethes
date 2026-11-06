"""SQLite snapshot/attempt ledger. No mutable 'current branch' pointer."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .history import Message, canonical
from .state import Snapshot


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                scope TEXT NOT NULL, domain TEXT NOT NULL, head TEXT NOT NULL,
                length INTEGER NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(scope, domain, head)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, domain TEXT NOT NULL,
                head TEXT NOT NULL, correlation TEXT NOT NULL, created REAL NOT NULL,
                wire TEXT NOT NULL, usage TEXT
            );
            CREATE INDEX IF NOT EXISTS attempts_scope ON attempts(scope, domain, created);
            CREATE INDEX IF NOT EXISTS attempts_correlation ON attempts(correlation, created);
        """)

    def ancestor(self, scope: str, domain: str, heads: tuple[str, ...]) -> Snapshot | None:
        best = None
        with self._lock:
            # Bounded SQL parameter batches avoid a full scan of large branch payloads.
            for offset in range(0, len(heads), 400):
                batch = heads[offset : offset + 400]
                slots = ",".join("?" for _ in batch)
                row = self._db.execute(
                    "SELECT length, payload FROM snapshots WHERE scope=? AND domain=? "
                    f"AND head IN ({slots}) ORDER BY length DESC LIMIT 1",
                    (scope, domain, *batch),
                ).fetchone()
                if row and (best is None or row[0] > best[0]):
                    best = row
        return Snapshot.deserialize(best[1]) if best else None

    def save(self, scope: str, domain: str, snapshot: Snapshot) -> Snapshot:
        # First writer wins. Concurrent identical inputs replay the same bytes.
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?)",
                (
                    scope,
                    domain,
                    snapshot.head,
                    snapshot.history_length,
                    snapshot.serialize(),
                ),
            )
            row = self._db.execute(
                "SELECT payload FROM snapshots WHERE scope=? AND domain=? AND head=?",
                (scope, domain, snapshot.head),
            ).fetchone()
        return Snapshot.deserialize(row[0])

    def attempt(self, scope: str, domain: str, snapshot: Snapshot, correlation: str = "") -> str:
        request_id = uuid.uuid4().hex
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,NULL)",
                (
                    request_id,
                    scope,
                    domain,
                    snapshot.head,
                    correlation,
                    time.time(),
                    canonical(snapshot.messages),
                ),
            )
        return request_id

    def recent_wires(self, scope: str, domain: str, ttl: float) -> list[list[Message]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT wire FROM attempts WHERE scope=? AND domain=? AND created>=? "
                "ORDER BY created DESC LIMIT 16",
                (scope, domain, time.time() - ttl),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def record_usage(self, request_id: str, usage: dict) -> None:
        # Idempotent: an outlet seeing the same usage does not double-charge.
        with self._lock, self._db:
            self._db.execute(
                "UPDATE attempts SET usage=? WHERE id=?", (canonical(usage), request_id)
            )

    def usage(self, request_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT usage FROM attempts WHERE id=?", (request_id,)
            ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def correlated_request(self, correlation: str) -> str | None:
        if not correlation:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM attempts WHERE correlation=? ORDER BY created DESC LIMIT 1",
                (correlation,),
            ).fetchone()
        return row[0] if row else None

    def delete_scope(self, scope: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM attempts WHERE scope=?", (scope,))
            self._db.execute("DELETE FROM snapshots WHERE scope=?", (scope,))

    def close(self) -> None:
        with self._lock:
            self._db.close()
