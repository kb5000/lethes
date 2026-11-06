"""SQLite snapshot/attempt ledger. No mutable 'current branch' pointer."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .history import Message, canonical, digest
from .state import Snapshot
from .tokens import Measurement


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
            CREATE TABLE IF NOT EXISTS attempt_models (
                id TEXT PRIMARY KEY, models TEXT NOT NULL,
                measurement_domain TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS attempt_observations (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS token_measurements (
                scope TEXT NOT NULL, domain TEXT NOT NULL, wire_hash TEXT NOT NULL,
                wire TEXT NOT NULL, tokens INTEGER NOT NULL, request_id TEXT NOT NULL,
                created REAL NOT NULL, PRIMARY KEY(scope, domain, wire_hash)
            );
            CREATE TABLE IF NOT EXISTS layout_versions (
                scope TEXT NOT NULL, domain TEXT NOT NULL, head TEXT NOT NULL,
                epoch TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(scope, domain, head, epoch)
            );
            CREATE TABLE IF NOT EXISTS measurement_barriers (
                scope TEXT NOT NULL, domain TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY(scope,domain)
            );
        """)
        columns = {r[1] for r in self._db.execute("PRAGMA table_info(attempt_models)")}
        if "measurement_domain" not in columns:
            self._db.execute(
                "ALTER TABLE attempt_models ADD COLUMN measurement_domain TEXT NOT NULL DEFAULT ''"
            )

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

    def replace_overbudget(
        self, scope: str, domain: str, snapshot: Snapshot, expected: Snapshot
    ) -> Snapshot:
        """CAS a preferred layout after real usage invalidates its budget.

        Prior layouts remain immutable in the archive and in the attempt ledger.
        Normal saves still use first-writer-wins and never recompile a valid replay.
        """
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO layout_versions VALUES (?,?,?,?,?)",
                (scope, domain, expected.head, expected.epoch, expected.serialize()),
            )
            updated = self._db.execute(
                "UPDATE snapshots SET payload=?, length=? WHERE scope=? AND domain=? "
                "AND head=? AND payload=?",
                (
                    snapshot.serialize(),
                    snapshot.history_length,
                    scope,
                    domain,
                    snapshot.head,
                    expected.serialize(),
                ),
            )
            if updated.rowcount:
                self._db.execute(
                    "INSERT OR IGNORE INTO layout_versions VALUES (?,?,?,?,?)",
                    (scope, domain, snapshot.head, snapshot.epoch, snapshot.serialize()),
                )
            row = self._db.execute(
                "SELECT payload FROM snapshots WHERE scope=? AND domain=? AND head=?",
                (scope, domain, snapshot.head),
            ).fetchone()
        return Snapshot.deserialize(row[0])

    def attempt(
        self,
        scope: str,
        domain: str,
        snapshot: Snapshot,
        correlation: str = "",
        models: tuple[str, ...] = (),
        measurement_domain: str = "",
    ) -> str:
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
            self._db.execute(
                "INSERT INTO attempt_models VALUES (?,?,?)",
                (request_id, canonical(models), measurement_domain),
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

    @staticmethod
    def authoritative_usage(usage: dict) -> bool:
        tokens = usage.get("input_tokens")
        raw = usage.get("raw", {})
        source = raw.get("litellm_count_source", raw.get("source"))
        raw_counts = [
            raw[k]
            for k in (
                "prompt_tokens",
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
            if k in raw
        ]
        return (
            isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and tokens > 0
            and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in raw_counts)
            and usage.get("measurement_eligible", True)
            and not usage.get("aggregate", False)
            and not raw.get("estimated", False)
            and (
                not isinstance(source, str)
                or source not in {"local_estimate", "estimated", "estimate", "token_count_model"}
            )
        )

    def _index_measurement(self, row: tuple, usage: dict) -> None:
        request_id, scope, domain, created, wire = row
        binding = self._db.execute(
            "SELECT measurement_domain FROM attempt_models WHERE id=?", (request_id,)
        ).fetchone()
        if binding and binding[0]:
            domain = binding[0]
        if usage.get("measurement_eligible") is False:
            self._db.execute(
                "INSERT INTO measurement_barriers VALUES (?,?,?) "
                "ON CONFLICT(scope,domain) DO UPDATE SET created=MAX(created,excluded.created)",
                (scope, domain, created),
            )
            self._db.execute(
                "DELETE FROM token_measurements WHERE scope=? AND domain=? AND created<=?",
                (scope, domain, created),
            )
            return
        barrier = self._db.execute(
            "SELECT created FROM measurement_barriers WHERE scope=? AND domain=?", (scope, domain)
        ).fetchone()
        if barrier and created <= barrier[0]:
            return
        if self.authoritative_usage(usage):
            self._db.execute(
                "INSERT INTO token_measurements VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(scope,domain,wire_hash) DO UPDATE SET wire=excluded.wire, "
                "tokens=excluded.tokens, request_id=excluded.request_id, created=excluded.created "
                "WHERE excluded.created>=token_measurements.created",
                (
                    scope,
                    domain,
                    digest(json.loads(wire)),
                    wire,
                    usage["input_tokens"],
                    request_id,
                    created,
                ),
            )
        else:
            self._db.execute("DELETE FROM token_measurements WHERE request_id=?", (request_id,))

    def record_usage(self, request_id: str, usage: dict, reported_model: str | None = None) -> None:
        # Idempotent: an outlet seeing the same usage does not double-charge.
        with self._lock, self._db:
            models = self._db.execute(
                "SELECT models FROM attempt_models WHERE id=?", (request_id,)
            ).fetchone()
            if reported_model is not None:
                usage = {
                    **usage,
                    "reported_model": reported_model,
                    "measurement_eligible": bool(
                        models and reported_model in json.loads(models[0])
                    ),
                }
            self._db.execute(
                "UPDATE attempts SET usage=? WHERE id=?", (canonical(usage), request_id)
            )
            row = self._db.execute(
                "SELECT id,scope,domain,created,wire FROM attempts WHERE id=?", (request_id,)
            ).fetchone()
            if row:
                self._index_measurement(row, usage)

    def measurements(self, scope: str, domain: str, legacy_domain: str = "") -> list[Measurement]:
        with self._lock, self._db:
            # Backfill reported usage saved by the earlier alpha under the same
            # strict model/config/counter domain; no model-wide count is copied blindly.
            legacy = self._db.execute(
                "SELECT id,scope,domain,created,wire,usage FROM attempts "
                "WHERE scope=? AND domain=? AND usage IS NOT NULL ORDER BY created DESC LIMIT 64",
                (scope, legacy_domain or domain),
            ).fetchall()
            for row in legacy:
                self._index_measurement(row[:5], json.loads(row[5]))
            rows = self._db.execute(
                "SELECT wire,tokens,request_id FROM token_measurements "
                "WHERE scope=? AND domain IN (?,?) ORDER BY created DESC LIMIT 16",
                (scope, domain, legacy_domain or domain),
            ).fetchall()
        return [Measurement(*row) for row in rows]

    def measurement(self, scope: str, domain: str, wire: str) -> Measurement | None:
        with self._lock:
            row = self._db.execute(
                "SELECT wire,tokens,request_id FROM token_measurements "
                "WHERE scope=? AND domain=? AND wire_hash=?",
                (scope, domain, digest(json.loads(wire))),
            ).fetchone()
        return Measurement(*row) if row else None

    def usage(self, request_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT usage FROM attempts WHERE id=?", (request_id,)
            ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def record_observation(self, request_id: str, observation: dict) -> None:
        """Keep dispatch-time counts even after later usage changes calibration."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO attempt_observations SELECT id,? FROM attempts WHERE id=?",
                (canonical(observation), request_id),
            )

    def observation(self, request_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM attempt_observations WHERE id=?", (request_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def correlated_request(self, correlation: str) -> str | None:
        if not correlation:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM attempts WHERE correlation=? ORDER BY created DESC LIMIT 1",
                (correlation,),
            ).fetchone()
        return row[0] if row else None

    def turn_attempts(self, request_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT correlation FROM attempts WHERE id=?", (request_id,)
            ).fetchone()
            if not row or not row[0]:
                return 1
            return self._db.execute(
                "SELECT COUNT(*) FROM attempts WHERE correlation=?", (row[0],)
            ).fetchone()[0]

    def delete_scope(self, scope: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM attempt_observations WHERE id IN "
                "(SELECT id FROM attempts WHERE scope=?)",
                (scope,),
            )
            self._db.execute(
                "DELETE FROM attempt_models WHERE id IN (SELECT id FROM attempts WHERE scope=?)",
                (scope,),
            )
            self._db.execute("DELETE FROM token_measurements WHERE scope=?", (scope,))
            self._db.execute("DELETE FROM layout_versions WHERE scope=?", (scope,))
            self._db.execute("DELETE FROM measurement_barriers WHERE scope=?", (scope,))
            self._db.execute("DELETE FROM attempts WHERE scope=?", (scope,))
            self._db.execute("DELETE FROM snapshots WHERE scope=?", (scope,))

    def close(self) -> None:
        with self._lock:
            self._db.close()
