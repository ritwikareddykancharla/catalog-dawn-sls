"""SQLite WAL Audit Ledger & Active Self-Learning Rule Store for CatalogDawn."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

GENESIS_HASH = "GENESIS_CATALOGDAWN_SQLITE_0000"


class CatalogAuditStore:
    """Append-only SHA-256 hash-chained SQLite WAL event log and promoted rule registry."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        base_dir = Path(os.environ.get("CATALOG_DAWN_RUNTIME_DIR", "/tmp/catalog-dawn-runtime"))
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (base_dir / "catalog_dawn.db")
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_audit_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    ts_utc TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS promoted_rules (
                    rule_id TEXT PRIMARY KEY,
                    promoted_at_utc TEXT NOT NULL,
                    category TEXT NOT NULL,
                    target_attribute TEXT NOT NULL,
                    holdout_precision REAL NOT NULL,
                    holdout_recall REAL NOT NULL,
                    false_positive_rate REAL NOT NULL,
                    rule_json TEXT NOT NULL
                );
                """
            )
            row = conn.execute("SELECT COUNT(*) AS cnt FROM catalog_audit_events").fetchone()
            if row and int(row["cnt"]) == 0:
                self._insert_event_unlocked(
                    conn,
                    stage="SYSTEM_BOOT",
                    entity_key="catalog-dawn-sls",
                    verdict="INITIALIZED",
                    duration_ms=0.15,
                    payload={"info": "CatalogDawn SQLite WAL store initialized"},
                )

    def _insert_event_unlocked(
        self,
        conn: sqlite3.Connection,
        stage: str,
        entity_key: str,
        verdict: str,
        duration_ms: float,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        last = conn.execute(
            "SELECT seq, sha256_hash FROM catalog_audit_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        next_seq = (int(last["seq"]) + 1) if last else 1
        prev_hash = str(last["sha256_hash"]) if last else GENESIS_HASH
        event_id = f"CDW-{next_seq:05d}"
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        canonical = json.dumps(
            {
                "event_id": event_id,
                "stage": stage,
                "entity_key": entity_key,
                "verdict": verdict,
                "payload_json": payload_json,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT INTO catalog_audit_events (
                event_id, ts_utc, stage, entity_key, verdict,
                duration_ms, payload_json, prev_hash, sha256_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, ts_utc, stage, entity_key, verdict, round(duration_ms, 3), payload_json, prev_hash, digest),
        )
        return {
            "event_id": event_id,
            "ts_utc": ts_utc,
            "stage": stage,
            "entity_key": entity_key,
            "verdict": verdict,
            "duration_ms": round(duration_ms, 3),
            "prev_hash": prev_hash,
            "sha256_hash": digest,
        }

    def record_event(
        self,
        stage: str,
        entity_key: str,
        verdict: str,
        duration_ms: float,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            return self._insert_event_unlocked(conn, stage, entity_key, verdict, duration_ms, payload)

    def save_promoted_rule(self, rule: Dict[str, Any]) -> None:
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO promoted_rules (
                    rule_id, promoted_at_utc, category, target_attribute,
                    holdout_precision, holdout_recall, false_positive_rate, rule_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(rule["rule_id"]),
                    ts_utc,
                    str(rule.get("category", "general")),
                    str(rule.get("target_attribute", "unknown")),
                    float(rule.get("holdout_precision", 1.0)),
                    float(rule.get("holdout_recall", 0.0)),
                    float(rule.get("false_positive_rate", 0.0)),
                    json.dumps(rule, sort_keys=True, default=str),
                ),
            )

    def list_promoted_rules(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT rule_json FROM promoted_rules ORDER BY promoted_at_utc DESC").fetchall()
            return [json.loads(str(r["rule_json"])) for r in rows]

    def list_recent_events(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, ts_utc, stage, entity_key, verdict, duration_ms, prev_hash, sha256_hash
                FROM catalog_audit_events ORDER BY seq DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def verify_chain_integrity(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, stage, entity_key, verdict, payload_json, prev_hash, sha256_hash
                FROM catalog_audit_events ORDER BY seq ASC
                """
            ).fetchall()

        expected_prev = GENESIS_HASH
        for r in rows:
            if str(r["prev_hash"]) != expected_prev:
                return {
                    "valid": False,
                    "checked_rows": len(rows),
                    "broken_at": str(r["event_id"]),
                    "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }
            canonical = json.dumps(
                {
                    "event_id": str(r["event_id"]),
                    "stage": str(r["stage"]),
                    "entity_key": str(r["entity_key"]),
                    "verdict": str(r["verdict"]),
                    "payload_json": str(r["payload_json"]),
                    "prev_hash": str(r["prev_hash"]),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if recomputed != str(r["sha256_hash"]):
                return {
                    "valid": False,
                    "checked_rows": len(rows),
                    "broken_at": str(r["event_id"]),
                    "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }
            expected_prev = recomputed

        return {
            "valid": True,
            "checked_rows": len(rows),
            "head_hash": expected_prev,
            "verify_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def reset(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM catalog_audit_events;")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='catalog_audit_events';")
            conn.execute("DELETE FROM promoted_rules;")
            self._insert_event_unlocked(
                conn,
                stage="SYSTEM_RESET",
                entity_key="catalog-dawn-sls",
                verdict="RESET_CLEAN",
                duration_ms=0.12,
                payload={"info": "Transient state reset"},
            )
