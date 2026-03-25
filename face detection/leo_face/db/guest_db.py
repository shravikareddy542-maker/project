"""
SQLite database layer for guest profiles and face embeddings.
Tables:
  guests  – guest_id, name, designation, achievements, created_at
  embeddings – embedding_id, guest_id, embedding (BLOB), created_at
"""
import sqlite3
import os
import pickle
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_connection(db_path: str = None) -> sqlite3.Connection:
    db_path = db_path or config.DB_PATH
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS guests (
            guest_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            designation TEXT DEFAULT '',
            achievements TEXT DEFAULT '',
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_id     INTEGER NOT NULL,
            embedding    BLOB NOT NULL,
            created_at   TEXT NOT NULL,
            FOREIGN KEY (guest_id) REFERENCES guests(guest_id) ON DELETE CASCADE
        );
    """)
    conn.commit()


# ───────────────────── Guest CRUD ─────────────────────────

def add_guest(conn: sqlite3.Connection,
              name: str,
              designation: str = "",
              achievements: str = "") -> int:
    """Insert a new guest and return guest_id."""
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO guests (name, designation, achievements, created_at) VALUES (?, ?, ?, ?)",
        (name, designation, achievements, now),
    )
    conn.commit()
    return cur.lastrowid


def list_guests(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT g.guest_id, g.name, g.designation, g.achievements, g.created_at, "
        "COUNT(e.embedding_id) AS num_embeddings "
        "FROM guests g LEFT JOIN embeddings e ON g.guest_id = e.guest_id "
        "GROUP BY g.guest_id ORDER BY g.name"
    ).fetchall()
    return [dict(r) for r in rows]


def get_guest(conn: sqlite3.Connection, guest_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM guests WHERE guest_id = ?", (guest_id,)).fetchone()
    return dict(row) if row else None


def get_guest_by_name(conn: sqlite3.Connection, name: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM guests WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))",
        (name,)
    ).fetchone()
    return dict(row) if row else None


def delete_guest(conn: sqlite3.Connection, guest_id: int):
    conn.execute("DELETE FROM embeddings WHERE guest_id = ?", (guest_id,))
    conn.execute("DELETE FROM guests WHERE guest_id = ?", (guest_id,))
    conn.commit()


# ──────────────────── Embedding CRUD ──────────────────────

def _serialize_embedding(emb: np.ndarray) -> bytes:
    return pickle.dumps(emb.astype(np.float32))


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


def add_embedding(conn: sqlite3.Connection,
                  guest_id: int,
                  embedding: np.ndarray) -> int:
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO embeddings (guest_id, embedding, created_at) VALUES (?, ?, ?)",
        (guest_id, _serialize_embedding(embedding), now),
    )
    conn.commit()
    return cur.lastrowid


def get_all_embeddings(conn: sqlite3.Connection) -> list[dict]:
    """Return list of {embedding_id, guest_id, embedding(np), name}."""
    rows = conn.execute(
        "SELECT e.embedding_id, e.guest_id, e.embedding, g.name "
        "FROM embeddings e JOIN guests g ON e.guest_id = g.guest_id"
    ).fetchall()
    result = []
    for r in rows:
        result.append({
            "embedding_id": r["embedding_id"],
            "guest_id": r["guest_id"],
            "embedding": _deserialize_embedding(r["embedding"]),
            "name": r["name"],
        })
    return result


def count_embeddings_for_guest(conn: sqlite3.Connection, guest_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM embeddings WHERE guest_id = ?", (guest_id,)
    ).fetchone()
    return row["cnt"]