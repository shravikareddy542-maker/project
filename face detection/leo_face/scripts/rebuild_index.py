"""
rebuild_index.py – CLI tool to rebuild the FAISS index from SQLite embeddings.
Usage: python scripts/rebuild_index.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.guest_db import get_connection, init_db, get_all_embeddings
from recognition.matcher_faiss import FAISSMatcher
import config


def main():
    print("[INFO] Connecting to database...")
    conn = get_connection()
    init_db(conn)
    records = get_all_embeddings(conn)
    conn.close()

    print(f"[INFO] Found {len(records)} embeddings across guests.")
    matcher = FAISSMatcher()
    matcher.build_index(records)
    matcher.save()

    print(f"[INFO] FAISS index built and saved ({matcher.total} vectors).")
    print(f"       Index: {config.FAISS_INDEX_PATH}")
    print(f"       Map:   {config.FAISS_MAP_PATH}")


if __name__ == "__main__":
    main()
