#!/usr/bin/env python3
"""Migrate embeddings stored as text (Python literal or JSON) to numpy float32 BLOBs.

Usage:
  python3 scripts/migrate_embeddings_to_blob.py --db data/cache.db

The script will (by default) make a backup of the DB as data/cache.db.bak before modifying rows.
It processes rows in batches to avoid high memory use and will skip rows that already contain BLOBs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sqlite3
import sys
import time
from typing import Optional

import numpy as np


def migrate(db_path: str, backup: bool = True, batch_size: int = 1000, commit: bool = True) -> None:
    if not os.path.exists(db_path):
        raise SystemExit(f'Database file not found: {db_path}')

    if backup:
        bak = f"{db_path}.bak"
        print(f'Creating backup: {bak}')
        shutil.copy2(db_path, bak)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    try:
        cur.execute('SELECT COUNT(*) FROM embeddings')
        total = cur.fetchone()[0]
    except Exception as e:
        cur.close()
        con.close()
        raise SystemExit(f'Failed to read embeddings table: {e}')

    print(f'Found {total} rows in embeddings table. Processing in batches of {batch_size}.')

    offset = 0
    processed_rows = 0
    updated_rows = 0
    start_time = time.time()

    while True:
        cur.execute('SELECT hash, embedding, model FROM embeddings LIMIT ? OFFSET ?', (batch_size, offset))
        rows = cur.fetchall()
        if not rows:
            break

        updates = []
        for h, embedding_value, model in rows:
            # Skip if already binary/blob
            if isinstance(embedding_value, (bytes, bytearray)):
                continue

            if embedding_value is None:
                continue

            s = embedding_value
            # Convert memoryview/bytes to string if necessary
            if isinstance(s, memoryview):
                try:
                    s = s.tobytes().decode('utf-8')
                except Exception:
                    continue
            if isinstance(s, bytes):
                try:
                    s = s.decode('utf-8')
                except Exception:
                    continue

            # Try JSON first, then Python literal
            parsed = None
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    print(f'Warning: could not parse embedding for hash {h!r}; skipping')
                    continue

            try:
                arr = np.asarray(parsed, dtype=np.float32)
            except Exception as e:
                print(f'Warning: failed to convert parsed embedding to float32 for hash {h!r}: {e}; skipping')
                continue

            blob = arr.tobytes()
            updates.append((blob, h, model))

        if updates:
            try:
                cur.executemany('UPDATE embeddings SET embedding = ? WHERE hash = ? AND model = ?', updates)
                if commit:
                    con.commit()
                updated_rows += len(updates)
                print(f'Applied {len(updates)} updates for batch starting at offset {offset}')
            except Exception as e:
                print(f'Error applying updates at offset {offset}: {e}')

        processed_rows += len(rows)
        offset += batch_size
        elapsed = time.time() - start_time
        print(f'Progress: {processed_rows}/{total} rows processed ({updated_rows} updated) — {elapsed:.1f}s elapsed')

    cur.close()
    con.close()

    print(f'Done. Processed {processed_rows} rows; updated {updated_rows} rows.')


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Migrate embeddings to numpy float32 BLOBs')
    parser.add_argument('--db', '-d', default='data/cache.db', help='Path to SQLite database')
    parser.add_argument('--no-backup', dest='backup', action='store_false', help='Do not create a .bak backup')
    parser.add_argument('--batch-size', type=int, default=1000, help='Number of rows to process per batch')
    parser.add_argument('--no-commit', dest='commit', action='store_false', help="Don't commit changes (dry-run)")
    args = parser.parse_args(argv)

    try:
        migrate(db_path=args.db, backup=args.backup, batch_size=args.batch_size, commit=args.commit)
    except Exception as e:
        print(f'Fatal error: {e}')
        sys.exit(2)


if __name__ == '__main__':
    main()
