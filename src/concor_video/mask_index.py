"""One-time SQLite indexing for the large ReVOS mask dictionary."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def index_revos_masks(source: Path, destination: Path) -> int:
    if destination.is_file() and destination.stat().st_size > 0:
        with sqlite3.connect(
            f"file:{destination}?mode=ro&immutable=1", uri=True
        ) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
    values = json.loads(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE masks (annotation_id TEXT PRIMARY KEY, sequence_json TEXT NOT NULL)"
        )
        batch = []
        for annotation_id, sequence in values.items():
            batch.append(
                (
                    str(annotation_id),
                    json.dumps(sequence, separators=(",", ":"), ensure_ascii=False),
                )
            )
            if len(batch) == 2048:
                connection.executemany("INSERT INTO masks VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO masks VALUES (?, ?)", batch)
        connection.commit()
        connection.execute("PRAGMA optimize")
        count = int(connection.execute("SELECT COUNT(*) FROM masks").fetchone()[0])
    finally:
        connection.close()
    temporary.replace(destination)
    return count
