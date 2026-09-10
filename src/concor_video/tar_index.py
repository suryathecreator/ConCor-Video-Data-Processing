"""Indexed, process-safe random access to uncompressed tar archives."""

from __future__ import annotations

import os
import sqlite3
import tarfile
from pathlib import Path
from typing import Iterable


def _source_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _valid_index(source: Path, index: Path) -> bool:
    if not index.is_file() or index.stat().st_size == 0:
        return False
    try:
        with sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True) as connection:
            values = dict(connection.execute("SELECT key, value FROM metadata"))
            size, mtime_ns = _source_signature(source)
            return (
                int(values.get("source_size", -1)) == size
                and int(values.get("source_mtime_ns", -1)) == mtime_ns
                and int(values.get("member_count", 0)) > 0
            )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return False


def index_tar(source: Path, destination: Path) -> int:
    """Index member byte ranges without expanding archive contents."""

    source = source.resolve()
    destination = destination.resolve()
    if _valid_index(source, destination):
        with sqlite3.connect(
            f"file:{destination}?mode=ro&immutable=1", uri=True
        ) as connection:
            return int(
                connection.execute(
                    "SELECT value FROM metadata WHERE key='member_count'"
                ).fetchone()[0]
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE members ("
            "name TEXT PRIMARY KEY, offset_data INTEGER NOT NULL, size INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        batch: list[tuple[str, int, int]] = []
        count = 0
        # ReVOS.tar is deliberately uncompressed. TarFile advances between
        # headers with seeks, so indexing reads metadata rather than 47 GB of payload.
        with tarfile.open(source, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                batch.append((member.name, int(member.offset_data), int(member.size)))
                count += 1
                if len(batch) == 4096:
                    connection.executemany("INSERT INTO members VALUES (?, ?, ?)", batch)
                    batch.clear()
        if batch:
            connection.executemany("INSERT INTO members VALUES (?, ?, ?)", batch)
        size, mtime_ns = _source_signature(source)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("source_size", str(size)),
                ("source_mtime_ns", str(mtime_ns)),
                ("member_count", str(count)),
            ],
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()
    temporary.replace(destination)
    return count


class IndexedTarReader:
    """Read named members with SQLite lookups plus ``pread`` byte ranges."""

    def __init__(self, source: Path, index: Path | None = None) -> None:
        self.source = source.resolve()
        self.index = (index or self.source.with_suffix(self.source.suffix + ".sqlite")).resolve()
        if not _valid_index(self.source, self.index):
            raise RuntimeError(
                f"missing or stale tar index {self.index}; run index-revos-tar first"
            )
        self._connection = sqlite3.connect(
            f"file:{self.index}?mode=ro&immutable=1", uri=True, check_same_thread=False
        )
        self._fd = os.open(self.source, os.O_RDONLY)

    def find(self, candidates: Iterable[str]) -> tuple[str, int, int]:
        for name in candidates:
            row = self._connection.execute(
                "SELECT offset_data, size FROM members WHERE name = ?", (name,)
            ).fetchone()
            if row is not None:
                return name, int(row[0]), int(row[1])
        raise FileNotFoundError(
            f"none of {list(candidates)} exists in archive {self.source}"
        )

    def read(self, candidates: Iterable[str]) -> tuple[str, bytes]:
        name, offset, size = self.find(candidates)
        chunks: list[bytes] = []
        remaining = size
        cursor = offset
        while remaining:
            payload = os.pread(self._fd, remaining, cursor)
            if not payload:
                raise EOFError(f"short read for {name} in {self.source}")
            chunks.append(payload)
            cursor += len(payload)
            remaining -= len(payload)
        return name, b"".join(chunks)

    def extract(self, member: str, destination: Path) -> None:
        _, offset, size = self.find((member,))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
        try:
            with temporary.open("wb") as output:
                remaining = size
                cursor = offset
                while remaining:
                    payload = os.pread(self._fd, min(8 * 1024 * 1024, remaining), cursor)
                    if not payload:
                        raise EOFError(f"short read for {member} in {self.source}")
                    output.write(payload)
                    cursor += len(payload)
                    remaining -= len(payload)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        self._connection.close()
        os.close(self._fd)

    def __enter__(self) -> "IndexedTarReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
