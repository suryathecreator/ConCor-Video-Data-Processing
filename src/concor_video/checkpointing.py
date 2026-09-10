"""Atomic per-sample claims and result commits for preemptible workers."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def sample_claim(
    claim_path: Path,
    *,
    owner: str,
    stale_after_seconds: int = 6 * 3600,
) -> Iterator[bool]:
    """Yield whether this worker owns the sample; recover only stale claims."""

    claim_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    try:
        stat = claim_path.stat()
    except FileNotFoundError:
        pass
    else:
        same_owner = False
        try:
            same_owner = json.loads(claim_path.read_text(encoding="utf-8")).get("owner") == owner
        except (OSError, ValueError, AttributeError):
            pass
        if same_owner or now - stat.st_mtime > stale_after_seconds:
            stale = claim_path.with_suffix(claim_path.suffix + f".stale-{int(now)}")
            try:
                claim_path.replace(stale)
            except FileNotFoundError:
                pass
        else:
            yield False
            return
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        yield False
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"owner": owner, "pid": os.getpid(), "claimed_at": now}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield True
    finally:
        try:
            claim_path.unlink()
        except FileNotFoundError:
            pass
