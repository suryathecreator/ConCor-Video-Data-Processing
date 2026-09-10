"""COCO run-length mask decoding without a compiled dependency."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compressed_counts(value: str | bytes) -> list[int]:
    """Decode COCO's compressed ASCII RLE counts."""

    if isinstance(value, bytes):
        value = value.decode("ascii")
    counts: list[int] = []
    position = 0
    while position < len(value):
        number = 0
        shift = 0
        more = True
        while more:
            code = ord(value[position]) - 48
            position += 1
            number |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                number |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            number += counts[-2]
        counts.append(number)
    return counts


def decode_rle(rle: dict | None) -> np.ndarray | None:
    """Decode one COCO RLE dictionary into a boolean ``H x W`` mask."""

    if not rle:
        return None
    height, width = map(int, rle["size"])
    raw = rle["counts"]
    runs: Sequence[int]
    if isinstance(raw, (str, bytes)):
        runs = compressed_counts(raw)
    else:
        runs = [int(value) for value in raw]

    flat = np.zeros(height * width, dtype=np.bool_)
    cursor = 0
    foreground = False
    for run in runs:
        if run < 0:
            raise ValueError(f"negative COCO RLE run: {run}")
        stop = min(cursor + run, flat.size)
        if foreground:
            flat[cursor:stop] = True
        cursor = stop
        foreground = not foreground
    if cursor != flat.size:
        raise ValueError(f"COCO RLE covers {cursor} pixels, expected {flat.size}")
    return flat.reshape((height, width), order="F")


def encode_rle(mask: np.ndarray | None) -> dict | None:
    """Encode a boolean mask as portable uncompressed COCO RLE.

    Uncompressed counts are somewhat larger than pycocotools' ASCII form but
    remain deterministic, JSON-native, and dependency-free. Parquet's Zstandard
    compression handles repeated structure in the published tables.
    """

    if mask is None:
        return None
    array = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError(f"expected HxW mask, got shape {array.shape}")
    flat = array.reshape(-1, order="F")
    counts: list[int] = []
    current = False
    run = 0
    for value in flat:
        foreground = bool(value)
        if foreground == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = foreground
    counts.append(run)
    return {"size": [int(array.shape[0]), int(array.shape[1])], "counts": counts}
