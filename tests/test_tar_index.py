import io
import tarfile
from pathlib import Path

import pytest

from concor_video.tar_index import IndexedTarReader, index_tar


def _archive(path: Path) -> None:
    with tarfile.open(path, mode="w:") as archive:
        for name, payload in (
            ("ReVOS/JPEGImages/a/00000.jpg", b"jpeg-one"),
            ("ReVOS/meta_expressions_train_.json", b"{}"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_indexed_tar_random_read_and_extract(tmp_path: Path) -> None:
    source = tmp_path / "ReVOS.tar"
    index = tmp_path / "ReVOS.tar.sqlite"
    _archive(source)
    assert index_tar(source, index) == 2
    assert index_tar(source, index) == 2

    with IndexedTarReader(source, index) as archive:
        name, payload = archive.read(
            ("missing", "ReVOS/JPEGImages/a/00000.jpg")
        )
        assert name.endswith("00000.jpg")
        assert payload == b"jpeg-one"
        destination = tmp_path / "extract" / "metadata.json"
        archive.extract("ReVOS/meta_expressions_train_.json", destination)
    assert destination.read_bytes() == b"{}"


def test_index_rejects_stale_source(tmp_path: Path) -> None:
    source = tmp_path / "ReVOS.tar"
    index = tmp_path / "ReVOS.tar.sqlite"
    _archive(source)
    index_tar(source, index)
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="missing or stale tar index"):
        IndexedTarReader(source, index)
