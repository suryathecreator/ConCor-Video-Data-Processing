import json

from concor_video.worker import _is_cuda_oom, _pending


def test_exhausted_cuda_oom_gets_one_memory_safe_retry(tmp_path):
    records = tmp_path / "records"
    errors = tmp_path / "errors"
    records.mkdir()
    errors.mkdir()
    unit = {"sample_id": "heavy"}
    path = errors / "heavy.json"
    path.write_text(json.dumps({"attempt": 3, "error_type": "OutOfMemoryError", "message": "CUDA out of memory"}))
    assert _is_cuda_oom(path)
    assert _pending(unit, records_dir=records, errors_dir=errors, retry_errors=True, max_error_attempts=3)
    path.write_text(json.dumps({"attempt": 4, "error_type": "OutOfMemoryError", "message": "CUDA out of memory"}))
    assert not _pending(unit, records_dir=records, errors_dir=errors, retry_errors=True, max_error_attempts=3)


def test_exhausted_non_oom_stays_terminal(tmp_path):
    records = tmp_path / "records"
    errors = tmp_path / "errors"
    records.mkdir()
    errors.mkdir()
    unit = {"sample_id": "bad"}
    (errors / "bad.json").write_text(json.dumps({"attempt": 3, "error_type": "ValueError", "message": "bad input"}))
    assert not _pending(unit, records_dir=records, errors_dir=errors, retry_errors=True, max_error_attempts=3)
