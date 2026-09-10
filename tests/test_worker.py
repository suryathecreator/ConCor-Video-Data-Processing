import json

from concor_video.worker import _is_cuda_oom, _memory_safe_object_limit, _pending


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
    # A pre-patch exhausted error receives one recovery attempt because it has
    # never used the microbatched memory-safe path.
    path.write_text(json.dumps({"attempt": 4, "error_type": "OutOfMemoryError", "message": "CUDA out of memory"}))
    assert _pending(unit, records_dir=records, errors_dir=errors, retry_errors=True, max_error_attempts=3)
    path.write_text(json.dumps({
        "attempt": 5,
        "error_type": "OutOfMemoryError",
        "message": "CUDA out of memory",
        "memory_safe_mode": True,
        "grounding_batch_size": 1,
        "max_num_objects": 8,
    }))
    assert not _pending(unit, records_dir=records, errors_dir=errors, retry_errors=True, max_error_attempts=3)


def test_exhausted_non_oom_stays_terminal(tmp_path):
    records = tmp_path / "records"
    errors = tmp_path / "errors"
    records.mkdir()
    errors.mkdir()
    unit = {"sample_id": "bad"}
    (errors / "bad.json").write_text(json.dumps({"attempt": 3, "error_type": "ValueError", "message": "bad input"}))
    assert not _pending(unit, records_dir=records, errors_dir=errors, retry_errors=True, max_error_attempts=3)


def test_memory_safe_object_limit_tightens_after_microbatch_oom(tmp_path):
    errors = tmp_path / "errors"
    errors.mkdir()
    units = [{"sample_id": "heavy"}]
    assert _memory_safe_object_limit(units, errors) == 16
    (errors / "heavy.json").write_text(json.dumps({
        "error_type": "OutOfMemoryError",
        "memory_safe_mode": True,
        "grounding_batch_size": 1,
        "max_num_objects": 16,
    }))
    assert _memory_safe_object_limit(units, errors) == 8
