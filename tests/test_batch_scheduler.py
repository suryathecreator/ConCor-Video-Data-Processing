"""Paid Batch scheduling: real admission limits, durable recovery, no duplicates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('batch_refine_fixture', ROOT / 'scripts/refine_hf.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class FakeAPI:
    def __init__(self):
        self.files, self.batches, self.posts = {}, {}, []
        self.on_create = None
        self.lose_response = False
        self.reject_before_accept = False

    def call(self, method, endpoint, **kwargs):
        if method == 'POST' and endpoint == '/files':
            fid = 'file-' + str(len(self.files))
            data = kwargs['files']['file'][1].read()
            self.files[fid] = data
            return SimpleNamespace(json=lambda: {'id': fid})
        if method == 'POST' and endpoint == '/batches':
            body = kwargs['json']; self.posts.append(body)
            if self.reject_before_accept:
                raise requests.Timeout('lost before acceptance')
            ids = [json.loads(line)['custom_id'] for line in self.files[body['input_file_id']].splitlines()]
            status = self.on_create(ids) if self.on_create else {'status': 'in_progress'}
            batch = {'id': 'batch-' + str(len(self.batches)), **body, **status,
                     'request_counts': {'total': len(ids), 'completed': 0}, 'ids': ids}
            self.batches[batch['id']] = batch
            if self.lose_response:
                self.lose_response = False
                raise requests.Timeout('accepted but response lost')
            return SimpleNamespace(json=lambda: batch)
        if endpoint == '/batches':
            return SimpleNamespace(json=lambda: {'data': list(self.batches.values()), 'has_more': False})
        if endpoint.startswith('/batches/'):
            return SimpleNamespace(json=lambda: self.batches[endpoint.rsplit('/', 1)[-1]])
        if endpoint.startswith('/files/out-'):
            bid = endpoint.removeprefix('/files/out-').removesuffix('/content')
            text = '\n'.join(json.dumps({'custom_id': rid, 'response': {'status_code': 200,
                  'body': {'status': 'completed', 'usage': {}}}}) for rid in self.batches[bid]['ids'])
            return SimpleNamespace(text=text)
        raise AssertionError((method, endpoint))

    def complete(self, bid):
        self.batches[bid].update(status='completed', output_file_id='out-' + bid)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'request_body', lambda m, prompt, output: {'model': r.MODEL, 'input': m['request_id']})
    monkeypatch.setattr(r, 'estimated_max_cost', lambda *a, **kw: 1.)
    monkeypatch.setattr(r, 'usage_cost', lambda *a, **kw: .1)
    monkeypatch.setattr(r, 'save_response', lambda m, body, source, output:
                        r.atomic_json(output / 'packets' / m['request_id'] / 'response.json', body))
    ledger = r.Ledger(tmp_path / 'budget.sqlite', 99.)
    api = FakeAPI()
    def packets(n):
        return [{'request_id': 'r' + str(i), 'mapping': {}} for i in range(n)]
    def collect(batches):
        return r.collect_batches(batches, packets(2000), None, tmp_path, ledger, api, False)
    yield tmp_path, ledger, api, packets, collect
    ledger.db.close()


def test_existing_batch_is_kept_and_pending_submits_without_wait(setup):
    path, ledger, api, packets, collect = setup
    ledger.reserve('r0', 1.)
    old = {'id': 'existing', 'requests': ['r0'], 'input_file_id': 'old-file'}
    r.atomic_json(path / 'batches.json', [old])
    api.batches['existing'] = {'id': 'existing', 'status': 'in_progress'}
    batches = r.submit_batches(packets(4), 'prompt', path, ledger, api, collect)
    assert batches[0] == old
    assert batches[1]['requests'] == ['r1', 'r2', 'r3']
    assert len(api.posts) == 1 and ledger.total() == 4.
    r.submit_batches(packets(4), 'prompt', path, ledger, api, collect)
    assert len(api.posts) == 1 and ledger.total() == 4.


def test_packing_has_no_1000_request_or_16m_token_cap(setup, monkeypatch):
    path, ledger, api, packets, collect = setup
    monkeypatch.setattr(r, 'estimated_max_cost', lambda *a, **kw: .001)
    monkeypatch.setattr(r, 'input_token_bound', lambda *a: pytest.fail('packing must not use the cost token bound'))
    batches = r.submit_batches(packets(1001), 'prompt', path, ledger, api, collect)
    assert len(batches) == 1 and len(batches[0]['requests']) == 1001
    assert r.BATCH_MAX_REQUESTS == 50_000 and r.BATCH_MAX_BYTES == 200_000_000


def test_only_documented_file_and_count_limits_split(setup, monkeypatch):
    path, ledger, api, packets, collect = setup
    monkeypatch.setattr(r, 'BATCH_MAX_REQUESTS', 2)
    sample = packets(1)[0]
    line_size = len((r.dumps({'custom_id': 'r0', 'method': 'POST', 'url': '/v1/responses',
                             'body': r.request_body(sample, '', path)}) + '\n').encode())
    monkeypatch.setattr(r, 'BATCH_MAX_BYTES', line_size * 2)
    batches = r.submit_batches(packets(5), 'prompt', path, ledger, api, collect)
    assert [len(b['requests']) for b in batches] == [2, 2, 1]
    assert all(len(data) <= line_size * 2 for data in api.files.values())


def test_budget_reserves_all_active_batches_and_reuses_settled_capacity(setup):
    path, ledger, api, packets, collect = setup
    ledger.limit = 2.5
    batches = r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    assert len(batches) == 1 and len(batches[0]['requests']) == 2 and ledger.total() == 2.
    api.complete(batches[0]['id'])
    batches = r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    assert batches[1]['requests'] == ['r2'] and ledger.total() == pytest.approx(1.2)
    assert len(api.posts) == 2


def test_budget_reservation_is_atomic(setup):
    _, ledger, _, _, _ = setup
    ledger.limit = 1.
    with pytest.raises(RuntimeError, match='budget'):
        ledger.reserve_batch([['a', .6], ['b', .6]])
    assert ledger.total() == 0 and ledger.db.execute('select count(*) from charges').fetchone()[0] == 0


@pytest.mark.parametrize('http_rejection', [False, True])
def test_only_confirmed_queue_rejection_is_split_and_retried(setup, http_rejection):
    path, ledger, api, packets, collect = setup
    error = {'code': 'token_limit_exceeded', 'message': 'Enqueued token limit reached. Limit: 20,000,000 enqueued tokens.'}
    def create(ids):
        if len(ids) > 2:
            if http_rejection: raise r.APIError(400, error)
            return {'status': 'failed', 'errors': {'data': [error]}}
        return {'status': 'in_progress'}
    api.on_create = create
    batches = r.submit_batches(packets(4), 'prompt', path, ledger, api, collect)
    accepted = [b for b in batches if api.batches[b['id']]['status'] == 'in_progress']
    assert [len(b['requests']) for b in accepted] == [2, 2]
    assert ledger.total() == 4.
    assert not (path / 'decisions').exists()  # queue pressure is not a semantic/data error
    assert json.loads((path / 'queue-capacity.json').read_text())['last_api_queue_info']['limit_tokens'] == 20_000_000
    r.submit_batches(packets(4), 'prompt', path, ledger, api, collect)
    assert len(api.posts) == 3 and ledger.total() == 4.  # old rejection must not release new reservations


def test_lost_submission_response_is_recovered_without_resubmitting(setup):
    path, ledger, api, packets, collect = setup
    api.lose_response = True
    with pytest.raises(requests.Timeout):
        r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    assert ledger.total() == 3.
    batches = r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    assert len(api.posts) == 1 and len(batches) == 1 and ledger.total() == 3.


def test_uncertain_submission_without_remote_proof_never_repeats(setup):
    path, ledger, api, packets, collect = setup
    api.reject_before_accept = True
    with pytest.raises(requests.Timeout):
        r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    with pytest.raises(RuntimeError, match='uncertain Batch submission'):
        r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    assert len(api.posts) == 1 and ledger.total() == 3.


def test_later_completed_batch_is_collected_while_first_is_running(setup, monkeypatch):
    path, ledger, api, packets, collect = setup
    monkeypatch.setattr(r, 'BATCH_MAX_REQUESTS', 1)
    sleeps = []
    def tick(_):
        sleeps.append(1)
        if len(sleeps) == 1:
            api.complete('batch-1')
        else:
            assert (path / 'packets/r1/response.json').exists()
            assert not (path / 'packets/r0/response.json').exists()
            api.complete('batch-0')
    monkeypatch.setattr(r.time, 'sleep', tick)
    r.submit_batches(packets(2), 'prompt', path, ledger, api, collect, wait=True)
    assert len(sleeps) == 2 and ledger.total() == pytest.approx(.2)


def test_unrelated_failure_does_not_release_or_retry_paid_work(setup):
    path, ledger, api, packets, collect = setup
    api.on_create = lambda ids: {'status': 'failed', 'errors': {'data': [{'code': 'other_error'}]}}
    r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    assert len(api.posts) == 1 and ledger.total() == 3.


def test_proxy_string_errors_remain_redacted():
    exc = r.APIError(403, 'arbitrary proxy response must not appear')
    assert str(exc) == 'OpenAI HTTP 403: request_failed'
    assert not exc.queue_info


def test_server_error_cannot_release_reservations_even_with_queue_text(setup):
    path, ledger, api, packets, collect = setup
    def create(ids):
        raise r.APIError(500, {'code': 'token_limit_exceeded'})
    api.on_create = create
    with pytest.raises(r.APIError):
        r.submit_batches(packets(3), 'prompt', path, ledger, api, collect)
    assert ledger.total() == 3.
    assert json.loads(next((path / 'submissions').glob('*.json')).read_text())['state'] == 'submitting'


def test_resume_collects_saved_retry_without_resubmission(setup):
    path, ledger, api, packets, _ = setup
    original = packets(1)[0]
    retry = {"request_id": "retry0", "validation_retry_of": "r0", "mapping": {}}
    r.atomic_json(path / "packets/retry0/input.json", retry)
    ledger.reserve("r0", 1.)
    ledger.settle("r0", .1)
    ledger.reserve("retry0", 1.)
    batch = {"id": "old-retry", "requests": ["retry0"]}
    api.batches["old-retry"] = {"id": "old-retry", "ids": ["retry0"],
                                "request_counts": {"total": 1, "completed": 1}}
    api.complete("old-retry")

    # The release wrapper initially provides originals only on restart.
    states = r.collect_batches([batch], [original], None, path, ledger, api, False)
    assert states["old-retry"]["status"] == "completed"
    assert (path / "packets/retry0/response.json").exists()
    assert (path / "batch-receipts/old-retry.json").exists()
    assert ledger.db.execute("select id,amount,state from charges order by id").fetchall() == [
        ("r0", .1, "completed"), ("retry0", .1, "completed")]
    r.collect_batches([batch], [original], None, path, ledger, api, False)
    assert ledger.total() == pytest.approx(.2)
    assert not api.posts  # Retrieval cannot create another paid request.


def test_resume_rejects_mismatched_saved_retry_manifest(setup):
    path, ledger, api, packets, _ = setup
    r.atomic_json(path / "packets/retry0/input.json", {"request_id": "unrelated", "mapping": {}})
    ledger.reserve("retry0", 1.)
    batch = {"id": "old-retry", "requests": ["retry0"]}
    api.batches["old-retry"] = {"id": "old-retry", "ids": ["retry0"]}
    api.complete("old-retry")
    with pytest.raises(ValueError, match="manifest ID disagrees"):
        r.collect_batches([batch], packets(1), None, path, ledger, api, False)
    assert ledger.total() == 1.
    assert not (path / "batch-receipts/old-retry.json").exists()
    assert not api.posts
