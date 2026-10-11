#!/usr/bin/env python3
"""Resume two-pass Luna Batch refinement of training data, then publish to HF."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

import cv2
import requests

from refine_hf import (APIClient, Ledger, MODEL, ROOT, Source, atomic_json, collect_batches,
                       digest, dumps, export, loads, prepare, prepare_nlp, read_prompt,
                       release_memory, submit_batches, video_key)
from refine_sample import CleanedSource, available_rows, candidate_count, density_bin, load_packets


class ReleaseAPIClient(APIClient):
    def call(self, method, endpoint, **kwargs):
        # Polling/download retries are idempotent. Never blindly repeat a POST:
        # a lost submission response may still represent billable work.
        for attempt in range(6):
            try:
                return super().call(method, endpoint, **kwargs)
            except (requests.RequestException, RuntimeError) as exc:
                transient = isinstance(exc, requests.RequestException) or any(
                    f'HTTP {code}:' in str(exc) for code in [429, 500, 502, 503, 504])
                if method != 'GET' or not transient or attempt == 5:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))


def load_env(path):
    """Read only named credentials; never evaluate shell code or print values."""
    if not path or not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.removeprefix('export ').partition('=')
        if sep and key.strip() in {'OPENAI_API_KEY', 'HF_TOKEN'}:
            os.environ.setdefault(key.strip(), value.strip().strip('\"\''))


def ledger_total(path):
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True) as db:
        return float(db.execute('SELECT COALESCE(SUM(amount),0) FROM charges').fetchone()[0])


def freeze_execution(output):
    files = ['scripts/refine_release.py', 'scripts/refine_hf.py', 'scripts/refine_sample.py',
             'scripts/publish_refined_hf.py', 'src/concor_video/tracklet_schema.py']
    hashes = {name: digest((ROOT / name).read_text()) for name in files}
    path = output / 'execution.json'
    if path.exists():
        if loads(path.read_text())['code_sha256'] != hashes:
            raise ValueError('execution code changed since first submission; restore the recorded version before resuming')
        return
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    atomic_json(path, {'git_commit': commit, 'code_sha256': hashes})


def estimate_release(source, pilot, prior):
    """Density-stratified pilot charges, Batch discount, uncertainty and 50% margin.

    Allocate grouped calls equally to expressions and include validation retries.
    Project one visual request per expression, taking no credit for full-run grouping.
    """
    summary = loads((pilot / 'summary.json').read_text())
    if not summary.get('complete'):
        raise ValueError('cost pilot must be complete before any full-run API submission')
    config = loads((pilot / 'run.json').read_text())
    for key, filename in [('prompt_sha256', 'video_refinement.md'), ('nlp_prompt_sha256', 'nlp_cleanup.md')]:
        if config[key] != digest(read_prompt(ROOT / 'prompts' / filename)[1]):
            raise ValueError('cost pilot used different prompts')
    selection = loads((pilot / 'selection.json').read_text())['videos']
    costs = dict.fromkeys(config['selected_ids'], 0.)
    with sqlite3.connect(pilot / 'budget.sqlite') as db:
        charges = dict(db.execute('SELECT id,amount FROM charges'))
    for folder in [pilot / 'nlp', pilot]:
        for rid in loads((folder / 'requests.json').read_text()):
            manifest = loads((folder / 'packets' / rid / 'input.json').read_text())
            for sid in manifest['mapping']:
                costs[sid] += charges.get(rid, 0.) / len(manifest['mapping'])
    strata = defaultdict(list)
    for video in selection:
        for sid in video['sample_ids']:
            strata[(video['dataset'], video['density_bin'])].append(costs[sid])
    counts = Counter((r['dataset'], density_bin(candidate_count(r))) for r in source.rows.values())
    rows, expected, conservative = [], 0., 0.
    for key, count in sorted(counts.items()):
        values = strata.get(key)
        if not values:
            raise ValueError(f'cost pilot lacks training stratum {key}')
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        upper = mean + 2 * math.sqrt(variance / len(values))
        expected += count * mean * .5
        conservative += count * upper * .5 * 1.5
        rows.append({'dataset': key[0], 'density': key[1], 'training_rows': count,
                     'pilot_rows': len(values), 'pilot_mean_regular_usd': mean})
    return {'training_rows': len(source.rows), 'training_videos': len({video_key(r) for r in source.rows.values()}),
            'expected_batch_usd': round(expected, 4), 'buffered_batch_usd': round(conservative, 4),
            'prior_spend_and_uncertain_usd': prior, 'buffered_total_usd': round(conservative + prior, 4),
            'method': 'Same-prompt, density-stratified pilot; half-price Batch; two standard errors plus 50% margin; no grouping savings.',
            'strata': rows}


def retry_packets(output, originals):
    """One protocol retry, only for invalid responses with a durable API response."""
    retry = []
    request_ids = loads((output / 'requests.json').read_text())
    for manifest in originals:
        if not (output / 'packets' / manifest['request_id'] / 'response.json').exists():
            continue  # Never repeat an uncertain transport failure automatically.
        invalid = []
        for sid in manifest['mapping']:
            path = output / 'decisions' / (digest(sid) + '.json')
            result = loads(path.read_text()) if path.exists() else {}
            if result.get('decision') == 'error' and not result.get('validation_retry_of'):
                invalid.append(sid)
        if not invalid:
            continue
        item = deepcopy(manifest)
        item['request_id'] = digest([manifest['request_id'], 'validation-retry', sorted(invalid)])[:24]
        item['validation_retry_of'] = manifest['request_id']
        item['mapping'] = {sid: manifest['mapping'][sid] for sid in invalid}
        item['packet']['expressions'] = [e for e in manifest['packet']['expressions'] if e['sample_id'] in invalid]
        item['packet']['validation_errors'] = {
            sid: loads((output / 'decisions' / (digest(sid) + '.json')).read_text())['reason_codes'] for sid in invalid}
        atomic_json(output / 'packets' / item['request_id'] / 'input.json', item)
        if item['request_id'] not in request_ids:
            request_ids.append(item['request_id'])
        retry.append(item)
    atomic_json(output / 'requests.json', request_ids)
    # On resume, collect an already submitted retry even if original decisions changed.
    known = {p['request_id'] for p in retry}
    for rid in request_ids:
        item = loads((output / 'packets' / rid / 'input.json').read_text())
        if item.get('validation_retry_of') and rid not in known:
            retry.append(item)
    return retry


def run_stage(packets, prompt, source, output, ledger, client):
    all_packets = packets
    def collect(batches):
        return collect_batches(batches, all_packets, source, output, ledger, client, False)
    submit_batches(packets, prompt, output, ledger, client, collect, wait=True)
    release_memory()
    retries = retry_packets(output, packets)
    if retries:
        all_packets = packets + retries
        submit_batches(retries, prompt, output, ledger, client, collect, wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Training Parquet only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-root', type=Path, action='append', required=True)
    parser.add_argument('--pilot-run', type=Path, required=True)
    parser.add_argument('--prior-ledger', type=Path, action='append', default=[])
    parser.add_argument('--total-budget-usd', type=float, default=99.)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--hf-token-file', type=Path)
    parser.add_argument('--hf-repo', default='suryadv/concor-video-data-filtered')
    parser.add_argument('--mode', choices=['prepare', 'submit', 'submit-first', 'run'], default='prepare',
                        help='submit queues all currently admissible NLP work; run resumes, monitors both passes and exports (submit-first is a legacy alias)')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if not 0 < args.total_budget_usd < 100:
        parser.error('total budget must be strictly below $100')
    cv2.setNumThreads(1)
    args.group_size, args.group_tracklets, args.max_tracklets, args.sample_count = 12, 24, 64, 12
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        value = {'stage': stage, 'updated_unix': time.time(), **extra}
        atomic_json(output / 'status.json', value)
        print(dumps(value), flush=True)
    client = ledger = None
    try:
        source = Source([args.input.resolve()], include_media=True)
        if any(r['split'] != 'train' for r in source.rows.values()):
            raise ValueError('release input contains non-training rows')
        ids = sorted(source.rows, key=lambda sid: source.locations[sid])
        if len(available_rows(source, args.media_root)) != len(ids):
            raise ValueError('some training media sources are unavailable')
        prior_paths = set(p.resolve() for p in args.prior_ledger) | {(args.pilot_run / 'budget.sqlite').resolve()}
        if (output / 'budget.sqlite').resolve() in prior_paths:
            raise ValueError('current ledger cannot also be prior spending')
        prior = sum(ledger_total(p) for p in prior_paths)
        if args.mode == 'prepare' and not (args.pilot_run / 'summary.json').exists():
            estimate = {'status': 'awaiting_completed_pilot', 'training_rows': len(ids)}
        else:
            estimate = estimate_release(source, args.pilot_run, prior)
        atomic_json(output / 'estimate.json', estimate)
        if estimate.get('buffered_total_usd', 0) >= args.total_budget_usd:
            raise ValueError('buffered estimate exceeds the total budget; no full-run calls submitted')
        prompts = [read_prompt(ROOT / 'prompts' / name) for name in ['video_refinement.md', 'nlp_cleanup.md']]
        config = {'version': 'concor-training-two-pass-v2', 'model': MODEL, 'reasoning_effort': 'medium',
                  'selected_ids': ids, 'prompt_sha256': digest(prompts[0][1]), 'nlp_prompt_sha256': digest(prompts[1][1]),
                  'source': {'path': str(args.input.resolve()), 'size': args.input.stat().st_size,
                             'mtime_ns': args.input.stat().st_mtime_ns},
                  'prior_ledgers': sorted(str(p) for p in prior_paths), 'total_budget_usd': args.total_budget_usd}
        if (output / 'run.json').exists() and loads((output / 'run.json').read_text()) != config:
            raise ValueError('inputs/prompts/budget changed; use the matching run directory')
        atomic_json(output / 'run.json', config)
        ledger = Ledger(output / 'budget.sqlite', args.total_budget_usd - prior)
        if ledger.total() > ledger.limit:
            raise ValueError('prior spending leaves insufficient budget to continue')
        nlp = output / 'nlp'
        for folder, (header, prompt) in zip([output, nlp], prompts):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'prompt_sent.txt').write_text(prompt)
            (folder / 'prompt_metadata.txt').write_text(header)
        status('preparing_nlp', estimate=estimate)
        if (nlp / 'requests.json').exists():
            packets, early = load_packets(nlp), loads((nlp / 'heuristics.json').read_text())
        else:
            packets, early = prepare_nlp(source, ids, args, prompts[1][1], nlp)
        if args.mode == 'prepare':
            status('prepared_nlp', requests=len(packets), structural_rejections=len(early), estimate=estimate)
            return
        freeze_execution(output)
        load_env(args.env_file)
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY is missing')
        client = ReleaseAPIClient()
        status('nlp_batch', requests=len(packets))
        if args.mode in {'submit', 'submit-first'}:
            collect = lambda batches: collect_batches(batches, packets, source, nlp, ledger, client, False)
            batches = submit_batches(packets, prompts[1][1], nlp, ledger, client, collect)
            status('nlp_batch_queued', batches=[b['id'] for b in batches],
                   cost_upper_usd=ledger.total(), resume_mode='run')
            return
        run_stage(packets, prompts[1][1], source, nlp, ledger, client)
        ready, nlp_counts = [], Counter()
        for sid in ids:
            path = nlp / 'decisions' / (digest(sid) + '.json')
            decision = loads(path.read_text()) if path.exists() else early.get(sid, {'decision': 'pending'})
            nlp_counts[decision['decision']] += 1
            if decision['decision'] in {'keep', 'rewrite'}:
                ready.append(sid)
            else:
                early[sid] = {'sample_id': sid, **decision}
        if nlp_counts['error'] or nlp_counts['pending']:
            atomic_json(output / 'unresolved_nlp.json', early)
            raise RuntimeError('NLP errors/pending results require recovery before visual processing')
        cleaned = CleanedSource(source, nlp)
        status('preparing_visual', nlp_counts=dict(nlp_counts))
        if (output / 'requests.json').exists():
            packets, visual_early = load_packets(output), loads((output / 'heuristics.json').read_text())
        else:
            packets, visual_early = prepare(cleaned, ready, args, prompts[0][1], output)
        early.update(visual_early)
        atomic_json(output / 'heuristics.json', early)
        release_memory()
        status('visual_batch', requests=len(packets), cost_upper_usd=ledger.total())
        run_stage(packets, prompts[0][1], cleaned, output, ledger, client)
        status('exporting')
        summary = export(cleaned, ids, output, early, ledger)
        summary.update(nlp_counts=dict(nlp_counts), videos=estimate['training_videos'],
                       prior_spend_upper_usd=prior, total_with_prior_upper_usd=prior + ledger.total())
        atomic_json(output / 'summary.json', summary)
        if not summary['complete']:
            raise RuntimeError('release has unresolved errors/pending rows; automatic HF publication withheld')
        if args.publish:
            from publish_refined_hf import publish
            load_env(args.env_file)  # also pick up a credential added during the Batch run
            status('publishing_hf', summary=summary)
            receipt = publish(output, args.hf_repo, args.hf_token_file)
            status('published', summary=summary, publication=receipt)
        else:
            status('complete', summary=summary)
    except Exception as exc:
        # Exception text from credential/network libraries can contain secrets.
        status('stopped', error_type=type(exc).__name__,
               detail=re.sub(r'(?:hf_|sk-)[A-Za-z0-9_-]+', '[REDACTED]', str(exc))
                      if isinstance(exc, (ValueError, RuntimeError)) else 'See local checkpoints; credentials omitted.')
        raise
    finally:
        if client: client.close()
        if ledger: ledger.db.close()
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Release stopped ({type(exc).__name__}); inspect status.json.', file=sys.stderr)
        raise SystemExit(1)
