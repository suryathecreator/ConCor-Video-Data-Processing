#!/usr/bin/env python3
"""Publish a completed training refinement as the additive gpt_refined config."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from refine_hf import atomic_json, digest, loads, release_memory

# All 22 source fields remain available, with explicit, readable names. JSON
# fields retain exact bytes; common navigation fields also have native columns.
COLUMNS = {
    'sample_id': 'sample_id', 'dataset': 'source_dataset', 'split': 'source_split',
    'video_id': 'video_id', 'expression_id': 'expression_id', 'text': 'caption',
    'negative': 'is_nonexistent', 'cohort': 'annotation_category',
    'annotation_protocol': 'annotation_protocol', 'target_source': 'target_mask_source',
    'disposition': 'annotation_status', 'frame_ids_json': 'frame_ids_json',
    'frame_files_json': 'frame_paths_json', 'tracklets_json': 'object_tracks_json',
    'groups_json': 'text_entities_json', 'span_links_json': 'caption_object_links_json',
    'dataset_root': 'source_media_root', 'frame_source': 'source_media_location',
    'provenance_warning': 'provenance_note', 'extraction_json': 'original_entity_extraction_json',
    'sam_prompt_audit_json': 'segmentation_audit_json', 'runtime_seconds': 'generation_runtime_seconds',
}
LINK_TYPE = pa.list_(pa.struct([pa.field('start', pa.int64()), pa.field('end', pa.int64()),
                              pa.field('text', pa.string()), pa.field('object_ids', pa.list_(pa.string()))]))


def read_token(path=None):
    value = os.environ.get('HF_TOKEN')
    if not value and path:
        matches = list(dict.fromkeys(re.findall(r'hf_[A-Za-z0-9]{25,}', Path(path).read_text())))
        if len(matches) != 1:
            raise ValueError('HF credential file must contain exactly one distinct token')
        value = matches[0]
    if not value:
        from huggingface_hub import get_token
        value = get_token()
    if not value:
        raise ValueError('HF write credential is missing')
    return value


def public_table(table, decisions):
    if set(table.column_names) != set(COLUMNS):
        raise ValueError('unexpected source columns; do not silently drop provenance')
    result = table.select(list(COLUMNS)).rename_columns(list(COLUMNS.values()))
    rows = table.select(['sample_id', 'tracklets_json', 'span_links_json', 'frame_ids_json']).to_pylist()
    links = [[{'start': link['start'], 'end': link['end'], 'text': link['text'],
               'object_ids': link['tracklet_ids']} for link in loads(r['span_links_json'], [])] for r in rows]
    for name, values, typ in [
        ('frame_ids', [loads(r['frame_ids_json'], []) for r in rows], pa.list_(pa.string())),
        ('caption_links', links, LINK_TYPE),
        ('object_count', [len(loads(r['tracklets_json'], [])) for r in rows], pa.int32()),
        ('linked_phrase_count', [len(v) for v in links], pa.int32()),
        ('refinement_decision', [decisions[r['sample_id']] for r in rows], pa.string()),
    ]:
        result = result.append_column(name, pa.array(values, type=typ))
    return result.replace_schema_metadata({b'concor_refinement': b'gpt-6-luna; medium; NLP then sampled-video review; v2',
                                           b'original_column_mapping': json.dumps(COLUMNS, sort_keys=True).encode()})


def prepare_publication(run):
    summary = loads((run / 'summary.json').read_text())
    if not summary.get('complete') or summary['counts'].get('error') or summary['counts'].get('pending'):
        raise ValueError('cannot publish incomplete refinement')
    audit = loads((run / 'audit.json').read_text())
    config = loads((run / 'run.json').read_text())
    if len(audit) != summary['selected'] or {r['sample_id'] for r in audit} != set(config['selected_ids']):
        raise ValueError('audit does not cover every selected training expression')
    decisions = {r['sample_id']: r['decision'] for r in audit}
    source = pq.ParquetFile(run / 'train.parquet')
    expected = sum(summary['counts'].get(k, 0) for k in ['keep', 'rewrite'])
    if source.metadata.num_rows != expected:
        raise ValueError('retained row count disagrees with audit')
    destination = run / 'publication'
    data_dir = destination / 'gpt_refined' / 'train'
    data_dir.mkdir(parents=True, exist_ok=True)
    writer, shard, size, counts = None, -1, 0, Counter()
    files = []
    try:
        for batch in source.iter_batches(batch_size=32):
            table = pa.Table.from_batches([batch])
            if any(v != 'train' for v in table['split'].to_pylist()):
                raise ValueError('publication contains non-training data')
            counts.update(table['dataset'].to_pylist())
            public = public_table(table, decisions)
            if writer is None or size >= 200_000_000:
                if writer: writer.close()
                shard += 1
                path = data_dir / f'train-{shard:05d}.parquet'
                files.append(path)
                writer = pq.ParquetWriter(path, public.schema, compression='zstd')
                size = 0
            writer.write_table(public, row_group_size=32)
            size += public.nbytes
            del table, public, batch
            release_memory()
    finally:
        if writer: writer.close()
    if not files:
        raise ValueError('no retained training rows to publish')
    # A resume must not publish stale shards from an older/incomplete build.
    files += [run / name for name in ['summary.json', 'audit.json', 'estimate.json', 'prompt_sent.txt']]
    manifest = {'config': 'gpt_refined', 'split': 'train', 'source_rows': summary['selected'],
                'retained_rows': expected, 'rows_by_dataset': dict(counts),
                'model': 'gpt-6-luna', 'reasoning_effort': 'medium',
                'prompt_sha256': config['prompt_sha256'], 'nlp_prompt_sha256': config['nlp_prompt_sha256'],
                'column_mapping': COLUMNS, 'summary': summary,
                'source': {k: v for k, v in config.get('source', {}).items() if k != 'path'},
                'execution': loads((run / 'execution.json').read_text()) if (run / 'execution.json').exists() else None,
                'pipeline': ['structural validation and per-entity 10-track ceiling', 'text-only entity-link cleanup',
                             'sampled-video caption and BCC refinement', 'local exact-span and bidirectional-link validation']}
    atomic_json(destination / 'processing.json', manifest)
    return destination, files, manifest


def update_card(text, existing_files, manifest):
    if not text.startswith('---\n') or '\n---' not in text[4:]:
        raise ValueError('existing dataset card has no YAML metadata')
    header, body = text[4:].split('\n---', 1)
    metadata = yaml.safe_load(header) or {}
    configs = metadata.get('configs')
    if configs is None:
        # Preserve the original train/eval paths as an explicit source config;
        # adding differently shaped Parquets must not break automatic discovery.
        splits = [{'split': 'train', 'path': 'train/verification.parquet'},
                  {'split': 'eval_unfiltered', 'path': 'eval/unfiltered/verification.parquet'}]
        for filename in ['eval/filtered/verified.parquet', 'eval/filtered/verification.parquet']:
            if filename in existing_files:
                splits.append({'split': 'eval_filtered', 'path': filename})
                break
        configs = [{'config_name': 'default', 'default': True, 'data_files': splits}]
    configs = [c for c in configs if c['config_name'] != 'gpt_refined']
    configs.append({'config_name': 'gpt_refined', 'data_files': [{'split': 'train', 'path': 'gpt_refined/train/*.parquet'}]})
    metadata['configs'] = configs
    start, end = '<!-- concor-gpt-refined:start -->', '<!-- concor-gpt-refined:end -->'
    if start in body:
        before, rest = body.split(start, 1)
        if end not in rest: raise ValueError('incomplete refinement card section')
        body = before + rest.split(end, 1)[1]
    section = f'''{start}
## GPT-refined training data

Select **`gpt_refined` → `train`** for {manifest['retained_rows']:,} retained expressions from all {manifest['source_rows']:,} original training expressions. Original source files and manually reviewed evaluation annotations remain available.

```python
from datasets import load_dataset
data = load_dataset("suryadv/concor-video-data-filtered", "gpt_refined", split="train")
```

Processing uses **GPT-6 Luna, medium reasoning**, in two passes: text-only correction of linguistic/entity links, followed by shared sampled-video caption and correspondence review. Structural checks reject malformed targets and any entity linked to more than 10 distinct nonempty tracks, remove empty context tracks and exact duplicates, and cap pathological examples at 64 tracks. Model evidence pairs original frames with numbered mask overlays (540px maximum height, JPEG quality 88), using 12 evenly spaced indices including endpoints plus first appearances. Groups share up to 12 expressions and split above 24 distinct tracks. No new segmentation or exhaustive frame inspection is performed.

Taxonomy/parenthetical glosses are not grounded as extra entity mentions. Captions omit unsupported unsegmented background entities. Intentional nonexistent targets are accepted by default with their exact text and no target masks; examples with unsegmented context entities are rejected. Segmented context may retain its own links. Local validation rebuilds exact character spans and both correspondence directions, preserves intended positive targets, and retains complete mask sequences. Errors and pending results are never counted as semantic rejections; automatic publication requires resolved outcomes for every input.

Useful columns: **`caption`**, **`is_nonexistent`**, **`source_dataset`**, **`video_id`**, **`object_count`**, **`caption_links`** (native lists containing phrase offsets, text, and object IDs), and **`refinement_decision`**. Offsets are zero-based Unicode character positions, with exclusive `end`. `frame_ids` is a native list. Full masks are preserved losslessly in `object_tracks_json`; `text_entities_json` includes the reverse object-to-mention grouping. All 22 original fields remain under clearer names; JSON-suffixed fields are decoded with `json.loads`. The exact field mapping, outcome counts, cost ledger summary and prompts are in [`gpt_refined/processing.json`](gpt_refined/processing.json) and [`gpt_refined/audit/`](gpt_refined/audit/).

This is automated refinement, not a claim of perfect labels. Sparse sampled evidence can miss events; source media and annotations retain their upstream terms. [Wrapper and prompts](https://github.com/suryathecreator/ConCor-Video-Data-Processing) · [Public 200-video review](https://concor-video-refinement-pilot.daisydv.chatgpt.site).
{end}'''
    return '---\n' + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + '---' + body.rstrip() + '\n\n' + section + '\n'


def publish(run, repo, token_file=None):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    token = read_token(token_file)
    api = HfApi(token=token)
    who = api.whoami()
    if who.get('auth', {}).get('accessToken', {}).get('role') == 'read':
        raise ValueError('HF credential is read-only; publication needs write access')
    destination, files, manifest = prepare_publication(run)
    info = api.dataset_info(repo)
    existing_files = [f.rfilename for f in info.siblings]
    readme = Path(hf_hub_download(repo, 'README.md', repo_type='dataset', revision=info.sha,
                                token=token, cache_dir=run / 'hf-cache')).read_text()
    card = update_card(readme, existing_files, manifest)
    (destination / 'README.md').write_text(card)
    operations = [CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=(destination / 'README.md')),
                  CommitOperationAdd(path_in_repo='gpt_refined/processing.json', path_or_fileobj=destination / 'processing.json')]
    for file in files:
        path = str(file.relative_to(destination)) if file.is_relative_to(destination) else 'gpt_refined/audit/' + file.name
        operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=file))
    operations.append(CommitOperationAdd(path_in_repo='gpt_refined/audit/nlp_prompt_sent.txt', path_or_fileobj=run / 'nlp/prompt_sent.txt'))
    fingerprint = digest(manifest)
    receipt_path = run / 'publication.json'
    # Confirm a timed-out successful commit before attempting another publication.
    if 'gpt_refined/processing.json' in existing_files:
        remote = Path(hf_hub_download(repo, 'gpt_refined/processing.json', repo_type='dataset', revision=info.sha,
                                     token=token, cache_dir=run / 'hf-cache'))
        if digest(loads(remote.read_text())) == fingerprint:
            receipt = {'repo': repo, 'commit': info.sha, 'config': 'gpt_refined', 'split': 'train', 'retained_rows': manifest['retained_rows']}
            atomic_json(receipt_path, receipt)
            return receipt
        raise ValueError('gpt_refined already contains a different release; refusing an unattended replacement')
    commit = api.create_commit(repo_id=repo, repo_type='dataset', operations=operations,
                               parent_commit=info.sha, commit_message='Add GPT-refined training annotations with NLP and video BCC review',
                               commit_description='Two-pass GPT-6 Luna medium refinement; per-entity mask ceiling, taxonomy/background rules, preserved intentional negatives, readable columns and complete processing audit.')
    verified = api.dataset_info(repo, revision=commit.oid)
    expected_paths = {op.path_in_repo for op in operations}
    if not expected_paths <= {f.rfilename for f in verified.siblings}:
        raise RuntimeError('HF commit verification failed')
    receipt = {'repo': repo, 'commit': commit.oid, 'config': 'gpt_refined', 'split': 'train',
               'retained_rows': manifest['retained_rows'], 'url': f'https://huggingface.co/datasets/{repo}/tree/{commit.oid}/gpt_refined'}
    atomic_json(receipt_path, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repo', default='suryadv/concor-video-data-filtered')
    parser.add_argument('--hf-token-file', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(publish(args.run.resolve(), args.repo, args.hf_token_file)))
    except Exception as exc:
        raise SystemExit(f'Publication stopped ({type(exc).__name__}); credentials omitted.')
