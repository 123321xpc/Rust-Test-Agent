#!/usr/bin/env python3
"""Freeze the balanced 100-task experiment set."""
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path.home() / 'xpc'
HERE = Path(__file__).resolve().parent

EXPERIMENT_TASK_COUNT = 100
SMALL_TASK_COUNT = 40
MEDIUM_TASK_COUNT = 55
LARGE_TASK_COUNT = 5
EXPECTED_SOURCE_IMAGE_COUNT = 109

SELECTED_LARGE_IDS = (
    'async-graphql__async-graphql-10',
    'bevyengine__bevy-15546',
    'biomejs__biome-2404',
    'cargo-generate__cargo-generate-372',
    'killercup__cargo-edit-732',
)

SELECTION = (
    '100 tasks: 40 small patches (<=15 lines), 55 medium patches (16-150 lines), '
    'and 5 large patches (>150 lines); encompasses all 50 testset tasks, excludes '
    'validation small tasks, and selects 5 diverse cached large tasks'
)


def changed_lines(patch):
    in_hunk = False
    count = 0
    for line in (patch or '').splitlines():
        if line.startswith('diff --git '):
            in_hunk = False
        elif line.startswith('@@ '):
            in_hunk = True
        elif in_hunk and line.startswith(('+', '-')):
            count += 1
    return count


def patch_size(line_count):
    return 'small' if line_count <= 15 else 'medium' if line_count <= 150 else 'large'


def select_tasks(tasks, val_ids, ts_ids):
    by_id = {t['instance_id']: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError('Duplicate task in candidate pool')

    small_tasks = [t for t in tasks if t['size'] == 'small']
    med_tasks = [t for t in tasks if t['size'] == 'medium']
    large_tasks = [t for t in tasks if t['size'] == 'large']

    # 40 small: all non-validation small tasks
    sel_small = [t for t in small_tasks if t['instance_id'] not in val_ids]
    if len(sel_small) != SMALL_TASK_COUNT:
        raise ValueError(f'Expected {SMALL_TASK_COUNT} small tasks, got {len(sel_small)}')

    # 55 medium: all available medium tasks
    sel_med = list(med_tasks)
    if len(sel_med) != MEDIUM_TASK_COUNT:
        raise ValueError(f'Expected {MEDIUM_TASK_COUNT} medium tasks, got {len(sel_med)}')

    # 5 large: 5 distinct repositories
    sel_large = [t for t in large_tasks if t['instance_id'] in SELECTED_LARGE_IDS]
    if len(sel_large) != LARGE_TASK_COUNT:
        raise ValueError(f'Expected {LARGE_TASK_COUNT} large tasks, got {len(sel_large)}')

    selected = sorted(sel_small + sel_med + sel_large, key=lambda t: t['instance_id'])
    if len(selected) != EXPERIMENT_TASK_COUNT or len({t['instance_id'] for t in selected}) != EXPERIMENT_TASK_COUNT:
        raise ValueError('Must contain 100 distinct task IDs')
    if len({t['image'] for t in selected}) != EXPERIMENT_TASK_COUNT:
        raise ValueError('Must contain 100 distinct task images')
    if not ts_ids.issubset({t['instance_id'] for t in selected}):
        raise ValueError('Must contain all 50 testset tasks')

    counts = Counter(t['size'] for t in selected)
    if counts != Counter({'small': SMALL_TASK_COUNT, 'medium': MEDIUM_TASK_COUNT, 'large': LARGE_TASK_COUNT}):
        raise ValueError(f'Size breakdown mismatch: {dict(counts)}')

    return selected


def build_manifest(images, rows, validation, testset):
    canonical_tag = re.compile(r'^rustbench/[^:]+:pr-\d+-rust-.+-[0-9a-f]{12}-[0-9a-f]{10}$')
    images = sorted({image for image in images if image.startswith('rustbench/') and canonical_tag.fullmatch(image)})
    if len(images) != EXPECTED_SOURCE_IMAGE_COUNT:
        raise ValueError(f'Expected exactly {EXPECTED_SOURCE_IMAGE_COUNT} images, found {len(images)}')

    val_ids = {t['instance_id'] for t in validation['tasks']}
    ts_ids = {t['instance_id'] for t in testset['tasks']}

    by_image = {}
    for r in rows:
        prefix = f"rustbench/{r['repo'].lower().replace('/', '__')}:pr-{r['pull_number']}-rust-"
        for image in images:
            if image.startswith(prefix) and f"-{r['base_commit'][:12]}-" in image:
                by_image.setdefault(image, []).append(r)

    tasks = []
    for image in images:
        matches = by_image.get(image, [])
        if len(matches) != 1:
            raise ValueError(f'{image}: expected 1 match, found {len(matches)}')
        row = matches[0]
        match = re.match(r'^.*:pr-\d+-rust-(.+)-[0-9a-f]{12}-[0-9a-f]{10}$', image)
        if not match:
            raise ValueError(f'Cannot parse toolchain from {image}')
        size = changed_lines(row.get('patch') or '')
        tasks.append({
            'instance_id': row['instance_id'],
            'repo': row['repo'],
            'base_commit': row['base_commit'],
            'image': image,
            'rust_toolchain': match.group(1),
            'patch_changed_lines': size,
            'size': patch_size(size),
        })

    return {
        'version': 1,
        'source_image_count': len(images),
        'selection': SELECTION,
        'tasks': select_tasks(tasks, val_ids, ts_ids),
    }


def main():
    output = HERE / 'manifest.json'
    dataset_path = ROOT / 'data/rustbench-all.jsonl'
    val_path = ROOT / 'validation/manifest.json'
    ts_path = ROOT / 'testset/manifest.json'

    rows = [json.loads(line) for line in dataset_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    images = subprocess.check_output(
        ['docker', '-H', 'unix:///var/run/docker.sock', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
        text=True,
    ).splitlines()

    manifest = build_manifest(images, rows, json.loads(val_path.read_text(encoding='utf-8')),
                              json.loads(ts_path.read_text(encoding='utf-8')))
    manifest['dataset_sha256'] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest['testset_manifest_sha256'] = hashlib.sha256(ts_path.read_bytes()).hexdigest()
    manifest['validation_manifest_sha256'] = hashlib.sha256(val_path.read_bytes()).hexdigest()

    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Frozen {len(manifest["tasks"])} tasks in {output}')


if __name__ == '__main__':
    main()
