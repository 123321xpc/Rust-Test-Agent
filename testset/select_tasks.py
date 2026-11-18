#!/usr/bin/env python3
"""Freeze the balanced 50-task test set."""
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path.home() / 'xpc'
HERE = Path(__file__).resolve().parent
TEST_TASK_COUNT = 50
SMALL_TASK_COUNT = 25
MEDIUM_TASK_COUNT = 25
EXPECTED_SOURCE_IMAGE_COUNT = 113
MIN_TOOLCHAIN_VERSION = (1, 40)
SELECTION = ('25 small patches (at most 15 changed lines) and 25 medium patches '
             '(16-150 changed lines); all six large patches removed; ten small tasks '
             'chosen from already-cached candidate images and three additional small '
             'task images built through the configured proxy; selected Cargo toolchains '
             'are Rust 1.40 or newer')
SELECTED_INSTANCE_IDS = (
    'BurntSushi__ripgrep-2209',
    'ast-grep__ast-grep-1558',
    'async-graphql__async-graphql-1048',
    'async-graphql__async-graphql-1099',
    'async-graphql__async-graphql-1228',
    'async-graphql__async-graphql-1491',
    'axodotdev__cargo-dist-1024',
    'axodotdev__cargo-dist-1655',
    'aya-rs__aya-431',
    'aya-rs__aya-656',
    'aya-rs__aya-774',
    'aya-rs__aya-897',
    'aya-rs__aya-90',
    'boa-dev__boa-4047',
    'cargo-generate__cargo-generate-474',
    'cargo-generate__cargo-generate-511',
    'chronotope__chrono-1621',
    'chronotope__chrono-378',
    'chronotope__chrono-445',
    'chronotope__chrono-508',
    'chronotope__chrono-686',
    'chronotope__chrono-999',
    'dtolnay__async-trait-100',
    'dtolnay__async-trait-160',
    'dtolnay__async-trait-227',
    'dtolnay__async-trait-90',
    'imsnif__bandwhich-118',
    'mozilla__cbindgen-494',
    'mozilla__cbindgen-501',
    'nushell__nushell-12901',
    'rust-fuzz__cargo-fuzz-231',
    'rust-fuzz__cargo-fuzz-388',
    'rust-lang__cc-rs-1181',
    'rust-lang__cc-rs-1212',
    'rust-lang__cc-rs-1279',
    'rust-lang__cc-rs-914',
    'rust-lang__cc-rs-991',
    'rust-lang__chalk-755',
    'serde-rs__serde-2709',
    'starkware-libs__cairo-4909',
    'tokio-rs__bytes-361',
    'tokio-rs__bytes-376',
    'tokio-rs__bytes-413',
    'tokio-rs__bytes-560',
    'tokio-rs__bytes-721',
    'tokio-rs__bytes-732',
    'tokio-rs__tokio-4519',
    'tokio-rs__tokio-5343',
    'tracel-ai__burn-815',
    'bevyengine__bevy-15398',
)


def toolchain_version(task):
    match = re.match(r'^(\d+)\.(\d+)', task['rust_toolchain'])
    if not match:
        return (0, 0)
    return tuple(map(int, match.groups()))


def patch_size(line_count):
    return 'small' if line_count <= 15 else 'medium' if line_count <= 150 else 'large'


def select_tasks(tasks):
    """Return the explicit frozen selection after checking its size/toolchain contract."""
    by_id = {}
    for task in tasks:
        iid = task['instance_id']
        if iid in by_id:
            raise ValueError(f'{iid}: duplicate task in candidate pool')
        by_id[iid] = task
    missing = sorted(set(SELECTED_INSTANCE_IDS) - set(by_id))
    if missing:
        raise ValueError('Selected tasks are missing from the candidate pool: ' + ', '.join(missing))
    if len(SELECTED_INSTANCE_IDS) != TEST_TASK_COUNT or len(set(SELECTED_INSTANCE_IDS)) != TEST_TASK_COUNT:
        raise ValueError('Frozen selection must contain 50 distinct task IDs')

    selected = [by_id[iid] for iid in SELECTED_INSTANCE_IDS]
    for task in selected:
        expected_size = patch_size(task['patch_changed_lines'])
        if task['size'] != expected_size:
            raise ValueError(f"{task['instance_id']}: size does not match changed-line count")
        if expected_size == 'large':
            raise ValueError(f"{task['instance_id']}: large patches are not allowed")
        if toolchain_version(task) < MIN_TOOLCHAIN_VERSION:
            raise ValueError(f"{task['instance_id']}: Rust toolchain is older than 1.40")
    counts = Counter(task['size'] for task in selected)
    if counts != Counter({'small': SMALL_TASK_COUNT, 'medium': MEDIUM_TASK_COUNT}):
        raise ValueError(f'Expected 25 small and 25 medium tasks; found {dict(counts)}')
    if len({task['image'] for task in selected}) != TEST_TASK_COUNT:
        raise ValueError('Frozen selection must contain 50 distinct task images')
    return sorted(selected, key=lambda task: task['instance_id'])


def changed_lines(patch):
    in_hunk = False
    count = 0
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            in_hunk = False
        elif line.startswith('@@ '):
            in_hunk = True
        elif in_hunk and line.startswith(('+', '-')):
            count += 1
    return count


def build_manifest(images, rows, validation, bug_reports):
    canonical_tag = re.compile(
        r'^rustbench/[^:]+:pr-\d+-rust-.+-[0-9a-f]{12}-[0-9a-f]{10}$')
    images = sorted({image for image in images
                     if image.startswith('rustbench/') and canonical_tag.fullmatch(image)})
    if len(images) != EXPECTED_SOURCE_IMAGE_COUNT:
        raise ValueError(f'Expected exactly {EXPECTED_SOURCE_IMAGE_COUNT} distinct canonical task image tags, found {len(images)}')
    validation_tasks = {task['instance_id']: task['image'] for task in validation['tasks']}
    source_validation_ids = set(json.loads(
        (HERE / 'manifest-100.json').read_text(encoding='utf-8'))['validation_instance_ids'])
    excluded_ids = source_validation_ids
    if len(source_validation_ids) != 10 or not set(validation_tasks).issubset(source_validation_ids):
        raise ValueError('Original validation exclusion list must contain the active validation tasks')
    if not set(validation_tasks.values()).issubset(images):
        raise ValueError('Some frozen validation images are missing')
    by_image = {}
    for row in rows:
        prefix = f"rustbench/{row['repo'].lower().replace('/', '__')}:pr-{row['pull_number']}-rust-"
        for image in images:
            if image.startswith(prefix) and f"-{row['base_commit'][:12]}-" in image:
                by_image.setdefault(image, []).append(row)
    tasks = []
    for image in images:
        matches = by_image.get(image, [])
        if len(matches) != 1:
            raise ValueError(f'{image}: expected one matching dataset row, found {len(matches)}')
        row = matches[0]
        iid = row['instance_id']
        if iid in excluded_ids:
            if iid in validation_tasks and image != validation_tasks[iid]:
                raise ValueError(f'{iid}: validation image mismatch')
            continue
        if iid not in bug_reports or not row.get('problem_statement'):
            raise ValueError(f'{iid}: missing bug report or problem statement')
        match = re.match(r'^.*:pr-\d+-rust-(.+)-[0-9a-f]{12}-[0-9a-f]{10}$', image)
        if not match:
            raise ValueError(f'Cannot parse toolchain from {image}')
        size = changed_lines(row.get('patch') or '')
        tasks.append({'instance_id': iid, 'repo': row['repo'], 'base_commit': row['base_commit'],
                      'image': image, 'rust_toolchain': match.group(1),
                      'patch_changed_lines': size, 'size': patch_size(size)})
    expected_candidates = EXPECTED_SOURCE_IMAGE_COUNT - len(excluded_ids)
    if len(tasks) != expected_candidates or len({t['instance_id'] for t in tasks}) != expected_candidates:
        raise ValueError(f'Expected {expected_candidates} distinct non-validation candidate tasks')
    return {'version': 1, 'source_image_count': len(images), 'selection': SELECTION,
            'validation_instance_ids': sorted(excluded_ids), 'tasks': select_tasks(tasks)}


def main():
    output = HERE / 'manifest.json'
    if output.exists():
        raise FileExistsError(f'{output} already exists; refusing to overwrite the frozen test set')
    dataset_path = ROOT / 'data/rustbench-all.jsonl'
    validation_path = ROOT / 'validation/manifest.json'
    rows = [json.loads(line) for line in dataset_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    images = subprocess.check_output(['docker', '-H', 'unix:///var/run/docker.sock', 'images',
                                      '--format', '{{.Repository}}:{{.Tag}}'], text=True).splitlines()
    manifest = build_manifest(images, rows, json.loads(validation_path.read_text(encoding='utf-8')),
                              json.loads((ROOT / 'RustForger/data/rb_all_bug_report.json').read_text(encoding='utf-8')))
    manifest['dataset_sha256'] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest['validation_manifest_sha256'] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Frozen {len(manifest["tasks"])} tasks in {output}')


if __name__ == '__main__':
    main()
