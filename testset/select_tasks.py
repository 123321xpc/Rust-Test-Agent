#!/usr/bin/env python3
"""Freeze 50 cached tasks outside the development validation set."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path.home() / 'xpc'
HERE = Path(__file__).resolve().parent
TEST_TASK_COUNT = 50
MIN_TOOLCHAIN_VERSION = (1, 40)
SELECTION = ('50 tasks selected round-robin by sorted repository and instance ID '
             'from the 100 cached tasks outside the frozen validation set; '
             'seven selected tasks below Rust 1.40 were replaced with tasks at Rust 1.40 or newer')
TOOLCHAIN_REPLACEMENTS = {
    'BurntSushi__ripgrep-727': 'chronotope__chrono-378',
    'killercup__cargo-edit-304': 'dtolnay__async-trait-90',
    'mozilla__cbindgen-332': 'mozilla__cbindgen-501',
    'mozilla__cbindgen-401': 'rust-lang__cc-rs-914',
    'tokio-rs__bytes-100': 'tokio-rs__bytes-361',
    'tokio-rs__bytes-186': 'tokio-rs__bytes-376',
    'tokio-rs__bytes-197': 'tokio-rs__bytes-413',
}


def toolchain_version(task):
    match = re.match(r'^(\d+)\.(\d+)', task['rust_toolchain'])
    if not match:
        return (0, 0)
    return tuple(map(int, match.groups()))


def select_tasks(tasks):
    """Select round-robin, then replace the seven pre-1.40 toolchains."""
    tasks = list(tasks)
    by_repo = {}
    for task in sorted(tasks, key=lambda t: t['instance_id']):
        by_repo.setdefault(task['repo'], []).append(task)
    selected = []
    for index in range(max(map(len, by_repo.values()), default=0)):
        for repo in sorted(by_repo):
            if index < len(by_repo[repo]):
                selected.append(by_repo[repo][index])
                if len(selected) == TEST_TASK_COUNT:
                    by_id = {task['instance_id']: task for task in tasks}
                    selected_ids = {task['instance_id'] for task in selected}
                    for old_id, new_id in TOOLCHAIN_REPLACEMENTS.items():
                        if old_id not in selected_ids:
                            raise ValueError(f'{old_id}: expected in original 50-task selection')
                        if new_id not in by_id:
                            raise ValueError(f'{new_id}: replacement task is missing')
                        replacement = by_id[new_id]
                        if toolchain_version(replacement) < MIN_TOOLCHAIN_VERSION:
                            raise ValueError(f'{new_id}: replacement toolchain is older than 1.40')
                        if new_id in selected_ids:
                            raise ValueError(f'{new_id}: replacement task is already selected')
                        selected = [replacement if task['instance_id'] == old_id else task
                                    for task in selected]
                        selected_ids.remove(old_id)
                        selected_ids.add(new_id)
                    if any(toolchain_version(task) < MIN_TOOLCHAIN_VERSION for task in selected):
                        raise ValueError('Selected task set still contains a toolchain older than 1.40')
                    return sorted(selected, key=lambda t: t['instance_id'])
    raise ValueError(f'Expected at least {TEST_TASK_COUNT} eligible tasks')


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
    images = sorted(image for image in images if image.startswith('rustbench/'))
    if len(images) != 110 or len(set(images)) != 110:
        raise ValueError(f'Expected exactly 110 distinct task image tags, found {len(images)}')
    excluded = {task['instance_id']: task['image'] for task in validation['tasks']}
    if len(excluded) != 10 or len(set(excluded.values())) != 10:
        raise ValueError('Validation manifest must contain 10 distinct tasks and images')
    if not set(excluded.values()).issubset(images):
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
        if iid in excluded:
            if image != excluded[iid]:
                raise ValueError(f'{iid}: validation image mismatch')
            continue
        if iid not in bug_reports or not row.get('problem_statement'):
            raise ValueError(f'{iid}: missing bug report or problem statement')
        match = re.match(r'^.*:pr-\d+-rust-(.+)-[0-9a-f]{12}-[0-9a-f]+$', image)
        if not match:
            raise ValueError(f'Cannot parse toolchain from {image}')
        size = changed_lines(row.get('patch') or '')
        bucket = 'small' if size <= 15 else 'medium' if size <= 150 else 'large'
        tasks.append({'instance_id': iid, 'repo': row['repo'], 'base_commit': row['base_commit'],
                      'image': image, 'rust_toolchain': match.group(1),
                      'patch_changed_lines': size, 'size': bucket})
    if len(tasks) != 100 or len({t['instance_id'] for t in tasks}) != 100:
        raise ValueError('Expected 100 distinct non-validation tasks')
    return {'version': 1, 'source_image_count': 110, 'selection': SELECTION,
            'validation_instance_ids': sorted(excluded), 'tasks': select_tasks(tasks)}


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
