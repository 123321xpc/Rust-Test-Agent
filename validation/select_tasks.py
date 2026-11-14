"""Freeze a stratified sample of locally cached Rust-bench task images."""
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "xpc"
OUT = Path(__file__).resolve().parent / "manifest.json"
SEED = "rustbench-validation-v1"


def lines_changed(patch):
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


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} already exists; refusing to change the frozen validation set')
    images = subprocess.check_output(
        ['docker', '-H', 'unix:///var/run/docker.sock', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
        text=True).splitlines()
    tasks = [json.loads(line) for line in (ROOT / 'data/rustbench-all.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    reports = json.loads((ROOT / 'RustForger/data/rb_all_bug_report.json').read_text(encoding='utf-8'))
    prior_reports = sorted({p.parent.name for p in (ROOT / 'Rust-bench/logs/run_evaluation').glob('*/*/*/report.json')})
    candidates = []
    for task in tasks:
        iid = task['instance_id']
        prefix = f"rustbench/{task['repo'].lower().replace('/', '__')}:pr-{task['pull_number']}-rust-"
        found = [image for image in images if image.startswith(prefix) and f"-{task['base_commit'][:12]}-" in image]
        if len(found) != 1 or iid not in reports or not task.get('patch'):
            continue
        if iid in prior_reports:
            continue
        match = re.match(r'^.*:pr-\d+-rust-(.+)-[0-9a-f]{12}-[0-9a-f]+$', found[0])
        if not match:
            raise ValueError(f'Cannot parse toolchain from {found[0]}')
        size = lines_changed(task['patch'])
        bucket = 'small' if size <= 15 else 'medium' if size <= 150 else 'large'
        candidates.append(dict(instance_id=iid, repo=task['repo'], base_commit=task['base_commit'],
                               image=found[0], rust_toolchain=match.group(1),
                               patch_changed_lines=size, size=bucket))
    counts = Counter(row['size'] for row in candidates)
    selected, used = [], set()
    for bucket, n in [('large', 3), ('medium', 4), ('small', 3)]:
        group = [c for c in candidates if c['size'] == bucket]
        group.sort(key=lambda c: hashlib.sha256((SEED + ':' + c['instance_id']).encode()).hexdigest())
        if len(group) < n:
            raise ValueError(f'Not enough {bucket}: {len(group)} (all candidates: {counts})')
        # Prefer distinct repositories across the whole validation set.
        for _ in range(n):
            chosen = next((c for c in group if c['repo'] not in used), group[0])
            group.remove(chosen)
            selected.append(chosen)
            used.add(chosen['repo'])
    manifest = {'version': 1, 'seed': SEED, 'source_image_count': len([x for x in images if x.startswith('rustbench/')]),
                'eligible_count': len(candidates), 'eligible_sizes': dict(counts),
                'size_definition': 'gold patch added plus deleted lines; small <=15, medium 16-150, large >150',
                'selection': 'Exclude tasks with prior evaluation reports; SHA-256 order within each size, preferring repositories not already selected',
                'excluded_prior_report_ids': prior_reports,
                'tasks': selected}
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
