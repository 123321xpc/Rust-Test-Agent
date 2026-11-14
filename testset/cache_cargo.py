#!/usr/bin/env python3
"""Fetch and verify test-set Cargo dependencies in a shared host cache."""
import argparse
import concurrent.futures
import datetime as dt
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / 'xpc'
HERE = Path(__file__).resolve().parent
CACHE = ROOT / 'cargo-cache'
PROXY = 'http://127.0.0.1:11321'
DOCKER = ['docker', '-H', 'unix:///var/run/docker.sock']
TARGET = 'x86_64-unknown-linux-gnu'

PREFETCH_SCRIPT = r'''set -euo pipefail
cd /testbed
instance="$RUSTBENCH_INSTANCE_ID"
target="$RUSTBENCH_TARGET"
dirs_file="/cargo-cache/test-manifests/$instance.txt"
lock_dir="/cargo-cache/lockfiles/$instance"
mkdir -p "$lock_dir"
fetch_target=()
if cargo fetch --help 2>&1 | grep -q -- '--target'; then
  fetch_target=(--target "$target")
fi
patch_file="/cargo-cache/test-patches/$instance.patch"
if [[ -s "$patch_file" ]]; then
  patch --batch --fuzz=5 -p1 -i "$patch_file"
fi
declare -A before_hash=()
while IFS= read -r -d '' lockfile; do
  relative="${lockfile#/testbed/}"
  before_hash["$relative"]=$(sha256sum "$lockfile" | cut -d ' ' -f1)
done < <(find /testbed -type f -name Cargo.lock \
  -not -path '*/.git/*' -not -path '*/target/*' -not -path '*/vendor/*' -print0)
mapfile -t workdirs < "$dirs_file"
manifests=()
for workdir in "${workdirs[@]}"; do
  cursor="/testbed/$workdir"
  while [[ "$cursor" == /testbed* ]]; do
    if [[ -f "$cursor/Cargo.toml" ]]; then
      manifests+=("$cursor/Cargo.toml")
      break
    fi
    [[ "$cursor" == /testbed ]] && break
    cursor=$(dirname "$cursor")
  done
done
if (( ${#manifests[@]} == 0 )) && [[ -f /testbed/Cargo.toml ]]; then
  manifests+=(/testbed/Cargo.toml)
fi
if (( ${#manifests[@]} == 0 )); then
  echo 'No Cargo.toml found for the selected test commands' >&2
  exit 20
fi
for manifest in "${manifests[@]}"; do
  directory=$(dirname "$manifest")
  cursor="$directory"
  has_lock=0
  while [[ "$cursor" == /testbed* ]]; do
    if [[ -f "$cursor/Cargo.lock" ]]; then has_lock=1; break; fi
    [[ "$cursor" == /testbed ]] && break
    cursor=$(dirname "$cursor")
  done
  if (( has_lock )); then
    if ! cargo fetch "${fetch_target[@]}" --locked --manifest-path "$manifest"; then
      echo "Existing lock file needs resolution for $manifest; updating it in this disposable container"
      cargo fetch "${fetch_target[@]}" --manifest-path "$manifest"
    fi
  else
    cargo fetch "${fetch_target[@]}" --manifest-path "$manifest"
  fi
done
while IFS= read -r -d '' lockfile; do
  relative="${lockfile#/testbed/}"
  current_hash=$(sha256sum "$lockfile" | cut -d ' ' -f1)
  if [[ "${before_hash[$relative]-}" != "$current_hash" ]]; then
    mkdir -p "$lock_dir/$(dirname "$relative")"
    cp "$lockfile" "$lock_dir/$relative"
  fi
done < <(find /testbed -type f -name Cargo.lock \
  -not -path '*/.git/*' -not -path '*/target/*' -not -path '*/vendor/*' -print0)
cargo --version
'''

VERIFY_SCRIPT = r'''set -euo pipefail
cd /testbed
instance="$RUSTBENCH_INSTANCE_ID"
target="$RUSTBENCH_TARGET"
dirs_file="/cargo-cache/test-manifests/$instance.txt"
lock_dir="/cargo-cache/lockfiles/$instance"
fetch_target=()
if cargo fetch --help 2>&1 | grep -q -- '--target'; then
  fetch_target=(--target "$target")
fi
patch_file="/cargo-cache/test-patches/$instance.patch"
if [[ -s "$patch_file" ]]; then
  patch --batch --fuzz=5 -p1 -i "$patch_file"
fi
if [[ -d "$lock_dir" ]]; then
  while IFS= read -r -d '' cached_lock; do
    relative="${cached_lock#"$lock_dir"/}"
    destination="/testbed/$relative"
    mkdir -p "$(dirname "$destination")"
    cp "$cached_lock" "$destination"
  done < <(find "$lock_dir" -type f -name Cargo.lock -print0)
fi
mapfile -t workdirs < "$dirs_file"
manifests=()
for workdir in "${workdirs[@]}"; do
  cursor="/testbed/$workdir"
  while [[ "$cursor" == /testbed* ]]; do
    if [[ -f "$cursor/Cargo.toml" ]]; then
      manifests+=("$cursor/Cargo.toml")
      break
    fi
    [[ "$cursor" == /testbed ]] && break
    cursor=$(dirname "$cursor")
  done
done
if (( ${#manifests[@]} == 0 )) && [[ -f /testbed/Cargo.toml ]]; then
  manifests+=(/testbed/Cargo.toml)
fi
if (( ${#manifests[@]} == 0 )); then
  echo 'No Cargo.toml found for the selected test commands' >&2
  exit 20
fi
for manifest in "${manifests[@]}"; do
  cargo fetch "${fetch_target[@]}" --frozen --manifest-path "$manifest"
done
cargo --version
'''


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def cargo_test_workdirs(task):
    """Return the project directories in which this task runs Cargo tests."""
    command_file = (ROOT / 'Rust-bench/swebench/harness/save/test_commands' /
                    task['repo'].replace('/', '__') / f"{task['instance_id']}.json")
    if not command_file.is_file():
        return ['.']
    commands = json.loads(command_file.read_text(encoding='utf-8'))
    cwd = '.'
    workdirs = []
    for command in commands:
        if not isinstance(command, str):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        if len(tokens) >= 2 and tokens[0] == 'cd':
            candidate = posixpath.normpath(posixpath.join(cwd, tokens[1]))
            cwd = candidate if candidate != '..' and not candidate.startswith('../') else '.'
        if re.search(r'\bcargo(?:\s+\+[^\s]+)?\s+test\b', command):
            workdirs.append(cwd)
    return list(dict.fromkeys(workdirs or ['.']))


def docker_command(task, script, offline):
    args = [*DOCKER, 'run', '--rm', '--network', 'none' if offline else 'host']
    for directory, target, mode in (
        ('registry/cache', '/root/.cargo/registry/cache', 'rw'),
        ('registry/src', '/root/.cargo/registry/src', 'rw'),
        ('git', '/root/.cargo/git', 'rw'),
        ('lockfiles', '/cargo-cache/lockfiles', 'rw'),
        ('test-patches', '/cargo-cache/test-patches', 'ro'),
        ('test-manifests', '/cargo-cache/test-manifests', 'ro'),
    ):
        args += ['-v', f'{CACHE / directory}:{target}:{mode}']
    args += ['-e', f'RUSTBENCH_INSTANCE_ID={task["instance_id"]}', '-e', f'RUSTBENCH_TARGET={TARGET}']
    if not offline:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'FTP_PROXY',
                     'http_proxy', 'https_proxy', 'all_proxy', 'ftp_proxy'):
            args += ['-e', f'{name}={PROXY}']
        args += ['-e', 'NO_PROXY=localhost,127.0.0.1,::1', '-e', 'no_proxy=localhost,127.0.0.1,::1']
    args += ['-w', '/testbed', '--entrypoint', '/bin/bash', task['image'], '-lc', script]
    return args


def run_stage(task, stage, script, offline):
    log_path = CACHE / 'logs' / f'{task["instance_id"]}.{stage}.log'
    with log_path.open('w', encoding='utf-8') as log:
        result = subprocess.run(docker_command(task, script, offline), stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    return result.returncode, log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2, help='Concurrent task containers (default: 2)')
    parser.add_argument('--force', action='store_true', help='Repeat tasks already marked offline-verified')
    parser.add_argument('--instance', action='append', help='Limit pre-caching to selected task IDs')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')

    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    rows = {row['instance_id']: row for row in load_jsonl(ROOT / 'data/rustbench-all.jsonl')}
    tasks = manifest['tasks']
    if args.instance:
        unknown = set(args.instance) - {task['instance_id'] for task in tasks}
        if unknown:
            parser.error('unknown test-set task: ' + ', '.join(sorted(unknown)))
        tasks = [task for task in tasks if task['instance_id'] in args.instance]
    missing_rows = sorted(task['instance_id'] for task in tasks if task['instance_id'] not in rows)
    if missing_rows:
        raise RuntimeError('Tasks missing from rustbench-all.jsonl: ' + ', '.join(missing_rows))

    for directory in ('registry/cache', 'registry/src', 'git', 'lockfiles', 'test-patches',
                      'test-manifests', 'logs'):
        (CACHE / directory).mkdir(parents=True, exist_ok=True)
    cache_manifest_path = CACHE / 'cache-manifest.json'
    cache_state = (json.loads(cache_manifest_path.read_text(encoding='utf-8'))
                   if cache_manifest_path.exists() else {'version': 1, 'tasks': {}})
    for task in tasks:
        patch_text = rows[task['instance_id']].get('test_patch') or ''
        (CACHE / 'test-patches' / f"{task['instance_id']}.patch").write_text(
            patch_text, encoding='utf-8')
        (CACHE / 'test-manifests' / f"{task['instance_id']}.txt").write_text(
            '\n'.join(cargo_test_workdirs(task)) + '\n', encoding='utf-8')

    proxy_check = ['curl', '-fsS', '-o', os.devnull, '-x', PROXY,
                   'https://index.crates.io/config.json', '--max-time', '15']
    for attempt in range(3):
        if subprocess.run(proxy_check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError('Jifang cannot download crates.io through the configured proxy')
        time.sleep(2)
    subprocess.run([*DOCKER, 'info'], stdout=subprocess.DEVNULL, check=True)
    pending = [task for task in tasks if args.force or cache_state['tasks'].get(task['instance_id'], {}).get('status')
               != 'offline_verified']
    print(f'{len(pending)} tasks to pre-cache; {len(tasks) - len(pending)} already verified', flush=True)

    def process(task):
        for stage, script, offline in (
            ('fetch', PREFETCH_SCRIPT, False),
            ('offline', VERIFY_SCRIPT, True),
        ):
            rc, log_path = run_stage(task, stage, script, offline)
            if rc:
                return {'instance_id': task['instance_id'], 'status': 'failed',
                        'failed_stage': stage, 'log': str(log_path), 'returncode': rc}
        return {'instance_id': task['instance_id'], 'status': 'offline_verified',
                'image': task['image']}

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            cache_state['tasks'][result['instance_id']] = result
            cache_state['updated_at'] = dt.datetime.now().astimezone().isoformat()
            cache_manifest_path.write_text(json.dumps(cache_state, indent=2) + '\n', encoding='utf-8')
            print(f"{result['status']}: {result['instance_id']}" +
                  (f" ({result['log']})" if result['status'] == 'failed' else ''), flush=True)
            if result['status'] != 'offline_verified':
                failed.append(result)

    verified_count = sum(item.get('status') == 'offline_verified'
                         for item in cache_state['tasks'].values())
    print(f"Offline-verified {verified_count} tasks", flush=True)
    if failed:
        return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Cargo cache setup failed: {exc}', file=sys.stderr)
        sys.exit(1)
