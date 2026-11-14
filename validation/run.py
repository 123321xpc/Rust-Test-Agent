#!/usr/bin/env python3
"""Run the frozen Jifang validation set against one or more agents."""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path.home() / 'xpc'
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / 'envs/rustbench/bin/python'
SWE_PYTHON = ROOT / 'miniconda3/envs/swe-agent/bin/python'
HARNESS = ROOT / 'Rust-bench'
SWE = ROOT / 'sweagent-run/SWE-agent'
AGENTS = ('swe-agent', 'RustForger', 'normal', 'ACI-impr', 'trace-impr', 'test')
PROXY = 'http://127.0.0.1:11321'
PROXY_NAMES = tuple(name for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'FTP_PROXY')
                    for name in (key, key.lower()))
SET_NAME = 'validation'


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def child_env(network='proxy'):
    env = os.environ.copy()
    from dotenv import dotenv_values
    for key, value in dotenv_values(ROOT / '.env').items():
        if value is not None:
            env.setdefault(key, value)
    env['DOCKER_HOST'] = 'unix:///var/run/docker.sock'
    if network == 'direct':
        # Keep Docker's client-side proxy defaults out of this run.
        direct_config = HERE / 'docker-direct-config'
        direct_config.mkdir(exist_ok=True)
        env['DOCKER_CONFIG'] = str(direct_config)
    else:
        env['DOCKER_CONFIG'] = str(ROOT / 'sweagent-docker-config')
    env['PATH'] = f"{ROOT}/git-2.43.0/bin:{Path.home()}/.cargo/bin:" + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = f"{ROOT}/gcc-12/lib:{ROOT}/gcc-12/x86_64-conda-linux-gnu/lib:" + env.get('LD_LIBRARY_PATH', '')
    for name in (*PROXY_NAMES, 'NETWORK_PROXY'):
        env[name] = PROXY if network == 'proxy' else ''
    env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.134.0/24'
    env['LANGCHAIN_OPENAI_TCP_KEEPALIVE'] = '0'
    if SET_NAME == 'testset':
        env['RUSTBENCH_CARGO_CACHE_ROOT'] = str(ROOT / 'cargo-cache')
    return env


def verify_proxy(env):
    subprocess.run(['curl', '-fsSI', '-x', PROXY, 'https://github.com', '--max-time', '10'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, check=True)


def preflight(tasks, rows, agents, env):
    """Check the precise cached image and the harness's current image resolution."""
    missing = [str(p) for p in (PYTHON, HARNESS, ROOT / '.env') if not p.exists()]
    if 'swe-agent' in agents:
        missing += [str(p) for p in (SWE_PYTHON, SWE / 'run.py', SWE / 'config/jifang-ripgrep.yaml') if not p.exists()]
    if 'RustForger' in agents:
        missing += [str(p) for p in (ROOT / 'RustForger/src/rustforger_main_tracing.py',
                    ROOT / 'RustForger/rustforger-tracer/bin/trace_cli') if not p.exists()]
    missing += [str(ROOT / 'baselines' / a / 'agent.py') for a in agents
                if a in ('normal', 'ACI-impr', 'trace-impr', 'test') and not (ROOT / 'baselines' / a / 'agent.py').exists()]
    if missing:
        raise RuntimeError('Required paths missing: ' + ', '.join(missing))
    if SET_NAME == 'testset':
        cache_manifest = ROOT / 'cargo-cache/cache-manifest.json'
        if not cache_manifest.is_file():
            raise RuntimeError('Shared Cargo cache is not initialized; run testset/cache_cargo.py first')
        cache_state = json.loads(cache_manifest.read_text(encoding='utf-8'))
        cached_ids = {iid for iid, item in cache_state.get('tasks', {}).items()
                      if item.get('status') == 'offline_verified'}
        missing_cache = sorted(t['instance_id'] for t in tasks if t['instance_id'] not in cached_ids)
        if missing_cache:
            raise RuntimeError('Tasks missing offline Cargo cache: ' + ', '.join(missing_cache))
    subprocess.run(['docker', '-H', 'unix:///var/run/docker.sock', 'info'], env=env,
                   stdout=subprocess.DEVNULL, check=True)
    sys.path.insert(0, str(HARNESS))
    from swebench.harness.resolve_image import resolve_instance_image_key
    config = str(HARNESS / 'swebench/harness/logs/config.json')
    for task in tasks:
        iid = task['instance_id']
        row = rows[iid]
        if row['base_commit'] != task['base_commit']:
            raise RuntimeError(f'{iid}: dataset base commit differs from the manifest')
        canonical = resolve_instance_image_key(row, config_path=config)
        if canonical != task['image']:
            raise RuntimeError(f'{iid}: harness image {canonical} differs from frozen {task["image"]}')
        subprocess.run(['docker', '-H', 'unix:///var/run/docker.sock', 'image', 'inspect', task['image']],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if env.get('NETWORK_PROXY'):
        verify_proxy(env)


def docker_base(task, env=None):
    args = ['docker', '-H', 'unix:///var/run/docker.sock', 'run', '--rm', '--network', 'host',
            '--env-file', str(ROOT / '.env')]
    for key in ('MODEL', 'BASE_URL', 'API_KEY', 'GITHUB_TOKEN', 'API_LIMIT'):
        args += ['-e', key]
    values = {'USER': 'root', 'LANGCHAIN_OPENAI_TCP_KEEPALIVE': '0',
              'LD_LIBRARY_PATH': f'{ROOT}/gcc-12/lib:{ROOT}/gcc-12/x86_64-conda-linux-gnu/lib',
              'NO_PROXY': 'localhost,127.0.0.1,::1', 'no_proxy': 'localhost,127.0.0.1,::1'}
    proxy = env.get('NETWORK_PROXY', '') if env is not None else PROXY
    values.update({name: proxy for name in (*PROXY_NAMES, 'NETWORK_PROXY')})
    for key, value in values.items():
        args += ['-e', f'{key}={value}']
    for path in ('envs/rustbench', 'glibc-2.28-runtime', 'gcc-12'):
        args += ['-v', f'{ROOT / path}:{ROOT / path}:ro']
    cache_root = env.get('RUSTBENCH_CARGO_CACHE_ROOT') if env is not None else None
    if cache_root:
        for source, target, mode in (
            ('registry/cache', '/root/.cargo/registry/cache', 'rw'),
            ('registry/src', '/root/.cargo/registry/src', 'rw'),
            ('git', '/root/.cargo/git', 'rw'),
            ('lockfiles', '/cargo-cache/lockfiles', 'ro'),
        ):
            args += ['-v', f'{cache_root}/{source}:{target}:{mode}']
    return args


def run_logged(args, log_path, cwd, env):
    start = time.monotonic()
    with log_path.open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            return result.returncode, round(time.monotonic() - start, 2)
        except OSError as exc:
            print(f'Could not start process: {exc}', file=log)
            return 127, round(time.monotonic() - start, 2)


def run_baseline(agent, task, row, folder, env):
    (folder / 'issue.txt').write_text(row['problem_statement'], encoding='utf-8')
    args = docker_base(task, env)
    agent_path = ROOT / 'baselines' / agent
    args += ['-v', f'{agent_path}:{agent_path}:ro', '-v', f'{folder}:/run/task:rw']
    # Discard original Git history within the disposable task container.
    script = 'rm -rf /testbed/.git /home/"$1"/.git; rm -f /home/fix.patch /home/test.patch; cd /testbed; exec "$2" "$5" --repo /testbed --instance-id "$3" --issue-file /run/task/issue.txt --rust-toolchain "$4" --output /run/task/prediction.jsonl --trace-dir /run/task/traces'
    args += ['--entrypoint', '/bin/bash', task['image'], '-c', script, 'runner',
             task['repo'].split('/')[1], str(PYTHON), task['instance_id'], task['rust_toolchain'], str(agent_path / 'agent.py')]
    rc, seconds = run_logged(args, folder / 'agent.log', ROOT, env)
    pred = folder / 'prediction.jsonl'
    records = read_jsonl(pred) if pred.exists() else []
    patch = records[0].get('model_patch', '') if len(records) == 1 and records[0].get('instance_id') == task['instance_id'] else ''
    return rc, seconds, patch


def run_rustforger(task, row, folder, env):
    for d in ('log', 'data'):
        (folder / d).mkdir()
    # Keep the reference data and gold patches out of the container.
    bug_reports = json.loads((ROOT / 'RustForger/data/rb_all_bug_report.json').read_text(encoding='utf-8'))
    write_json(folder / 'data/rb_all_bug_report.json', {task['instance_id']: bug_reports[task['instance_id']]})
    args = docker_base(task, env)
    for source, dest, mode in [
        (ROOT / 'RustForger/src', '/RTAgent/src', 'ro'),
        (ROOT / 'RustForger/config', '/RTAgent/config', 'ro'),
        (ROOT / 'RustForger/rustforger-tracer', '/RTAgent/rustforger-tracer', 'ro'),
        (folder / 'data', '/RTAgent/data', 'ro'), (folder / 'log', '/RTAgent/log', 'rw')]:
        args += ['-v', f'{source}:{dest}:{mode}']
    script = r'''set -e
repo="$1"; workspace="$2"; iid="$3"; python="$4"
mkdir -p /workspace /root/.cargo/bin /home/rust_target
cp -a /testbed "/workspace/$workspace"
rm -rf "/workspace/$workspace/.git" /testbed/.git "/home/$repo/.git"
rm -f /home/fix.patch /home/test.patch
cd "/workspace/$workspace"
git init -q
git config user.email rustforger@example.com
git config user.name RustForger
git add -A && git commit -qm base --allow-empty
cp /RTAgent/rustforger-tracer/bin/trace_cli /root/.cargo/bin/trace_cli
export PATH="/root/.cargo/bin:$PATH" CARGO_TARGET_DIR=/home/rust_target RUSTFLAGS='-A warnings'
cd /workspace
exec "$python" /RTAgent/src/rustforger_main_tracing.py --instance "$iid"
'''
    workspace = re.sub(r'-\d+$', '', task['instance_id']) + '__0.1'
    args += ['--entrypoint', '/bin/bash', task['image'], '-c', script, 'runner',
             task['repo'].split('/')[1], workspace, task['instance_id'], str(PYTHON)]
    rc, seconds = run_logged(args, folder / 'agent.log', ROOT, env)
    result = folder / 'log' / f'rustforger_log_{task["instance_id"]}.json'
    if not result.exists():
        return rc, seconds, ''
    report = json.loads(result.read_text(encoding='utf-8')).get('task_report') or {}
    diffs = report.get('task_modify_files_diff') or {}
    patch = '\n'.join(v for v in diffs.values() if isinstance(v, str) and v != 'No changes detected by git')
    return rc, seconds, patch


def run_sweagent(task, row, folder, env, suffix):
    org, repo = row['repo'].split('/')
    pr = {'org': org, 'repo': repo, 'number': row['pull_number'], 'state': 'closed',
          'title': 'Issue', 'body': '', 'base': {'label': '', 'ref': '', 'sha': row['base_commit']},
          'resolved_issues': [{'number': str((row.get('issue_numbers') or [row['pull_number']])[0]),
                               'title': 'Issue', 'body': row['problem_statement']}],
          'fix_patch': '', 'test_patch': '', 'instance_id': task['instance_id'], 'image_name': task['image']}
    write_jsonl(folder / 'task.jsonl', [pr])
    args = [str(SWE_PYTHON), 'run.py', '--model_name', 'configured', '--pr_file', str(folder / 'task.jsonl'),
            '--workdir', str(folder / 'work'), '--config_file', 'config/jifang-ripgrep.yaml',
            '--pre_build_all_images', 'False', '--remove_image', 'False', '--skip_existing', 'False',
            '--print_config', 'False', '--max_workers_build_image', '1', '--suffix', suffix]
    rc, seconds = run_logged(args, folder / 'agent.log', SWE, env)
    files = list((SWE / 'trajectories').glob(f'*/*__{suffix}/all_preds.jsonl'))
    if len(files) != 1:
        return rc or 1, seconds, ''
    records = [p for p in read_jsonl(files[0]) if p.get('instance_id') == task['instance_id']]
    return rc, seconds, (records[0].get('prediction') or '') if len(records) == 1 else ''


def evaluate(agent, tasks, folder, run_id, env, workers, timeout):
    predictions = read_jsonl(folder / 'predictions.jsonl')
    if not any(p['model_patch'] for p in predictions):
        return None
    if env.get('NETWORK_PROXY'):
        verify_proxy(env)  # Required immediately before each proxied evaluation.
    command = [str(PYTHON), '-m', 'swebench.harness.run_evaluation',
               '--dataset_name', str(folder / 'dataset.jsonl'), '--predictions_path', str(folder / 'predictions.jsonl'),
               '--instance_ids', *[t['instance_id'] for t in tasks], '--max_workers', str(workers),
               '--timeout', str(timeout), '--cache_level', 'instance', '--clean', 'False',
               '--force_rebuild', 'False', '--run_id', run_id,
               '--config_path', str(HARNESS / 'swebench/harness/logs/config.json'), '--build_image_only', 'False']
    rc, _ = run_logged(command, folder / 'evaluation.log', HARNESS, env)
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', required=True, choices=(*AGENTS, 'all'), help='Agent to run; all runs all registered agents')
    parser.add_argument('--instance', action='append', help=f'Restrict to {SET_NAME} instance ID; repeat for multiple')
    parser.add_argument('--max-workers', type=int, default=2, help='Concurrent evaluation workers (default: 2)')
    parser.add_argument('--timeout', type=int, default=1800, help='Evaluation timeout in seconds (default: 1800)')
    parser.add_argument('--network', choices=('proxy', 'direct'), default='proxy',
                        help='Network mode for agents and evaluation (default: proxy)')
    parser.add_argument('--list', action='store_true', help='Print selection and exit without running')
    parser.add_argument('--dry-run', action='store_true', help='Validate dependencies and images without calling agents')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    tasks = manifest['tasks']
    if args.instance:
        unknown = set(args.instance) - {t['instance_id'] for t in tasks}
        if unknown:
            parser.error(f'not in {SET_NAME} set: ' + ', '.join(sorted(unknown)))
        tasks = [t for t in tasks if t['instance_id'] in args.instance]
    if args.list:
        for t in tasks:
            print(f"{t['size']:6} {t['patch_changed_lines']:4} {t['instance_id']}")
        return 0
    if args.max_workers < 1 or args.timeout < 1:
        parser.error('--max-workers and --timeout must be positive')
    agents = list(AGENTS) if args.agent == 'all' else [args.agent]
    rows = {r['instance_id']: r for r in read_jsonl(ROOT / 'data/rustbench-all.jsonl')}
    env = child_env(args.network)
    preflight(tasks, rows, agents, env)
    if args.dry_run:
        print(f'Preflight passed: {len(tasks)} cached images, agents: {", ".join(agents)}, network: {args.network}')
        return 0
    run_id = SET_NAME + '_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    run_dir = HERE / 'runs' / run_id
    run_dir.mkdir(parents=True)
    write_json(run_dir / 'manifest.json', manifest)
    sources = {'swe-agent': SWE / 'sweagent', 'RustForger': ROOT / 'RustForger/src',
               'normal': ROOT / 'baselines/normal', 'ACI-impr': ROOT / 'baselines/ACI-impr',
               'trace-impr': ROOT / 'baselines/trace-impr', 'test': ROOT / 'baselines/test'}
    hashes = {}
    for agent in agents:
        digest = hashlib.sha256()
        paths = list(sources[agent].rglob('*.py'))
        if agent == 'swe-agent':
            paths += [SWE / 'run.py', SWE / 'config/jifang-ripgrep.yaml']
        elif agent == 'RustForger':
            paths += list((ROOT / 'RustForger/config').rglob('*'))
            paths += [ROOT / 'RustForger/rustforger-tracer/bin/trace_cli']
        for path in sorted(p for p in paths if p.is_file()):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        hashes[agent] = digest.hexdigest()
    image_ids = {}
    for task in tasks:
        image_ids[task['instance_id']] = subprocess.check_output(
            ['docker', '-H', 'unix:///var/run/docker.sock', 'image', 'inspect', '--format', '{{.Id}}', task['image']],
            text=True, env=env).strip()
    write_json(run_dir / 'run_metadata.json', {'model': env.get('MODEL'), 'api_limit': env.get('API_LIMIT'),
               'agent_source_sha256': hashes, 'instances': [t['instance_id'] for t in tasks],
               'image_ids': image_ids, 'dataset_sha256': hashlib.sha256(
                   (ROOT / 'data/rustbench-all.jsonl').read_bytes()).hexdigest(),
               'max_workers': args.max_workers, 'timeout': args.timeout, 'network': args.network})
    print(f'Run directory: {run_dir}', flush=True)
    all_status = {}
    for agent in agents:
        folder = run_dir / agent
        folder.mkdir()
        write_jsonl(folder / 'dataset.jsonl', [rows[t['instance_id']] for t in tasks])
        predictions, statuses = [], []
        for index, task in enumerate(tasks):
            iid = task['instance_id']
            task_dir = folder / iid
            task_dir.mkdir()
            print(f'[{agent} {index + 1}/{len(tasks)}] {iid}', flush=True)
            try:
                if agent == 'swe-agent':
                    rc, seconds, patch = run_sweagent(task, rows[iid], task_dir, env, run_id + '_' + str(index))
                elif agent == 'RustForger':
                    rc, seconds, patch = run_rustforger(task, rows[iid], task_dir, env)
                else:
                    rc, seconds, patch = run_baseline(agent, task, rows[iid], task_dir, env)
                error = None
            except Exception as exc:
                rc, seconds, patch, error = 1, 0, '', f'{type(exc).__name__}: {exc}'
            predictions.append({'instance_id': iid, 'model_name_or_path': agent, 'model_patch': patch})
            statuses.append({'instance_id': iid, 'agent_exit_code': rc, 'seconds': seconds,
                             'patch_bytes': len(patch.encode('utf-8')), 'error': error})
            write_jsonl(folder / 'predictions.jsonl', predictions)
            write_json(folder / 'agent_status.json', statuses)
        eval_id = run_id + '_' + agent.replace('-', '_')
        try:
            eval_rc = evaluate(agent, tasks, folder, eval_id, env, args.max_workers, args.timeout)
            eval_error = None
        except Exception as exc:
            eval_rc, eval_error = 1, f'{type(exc).__name__}: {exc}'
        outcomes = []
        for status, prediction in zip(statuses, predictions):
            iid = status['instance_id']
            report = HARNESS / 'logs/run_evaluation' / eval_id / agent / iid / 'report.json'
            test_output = report.parent / 'test_output.txt'
            if report.exists() and test_output.exists():
                result = json.loads(report.read_text(encoding='utf-8'))[iid]
                verdict = 'error' if result.get('error') else ('resolved' if result.get('resolved') else 'unresolved')
            elif status['agent_exit_code'] != 0 or status['error']:
                verdict = 'agent_error'
            elif not prediction['model_patch']:
                verdict = 'no_patch'
            else:
                verdict = 'evaluation_error'
            outcomes.append({**status, 'verdict': verdict, 'report': str(report) if report.exists() else None})
        counts = {key: sum(o['verdict'] == key for o in outcomes)
                  for key in ('resolved', 'unresolved', 'no_patch', 'agent_error', 'evaluation_error', 'error')}
        all_status[agent] = {'evaluation_exit_code': eval_rc, 'evaluation_error': eval_error,
                             'counts': counts, 'total': len(tasks), 'instances': outcomes}
        write_json(run_dir / 'summary.json', all_status)
        print(f'[{agent}] {counts}; evaluation exit={eval_rc}', flush=True)
    return 1 if any(s['evaluation_exit_code'] not in (None, 0) or
                    any(s['counts'][k] for k in ('agent_error', 'evaluation_error', 'error'))
                    for s in all_status.values()) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Validation failed: {exc}', file=sys.stderr)
        sys.exit(1)
