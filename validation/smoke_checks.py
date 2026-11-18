"""No-model smoke checks for Jifang container adapters and evaluation."""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import run
import select_tasks

manifest = json.loads((run.HERE / 'manifest.json').read_text(encoding='utf-8'))
task = next(t for t in manifest['tasks'] if t['instance_id'] == 'chronotope__chrono-991')
rows = {r['instance_id']: r for r in run.read_jsonl(run.ROOT / 'data/rustbench-all.jsonl')}
row = rows[task['instance_id']]
env = run.child_env()
folder = run.HERE / 'checks' / (sys.argv[1] + '-' + uuid.uuid4().hex[:8])
folder.mkdir(parents=True)
print('Check directory:', folder, flush=True)

if sys.argv[1] == 'sizes':
    for item in manifest['tasks']:
        actual = select_tasks.lines_changed(rows[item['instance_id']]['patch'])
        assert actual == item['patch_changed_lines'], (item['instance_id'], actual)
    print('All frozen patch sizes agree with dataset')
elif sys.argv[1] == 'imports':
    for name in ('normal', 'ACI-impr', 'test'):
        source = run.ROOT / 'baselines' / name
        cmd = run.docker_base(task) + ['-v', f'{source}:{source}:ro', '--entrypoint', str(run.PYTHON),
                                      task['image'], str(source / 'agent.py'), '--help']
        code, _ = run.run_logged(cmd, folder / (name + '.log'), run.ROOT, env)
        print(name, code, flush=True)
        if code:
            raise SystemExit(code)
    for source, dest in [(run.ROOT / 'RustForger/src', '/RTAgent/src'),
                         (run.ROOT / 'RustForger/config', '/RTAgent/config')]:
        assert source.exists()
    cmd = run.docker_base(task) + ['-v', f'{run.ROOT / "RustForger/src"}:/RTAgent/src:ro',
                                  '--entrypoint', str(run.PYTHON), task['image'],
                                  '/RTAgent/src/rustforger_main_tracing.py', '--help']
    code, _ = run.run_logged(cmd, folder / 'RustForger.log', run.ROOT, env)
    print('RustForger', code, flush=True)
    if code:
        raise SystemExit(code)
elif sys.argv[1] == 'swe':
    original = run.run_logged
    def instant(args, log, cwd, environment):
        args = list(args)
        args[args.index('configured')] = 'instant_empty_submit'
        return original(args, log, cwd, environment)
    run.run_logged = instant
    rc, seconds, prediction = run.run_sweagent(task, row, folder, env, 'smoke_' + uuid.uuid4().hex[:8])
    print('SWE-agent exit:', rc, 'seconds:', seconds, 'empty prediction:', not prediction, flush=True)
    if rc:
        raise SystemExit(rc)
elif sys.argv[1] == 'gold':
    run.write_jsonl(folder / 'dataset.jsonl', [row])
    run.write_jsonl(folder / 'predictions.jsonl', [{'instance_id': task['instance_id'],
                    'model_name_or_path': 'validation_check', 'model_patch': row['patch']}])
    eval_id = 'validation_smoke_gold_' + uuid.uuid4().hex[:8]
    rc = run.evaluate('validation_check', [task], folder, eval_id, env, 1, 1800)
    report = run.HARNESS / 'logs/run_evaluation' / eval_id / 'validation_check' / task['instance_id'] / 'report.json'
    output = report.parent / 'test_output.txt'
    print('Evaluation exit:', rc, 'report exists:', report.exists(), 'test output exists:', output.exists(), flush=True)
    if report.exists():
        result = json.loads(report.read_text())[task['instance_id']]
        print('Resolved:', result.get('resolved'), 'error:', result.get('error'), flush=True)
    if rc != 0 or not report.exists() or not output.exists():
        raise SystemExit(1)
else:
    raise SystemExit('Expected sizes, imports, swe, or gold')
