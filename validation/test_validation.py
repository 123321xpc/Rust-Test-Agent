import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('run')
selector = load('select_tasks')


class ValidationTests(unittest.TestCase):
    def test_network_modes_override_inherited_proxy_and_container_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'HERE', Path(tmp)), \
                patch.dict(os.environ, {'HTTP_PROXY': 'http://inherited:1234', 'ftp_proxy': 'http://inherited:1234'}):
            for mode, expected in (('proxy', runner.PROXY), ('direct', '')):
                env = runner.child_env(mode)
                for key in (*runner.PROXY_NAMES, 'NETWORK_PROXY'):
                    self.assertEqual(env[key], expected, (mode, key))
                args = runner.docker_base({}, env)
                for key in (*runner.PROXY_NAMES, 'NETWORK_PROXY'):
                    self.assertIn(f'{key}={expected}', args)
            default_env = runner.child_env()
            self.assertEqual(default_env['NETWORK_PROXY'], '')
            self.assertEqual(env['DOCKER_CONFIG'], str(Path(tmp) / 'docker-direct-config'))

    def test_direct_evaluation_skips_proxy_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            runner.write_jsonl(folder / 'predictions.jsonl', [{'model_patch': 'patch'}])
            with patch.object(runner, 'verify_proxy') as proxy, \
                    patch.object(runner, 'run_logged', return_value=(0, 1)):
                self.assertEqual(runner.evaluate('test', [], folder, 'run', {'NETWORK_PROXY': ''}, 1, 100), 0)
                proxy.assert_not_called()
                self.assertEqual(runner.evaluate('test', [], folder, 'run', {'NETWORK_PROXY': runner.PROXY}, 1, 100), 0)
                proxy.assert_called_once()

    def test_changed_lines_count_only_hunks(self):
        diff = ('diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1,2 +1,2 @@\n'
                '-old\n+++valid added content\n context\n\\ No newline at end of file\n')
        self.assertEqual(selector.lines_changed(diff), 2)

    def test_manifest_has_fixed_strata_and_distinct_repositories(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        tasks = manifest['tasks']
        self.assertEqual(len(tasks), 5)
        self.assertEqual(len({t['instance_id'] for t in tasks}), 5)
        self.assertEqual(len({t['repo'] for t in tasks}), 5)
        self.assertEqual([sum(t['size'] == s for t in tasks) for s in ('large', 'medium', 'small')], [0, 2, 3])
        self.assertTrue(all(t['instance_id'] not in manifest['excluded_prior_report_ids'] for t in tasks))

    def test_sweagent_task_hides_reference_patches_and_reads_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            swe = folder / 'swe'
            output = swe / 'trajectories' / 'user' / 'name__unique' / 'all_preds.jsonl'
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({'instance_id': 'org__repo-1', 'prediction': 'agent diff'}) + '\n')
            task = {'instance_id': 'org__repo-1', 'image': 'example'}
            row = {'repo': 'org/repo', 'pull_number': 1, 'base_commit': 'abc',
                   'problem_statement': 'issue', 'patch': 'SECRET GOLD', 'test_patch': 'SECRET TEST'}
            with patch.object(runner, 'SWE', swe), patch.object(runner, 'run_logged', return_value=(0, 1)):
                rc, _, result = runner.run_sweagent(task, row, folder, {}, 'unique')
            self.assertEqual((rc, result), (0, 'agent diff'))
            text = (folder / 'task.jsonl').read_text()
            self.assertNotIn('SECRET GOLD', text)
            self.assertNotIn('SECRET TEST', text)

    def test_no_patch_skips_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            runner.write_jsonl(folder / 'predictions.jsonl', [{'model_patch': ''}])
            with patch.object(runner, 'verify_proxy') as proxy:
                self.assertIsNone(runner.evaluate('test', [], folder, 'run', {}, 1, 100))
                proxy.assert_not_called()



    def test_analyze_handoffs_correlates_snapshots_and_legacy_traces(self):
        analyzer = load('analyze_handoffs')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / 'validation' / 'runs' / 'validation_test_run'
            logs_dir = root / 'Rust-bench' / 'logs' / 'run_evaluation'
            eval_dir = logs_dir / f'{run_dir.name}_test' / 'test'
            (run_dir / 'test' / 'inst__a-1' / 'traces').mkdir(parents=True)
            (run_dir / 'test' / 'inst__b-2' / 'traces').mkdir(parents=True)
            (run_dir / 'test' / 'inst__c-3' / 'traces').mkdir(parents=True)
            (eval_dir / 'inst__a-1').mkdir(parents=True)
            (eval_dir / 'inst__b-2').mkdir(parents=True)

            (eval_dir / 'inst__a-1' / 'report.json').write_text(json.dumps({
                'inst__a-1': {
                    'patch_successfully_applied': True,
                    'resolved': True,
                    'tests_status': {'FAIL_TO_PASS': {'success': ['t1'], 'failure': []}, 'PASS_TO_PASS': {'success': ['p1'], 'failure': []}},
                }
            }), encoding='utf-8')
            (eval_dir / 'inst__b-2' / 'report.json').write_text(json.dumps({
                'inst__b-2': {
                    'patch_successfully_applied': True,
                    'resolved': False,
                    'tests_status': {'FAIL_TO_PASS': {'success': [], 'failure': ['t2']}, 'PASS_TO_PASS': {'success': [], 'failure': ['src/lib.rs - foo (line 45)']}},
                }
            }), encoding='utf-8')

            # Instance A: P0 handoff snapshots, resolved
            t_a = [
                {'event': 'multi_agent_start', 'issue_contract': {'requested_symbols': ['foo'], 'new_api_candidates': ['foo'], 'referenced_files': ['src/lib.rs'], 'expected_result_excerpt': 'returns 1', 'signals': {'mentions_new_api': True}}},
                {'event': 'handoff_feature_snapshot', 'stage': 'issue_contract', 'from_role': 'issue', 'to_role': 'navigator', 'attempt': 1, 'features': {'new_api_candidates': {'present': True, 'count': 1, 'items': ['foo']}, 'expected_result_excerpt': {'present': True}, 'referenced_files': {'present': True, 'count': 1}, 'requested_symbols': {'present': True, 'count': 1}}},
                {'event': 'handoff_feature_snapshot', 'stage': 'navigator_handoff', 'from_role': 'navigator', 'to_role': 'fixer', 'attempt': 1, 'route': 'fast', 'outcome': 'completed', 'auto_synthesized': False, 'features': {'candidate_chains': {'chain_count': 1, 'root_candidates': [{'path': 'src/lib.rs'}]}, 'reproduction_status': {'status': 'runtime_failure', 'clean_red_count': 1}, 'repro_test_contents': {'char_len': 120}, 'shared_read_excerpts': {'char_len': 400}, 'shared_ckg_notes': {'char_len': 200}}},
                {'event': 'handoff_feature_snapshot', 'stage': 'fixer_handoff', 'from_role': 'fixer', 'to_role': 'qa', 'attempt': 1, 'outcome': 'completed', 'fix_prompt_breakdown': {'total_chars': 2200, 'issue_chars': 300}, 'features': {'changed_files': {'paths': ['src/lib.rs']}, 'regression_verification': {'verified': True}, 'stuck_status': {'compile_repeats': 0, 'test_repeats': 0}, 'env_blocker': {'value': False}, 'deterministic_qa': {'present': True, 'ok': True, 'missing_contract_apis': [], 'doctest_shifts': []}}},
                {'event': 'handoff_feature_snapshot', 'stage': 'verifier_handoff', 'from_role': 'verifier', 'to_role': 'submission', 'attempt': 1, 'decision': 'APPROVE', 'ver_prompt_chars': 1500, 'features': {'missing_requirements': {'items': []}, 'objections': {'items': []}}},
                {'event': 'coordination_metrics', 'milestones_achieved': 5, 'milestones_total': 5, 'overall_kpi': 0.25, 'duplicate_work_rate': 0.0, 'zero_delta_handoffs': 0, 'pruned_replans': 0, 'coordination_efficiency_per_call': 0.4167, 'coordination_efficiency_per_100k_tokens': 10.0},
                {'event': 'agent_metrics', 'api_calls': 12, 'total_tokens': 50000},
            ]
            (run_dir / 'test' / 'inst__a-1' / 'traces' / 'a.jsonl').write_text('\n'.join(json.dumps(x) for x in t_a) + '\n', encoding='utf-8')

            # Instance B: legacy trace without snapshots, doctest regression
            t_b = [
                {'event': 'multi_agent_start', 'issue_contract': {'requested_symbols': [], 'new_api_candidates': [], 'referenced_files': [], 'expected_result_excerpt': '', 'signals': {}}},
                {'event': 'adaptive_route', 'route': 'planned', 'reasons': ['multi-file']},
                {'event': 'escalation', 'from_role': 'fixer', 'to_role': 'suggester', 'reason': 'stuck', 'attempt': 1},
                {'event': 'qa_gate', 'deterministic_ok': True, 'llm_decision': 'APPROVE', 'qa_round': 1},
                {'event': 'agent_metrics', 'api_calls': 40, 'total_tokens': 180000},
            ]
            (run_dir / 'test' / 'inst__b-2' / 'traces' / 'b.jsonl').write_text('\n'.join(json.dumps(x) for x in t_b) + '\n', encoding='utf-8')

            # Instance C: agent_error with no production edits
            t_c = [
                {'event': 'multi_agent_start', 'issue_contract': {'signals': {}}},
                {'event': 'run_error', 'stage': 'agent', 'error': 'Agent failed deterministic QA verification: No production source changes were made.', 'error_type': 'RuntimeError'},
                {'event': 'agent_metrics', 'api_calls': 70, 'total_tokens': 300000},
            ]
            (run_dir / 'test' / 'inst__c-3' / 'traces' / 'c.jsonl').write_text('\n'.join(json.dumps(x) for x in t_c) + '\n', encoding='utf-8')

            (run_dir / 'summary.json').write_text(json.dumps({
                'test': {
                    'instances': [
                        {'instance_id': 'inst__a-1', 'verdict': 'resolved', 'seconds': 40, 'patch_bytes': 200, 'report': '/remote/Rust-bench/logs/run_evaluation/' + run_dir.name + '_test/test/inst__a-1/report.json'},
                        {'instance_id': 'inst__b-2', 'verdict': 'unresolved', 'seconds': 120, 'patch_bytes': 400, 'report': '/remote/Rust-bench/logs/run_evaluation/' + run_dir.name + '_test/test/inst__b-2/report.json'},
                        {'instance_id': 'inst__c-3', 'verdict': 'agent_error', 'seconds': 200, 'patch_bytes': 0, 'report': None},
                    ]
                }
            }), encoding='utf-8')

            rep = analyzer.analyze_runs([run_dir], agent='test', harness_logs=logs_dir)
            self.assertEqual(rep['total_instances'], 3)
            self.assertEqual(rep['snapshot_coverage'], 1)
            self.assertEqual(rep['verdict_counts'], {'resolved': 1, 'unresolved': 1, 'agent_error': 1})
            by_id = {r['instance_id']: r for r in rep['instances']}
            self.assertEqual(by_id['inst__a-1']['failure_attribution'], 'resolved')
            self.assertEqual(by_id['inst__b-2']['failure_attribution'], 'doctest_or_pass_to_pass_regression')
            self.assertEqual(by_id['inst__c-3']['failure_attribution'], 'no_source_edits')
            self.assertEqual(rep['prompt_char_averages']['fix_prompt_breakdown']['total_chars'], 2200)
            self.assertEqual(rep['coordination_summary']['avg_overall_kpi'], 0.25)
            md = analyzer.format_markdown_report(rep)
            self.assertIn('inst__a-1', md)
            self.assertIn('doctest_or_pass_to_pass_regression', md)
            self.assertIn('Multi-Agent Coordination Metrics', md)



    def test_run_intervention_compare_outcomes_and_report(self):
        intervener = load('run_intervention')
        b_summary = {'instance_id': 'tokio-rs__bytes-377', 'verdict': 'unresolved', 'api_calls': 80, 'total_tokens': 1296627}
        i_summary = {'instance_id': 'tokio-rs__bytes-377', 'verdict': 'resolved', 'api_calls': 45, 'total_tokens': 750000, 'intervention': {'feature': 'arbitration_rationale', 'action': 'override'}}
        cmp = intervener.compare_outcomes(b_summary, i_summary)
        self.assertTrue(cmp['flipped'])
        self.assertTrue(cmp['positive_flip'])
        self.assertEqual(cmp['api_calls_diff'], -35)
        self.assertEqual(cmp['tokens_diff'], -546627)

        md = intervener.format_intervention_report([cmp])
        self.assertIn('tokio-rs__bytes-377', md)
        self.assertIn('▲ WIN', md)
        self.assertIn('arbitration_rationale', md)

    def test_resolve_api_env_profiles_and_model_defaults(self):
        base = {
            'YJ_BASE_URL': 'https://yj.example/v1',
            'YJ_API_KEY': 'yj-key',
            'YJ_MODEL': 'yj-default',
            'DS_BASE_URL': 'https://ds.example/v1',
            'DS_API_KEY': 'ds-key',
            'DS_MODEL': 'ds-default',
        }
        env = dict(base)
        self.assertEqual(runner.resolve_api_env(env), 'yj')
        self.assertEqual((env['BASE_URL'], env['API_KEY'], env['MODEL']),
                         ('https://yj.example/v1', 'yj-key', 'yj-default'))

        env = dict(base, BASE_URL='https://stale', API_KEY='stale', MODEL='stale')
        self.assertEqual(runner.resolve_api_env(env, api='DS'), 'ds')
        self.assertEqual((env['BASE_URL'], env['API_KEY'], env['MODEL']),
                         ('https://ds.example/v1', 'ds-key', 'ds-default'))

        env = dict(base)
        self.assertEqual(runner.resolve_api_env(env, api='ds', model='custom-ds'), 'ds')
        self.assertEqual((env['BASE_URL'], env['API_KEY'], env['MODEL']),
                         ('https://ds.example/v1', 'ds-key', 'custom-ds'))

        env = dict(base)
        self.assertEqual(runner.resolve_api_env(env, model='custom-default'), 'yj')
        self.assertEqual((env['BASE_URL'], env['API_KEY'], env['MODEL']),
                         ('https://yj.example/v1', 'yj-key', 'custom-default'))

        with self.assertRaises(ValueError):
            runner.resolve_api_env(dict(base), model='   ')
        with self.assertRaises(ValueError):
            runner.resolve_api_env({'DS_BASE_URL': 'https://ds.example/v1'}, api='ds')

    def test_resolve_api_limit_defaults_and_overrides(self):
        env = {}
        self.assertEqual(runner.resolve_api_limit(env), 70)
        self.assertEqual(env['API_LIMIT'], '70')

        env = {'API_LIMIT': '200'}
        self.assertEqual(runner.resolve_api_limit(env), 200)
        self.assertEqual(env['API_LIMIT'], '200')

        env = {'API_LIMIT': '200'}
        self.assertEqual(runner.resolve_api_limit(env, api_limit=35), 35)
        self.assertEqual(env['API_LIMIT'], '35')

        for bad in (0, -5, True):
            with self.assertRaises(ValueError):
                runner.resolve_api_limit({}, api_limit=bad)
        for bad_env in ('', '0', '-1', 'abc', '1.5'):
            with self.assertRaises(ValueError):
                runner.resolve_api_limit({'API_LIMIT': bad_env})

    def test_cli_runners_accept_api_limit_and_default_to_yj(self):
        base_env = {
            'YJ_BASE_URL': 'https://yj.example/v1',
            'YJ_API_KEY': 'yj-key',
            'YJ_MODEL': 'yj-default',
            'DS_BASE_URL': 'https://ds.example/v1',
            'DS_API_KEY': 'ds-key',
            'DS_MODEL': 'ds-default',
            'API_LIMIT': '200',
        }
        # validation/run.py --dry-run defaults to yj and honors --api-limit
        env = dict(base_env)
        buf = io.StringIO()
        with patch.object(runner, 'child_env', return_value=env), \
                patch.object(runner, 'read_jsonl', return_value=[]), \
                patch.object(runner, 'preflight'), \
                patch.object(sys, 'argv', ['run.py', '--agent', 'normal', '--api-limit', '45', '--dry-run']), \
                redirect_stdout(buf):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(env['API_LIMIT'], '45')
        self.assertEqual(env['MODEL'], 'yj-default')
        self.assertIn('api: yj', buf.getvalue())
        self.assertIn('api_limit: 45', buf.getvalue())
        self.assertIn('network: direct', buf.getvalue())

        # validation/run.py accepts --network proxy override and background/foreground flags
        env = dict(base_env)
        buf = io.StringIO()
        with patch.object(runner, 'child_env', return_value=env), \
                patch.object(runner, 'read_jsonl', return_value=[]), \
                patch.object(runner, 'preflight'), \
                patch.object(sys, 'argv', ['run.py', '--agent', 'normal', '--network', 'proxy', '--foreground', '--dry-run']), \
                redirect_stdout(buf):
            self.assertEqual(runner.main(), 0)
        self.assertIn('network: proxy', buf.getvalue())

        # testset/run.py delegates to validation/run.py with the same CLI flags
        testset_runner = load('../testset/run')
        env = dict(base_env)
        buf = io.StringIO()
        with patch.object(testset_runner.runner, 'child_env', return_value=env), \
                patch.object(testset_runner.runner, 'read_jsonl', return_value=[]), \
                patch.object(testset_runner.runner, 'preflight'), \
                patch.object(sys, 'argv', ['run.py', '--agent', 'normal', '--api-limit', '55', '--background', '--dry-run']), \
                redirect_stdout(buf):
            self.assertEqual(testset_runner.runner.main(), 0)
        self.assertEqual(env['API_LIMIT'], '55')
        self.assertIn('api: yj', buf.getvalue())
        self.assertIn('network: direct', buf.getvalue())

        # run_intervention.py defaults to yj and passes --api-limit to run_intervention_task
        intervener = load('run_intervention')
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ckpt = tmp_dir / 'ckpt'
            ckpt.mkdir()
            (ckpt / 'meta.json').write_text(json.dumps({'instance_id': 'tokio-rs__bytes-377'}), encoding='utf-8')
            issue = tmp_dir / 'issue.txt'
            issue.write_text('issue', encoding='utf-8')
            captured = {}

            def fake_run_task(**kwargs):
                captured.update(kwargs)
                return {'ok': True}

            env_with_home = dict(base_env, HOME=str(tmp_dir), USERPROFILE=str(tmp_dir))
            with patch.dict(os.environ, env_with_home, clear=True), \
                    patch('dotenv.load_dotenv'), \
                    patch.object(intervener, 'run_intervention_task', side_effect=fake_run_task):
                rc = intervener.main([
                    '--checkpoint', str(ckpt),
                    '--repo', str(tmp_dir),
                    '--issue-file', str(issue),
                    '--output-dir', str(tmp_dir / 'out'),
                    '--api-limit', '33',
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(captured['max_api_calls'], 33)
            self.assertEqual(captured['model'], 'yj-default')
            self.assertEqual(captured['api_key'], 'yj-key')
            self.assertEqual(captured['base_url'], 'https://yj.example/v1')


if __name__ == '__main__':
    unittest.main()
