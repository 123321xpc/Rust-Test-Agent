import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selector', HERE / 'select_tasks.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class ExperimentSetTests(unittest.TestCase):
    def test_manifest_is_complete_and_balanced(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        tasks = manifest['tasks']
        ids = {task['instance_id'] for task in tasks}
        images = {task['image'] for task in tasks}

        self.assertEqual((len(tasks), len(ids), len(images)), (100, 100, 100))
        self.assertEqual(Counter(task['size'] for task in tasks),
                         {'small': 40, 'medium': 55, 'large': 5})

        testset = json.loads((HERE.parent / 'testset/manifest.json').read_text(encoding='utf-8'))
        ts_ids = {t['instance_id'] for t in testset['tasks']}
        self.assertTrue(ts_ids.issubset(ids), 'All 50 testset tasks must be in experiment set')

        validation = json.loads((HERE.parent / 'validation/manifest.json').read_text(encoding='utf-8'))
        val_small = {t['instance_id'] for t in validation['tasks'] if t['size'] == 'small'}
        self.assertFalse(val_small & ids, 'Validation small tasks must not be in experiment set')

        self.assertTrue(all(selector.patch_size(task['patch_changed_lines']) == task['size']
                            for task in tasks))

    def test_reproducible_selection(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        validation = json.loads((HERE.parent / 'validation/manifest.json').read_text(encoding='utf-8'))
        testset = json.loads((HERE.parent / 'testset/manifest.json').read_text(encoding='utf-8'))

        val_ids = {t['instance_id'] for t in validation['tasks']}
        ts_ids = {t['instance_id'] for t in testset['tasks']}

        reproduced = selector.select_tasks(manifest['tasks'], val_ids, ts_ids)
        self.assertEqual(manifest['tasks'], reproduced)


if __name__ == '__main__':
    unittest.main()
