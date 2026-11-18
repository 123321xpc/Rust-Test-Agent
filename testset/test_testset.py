import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selector', HERE / 'select_tasks.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class TestSetTests(unittest.TestCase):
    def test_frozen_set_is_disjoint_complete_and_balanced(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        validation = json.loads((HERE.parent / 'validation/manifest.json').read_text(encoding='utf-8'))
        tasks = manifest['tasks']
        ids = {task['instance_id'] for task in tasks}
        images = {task['image'] for task in tasks}
        self.assertEqual((len(tasks), len(ids), len(images)), (50, 50, 50))
        self.assertEqual(Counter(task['size'] for task in tasks), {'small': 25, 'medium': 25})
        self.assertFalse(ids & {task['instance_id'] for task in validation['tasks']})
        self.assertFalse(images & {task['image'] for task in validation['tasks']})
        self.assertTrue({task['instance_id'] for task in validation['tasks']}
                        <= set(manifest['validation_instance_ids']))
        self.assertTrue(all(selector.toolchain_version(task) >= selector.MIN_TOOLCHAIN_VERSION
                            for task in tasks))
        self.assertTrue(all(selector.patch_size(task['patch_changed_lines']) == task['size']
                            for task in tasks))

    def test_frozen_set_matches_explicit_selection(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        original = json.loads((HERE / 'manifest-100.json').read_text(encoding='utf-8'))
        additions = json.loads((HERE / 'manifest-additions.json').read_text(encoding='utf-8'))
        candidates = original['tasks'] + additions['tasks']
        self.assertEqual(len(original['tasks']), 100)
        self.assertEqual(len(additions['tasks']), 3)
        self.assertEqual(manifest['source_image_count'], 113)
        self.assertEqual(manifest['tasks'], selector.select_tasks(candidates))
        self.assertTrue(all(task['size'] == 'small'
                            and selector.toolchain_version(task) >= selector.MIN_TOOLCHAIN_VERSION
                            for task in additions['tasks']))

    def test_selection_rejects_wrong_image_count(self):
        with self.assertRaisesRegex(ValueError, '113 distinct canonical task image tags'):
            selector.build_manifest([], [], {'tasks': []}, {})


if __name__ == '__main__':
    unittest.main()
