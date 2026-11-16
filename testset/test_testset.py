import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selector', HERE / 'select_tasks.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class TestSetTests(unittest.TestCase):
    def test_frozen_set_is_disjoint_and_complete(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        validation = json.loads((HERE.parent / 'validation/manifest.json').read_text(encoding='utf-8'))
        tasks = manifest['tasks']
        ids = {task['instance_id'] for task in tasks}
        images = {task['image'] for task in tasks}
        self.assertEqual((len(tasks), len(ids), len(images)), (50, 50, 50))
        self.assertFalse(ids & {task['instance_id'] for task in validation['tasks']})
        self.assertFalse(images & {task['image'] for task in validation['tasks']})
        self.assertTrue({t['instance_id'] for t in validation['tasks']}
                        <= set(manifest['validation_instance_ids']))
        self.assertTrue(all(selector.toolchain_version(task) >= selector.MIN_TOOLCHAIN_VERSION
                            for task in tasks))

    def test_frozen_set_matches_original_selection(self):
        manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
        original = json.loads((HERE / 'manifest-100.json').read_text(encoding='utf-8'))
        self.assertEqual(len(original['tasks']), 100)
        self.assertEqual(manifest['tasks'], selector.select_tasks(original['tasks']))
        self.assertEqual({task['repo'] for task in manifest['tasks']},
                         {task['repo'] for task in original['tasks']})

    def test_selection_rejects_wrong_image_count(self):
        with self.assertRaisesRegex(ValueError, '110 distinct'):
            selector.build_manifest([], [], {'tasks': []}, {})


if __name__ == '__main__':
    unittest.main()
