from pathlib import Path
import unittest
from unittest.mock import Mock

from sweagent.environment.swe_env import SWEEnv


PATCH = "diff --git a/example.rs b/example.rs\n--- a/example.rs\n+++ b/example.rs\n@@ -1 +1 @@\n-old\n+new\n"


def submission_output(patch):
    return f"<<SUBMISSION||\n{patch}||SUBMISSION>>\n"


class TestSubmission(unittest.TestCase):
    def setUp(self):
        # Exercise the real step logic without starting a Docker container.
        self.env = SWEEnv.__new__(SWEEnv)
        self.env.logger = Mock()
        self.env.communicate = Mock()

    def test_reading_command_source_does_not_submit(self):
        source = Path(__file__).parents[1] / "config/commands/defaults.sh"
        output = source.read_text()
        self.env.communicate.return_value = output
        observation, _, done, info = self.env.step("cat /root/commands/defaults.sh")
        self.assertEqual(observation, output)
        self.assertFalse(done)
        self.assertEqual(info, {})

    def test_regular_command_with_submission_markers_does_not_submit(self):
        self.env.communicate.return_value = submission_output(PATCH)
        _, _, done, info = self.env.step("cat saved-output.txt")
        self.assertFalse(done)
        self.assertNotIn("submission", info)

    def test_submit_collects_patch(self):
        self.env.communicate.return_value = submission_output(PATCH)
        observation, _, done, info = self.env.step(" submit\n")
        self.assertTrue(done)
        self.assertEqual(info["exit_status"], "submitted")
        self.assertEqual(info["submission"], PATCH)
        self.assertEqual(observation, PATCH)

    def test_autosubmit_still_collects_patch(self):
        self.env.communicate.return_value = submission_output(PATCH)
        _, _, done, info = self.env.step("exit_format")
        self.assertTrue(done)
        self.assertEqual(info["exit_status"], "submitted (exit_format)")
        self.assertEqual(info["submission"], PATCH)
        self.assertEqual(self.env.communicate.call_args.kwargs["input"], "submit")

    def test_long_submission_is_not_truncated(self):
        patch = PATCH + "+added line\n" * 5000
        self.env.communicate.return_value = submission_output(patch)
        _, _, done, info = self.env.step("submit")
        self.assertTrue(done)
        self.assertEqual(info["submission"], patch)

    def test_script_source_is_not_a_submission_envelope(self):
        source = Path(__file__).parents[1] / "config/commands/defaults.sh"
        self.assertIsNone(self.env.get_submission(source.read_text()))

    def test_empty_submission_preserves_existing_behavior(self):
        self.env.communicate.return_value = submission_output("")
        observation, _, done, info = self.env.step("submit")
        self.assertTrue(done)
        self.assertIsNone(info["submission"])
        self.assertIsNone(observation)



    def test_is_test_file_path(self):
        from sweagent.environment.swe_env import is_test_file_path
        self.assertTrue(is_test_file_path("tests/test_bytes.rs"))
        self.assertTrue(is_test_file_path("crates/foo/tests/bar.rs"))
        self.assertTrue(is_test_file_path("src/datetime/tests.rs"))
        self.assertTrue(is_test_file_path("src/lower/test.rs"))
        self.assertTrue(is_test_file_path("src/lib/types_test.rs"))
        self.assertTrue(is_test_file_path("src/lib/test_command.rs"))
        self.assertTrue(is_test_file_path("crates/parser/test_data/file.rast"))

        self.assertFalse(is_test_file_path("src/bytes.rs"))
        self.assertFalse(is_test_file_path("src/lib.rs"))
        self.assertFalse(is_test_file_path("reproduce.rs"))
        self.assertFalse(is_test_file_path("Cargo.toml"))
        self.assertFalse(is_test_file_path(""))

    def test_filter_test_files_from_patch(self):
        from sweagent.environment.swe_env import filter_test_files_from_patch
        mixed_patch = (
            "diff --git a/src/lib.rs b/src/lib.rs\n"
            "index 111..222 100644\n"
            "--- a/src/lib.rs\n"
            "+++ b/src/lib.rs\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs\n"
            "index 333..444 100644\n"
            "--- a/tests/test_bytes.rs\n"
            "+++ b/tests/test_bytes.rs\n"
            "@@ -5 +5 @@\n"
            "-assert(0)\n"
            "+assert(1)\n"
            "diff --git a/src/bytes.rs b/src/bytes.rs\n"
            "--- a/src/bytes.rs\n"
            "+++ b/src/bytes.rs\n"
            "@@ -10 +10 @@\n"
            "-x\n"
            "+y\n"
        )
        filtered = filter_test_files_from_patch(mixed_patch)
        self.assertIn("src/lib.rs", filtered)
        self.assertIn("src/bytes.rs", filtered)
        self.assertNotIn("tests/test_bytes.rs", filtered)

        # Test-only patch stripped to empty
        test_only = (
            "diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs\n"
            "--- a/tests/test_bytes.rs\n"
            "+++ b/tests/test_bytes.rs\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        self.assertEqual(filter_test_files_from_patch(test_only), "")

    def test_submission_filters_test_files_when_enabled(self):
        mixed = (
            "diff --git a/src/lib.rs b/src/lib.rs\n"
            "--- a/src/lib.rs\n"
            "+++ b/src/lib.rs\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs\n"
            "--- a/tests/test_bytes.rs\n"
            "+++ b/tests/test_bytes.rs\n"
            "@@ -1 +1 @@\n"
            "-test_old\n"
            "+test_new\n"
        )
        self.env.communicate.return_value = submission_output(mixed)
        self.env.protect_test_files = True
        obs, _, done, info = self.env.step("submit")
        self.assertTrue(done)
        self.assertIn("src/lib.rs", info["submission"])
        self.assertNotIn("tests/test_bytes.rs", info["submission"])

    def test_enforce_no_test_file_edits_reverts_changes(self):
        self.env.protect_test_files = True
        self.env.container_obj = Mock()
        self.env._repo_root = "/testbed"
        # Return git status showing test modifications and non-test modification
        self.env.communicate.side_effect = [
            " M tests/test_bytes.rs\n?? tests/bar.rs\n M src/lib.rs\n",
            "",  # revert command result
        ]
        reverted = self.env._enforce_no_test_file_edits("cargo test")
        self.assertEqual(reverted, ["tests/bar.rs", "tests/test_bytes.rs"])
        # Check that the revert command executed reset and checkout and rm
        revert_cmd = self.env.communicate.call_args_list[1].kwargs["input"]
        self.assertIn("git -C /testbed reset -q HEAD -- tests/test_bytes.rs", revert_cmd)
        self.assertIn("git -C /testbed checkout -q -- tests/test_bytes.rs", revert_cmd)
        self.assertIn("rm -rf -- /testbed/tests/bar.rs", revert_cmd)

    def test_step_appends_warning_when_test_file_reverted(self):
        self.env.protect_test_files = True
        self.env.container_obj = Mock()
        self.env._repo_root = "/testbed"
        # First call is the action command output, second is status check, third is revert
        self.env.communicate.side_effect = [
            "Running tests...\ntest failed",
            " M tests/test_bytes.rs\n",
            "",
        ]
        obs, _, done, info = self.env.step("cargo test")
        self.assertFalse(done)
        self.assertIn("[TEST PROTECTION ERROR]", obs)
        self.assertIn("- tests/test_bytes.rs", obs)


if __name__ == "__main__":
    unittest.main()
