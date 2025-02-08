import os
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

# Import the director module so we can call main()
from specs.director import director


class TestDirectorCLI(unittest.TestCase):
    def setUp(self):
        # Create a minimal temporary config file with a prompt template
        self.temp_config = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml")
        # Create a dummy editable file (must exist)
        self.temp_editable = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
        self.temp_editable.write("Dummy editable content")
        self.temp_editable.flush()
        self.temp_editable.close()

        config_content = f"""
prompt: "Task is: {{ task }}"
coder_model: o3-mini
evaluator_model: o3-mini
max_iterations: 1
execution_command: "echo 'Hello'"
context_editable: ["{self.temp_editable.name}"]
context_read_only: []
evaluator: default
"""
        self.temp_config.write(config_content)
        self.temp_config.flush()
        self.temp_config.close()

        # Create a temporary file for file-based --task testing
        self.temp_task_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
        self.temp_task_file.write("Task from file")
        self.temp_task_file.flush()
        self.temp_task_file.close()

    def tearDown(self):
        os.remove(self.temp_config.name)
        os.remove(self.temp_editable.name)
        os.remove(self.temp_task_file.name)

    def test_main_raw_task(self):
        # Simulate CLI: --config <config> --task "raw task string"
        test_args = [
            "director.py",
            "--config", self.temp_config.name,
            "--task", "This is a raw task string."
        ]
        # Patch Director.direct so that we do not actually run the iteration loop;
        # instead, print the rendered prompt for inspection.
        with patch.object(sys, "argv", test_args), \
             patch("specs.director.director.Director.direct",
                   lambda self: print("Director.direct() called. Rendered prompt:", self.config.prompt)):
            captured_stdout = StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_stdout
            try:
                try:
                    director.main()
                except SystemExit:
                    # main might call sys.exit(); we ignore it here.
                    pass
            finally:
                sys.stdout = original_stdout

            output = captured_stdout.getvalue()
            self.assertIn("This is a raw task string.", output, "Raw task string was not substituted into the prompt.")

    def test_main_file_task(self):
        # Simulate CLI: --config <config> --task <task-file-path>
        test_args = [
            "director.py",
            "--config", self.temp_config.name,
            "--task", self.temp_task_file.name
        ]
        with patch.object(sys, "argv", test_args), \
             patch("specs.director.director.Director.direct",
                   lambda self: print("Director.direct() called. Rendered prompt:", self.config.prompt)):
            captured_stdout = StringIO()
            original_stdout = sys.stdout
            sys.stdout = captured_stdout
            try:
                try:
                    director.main()
                except SystemExit:
                    pass
            finally:
                sys.stdout = original_stdout

            output = captured_stdout.getvalue()
            self.assertIn("Task from file", output, "File task content was not substituted into the prompt.")

if __name__ == "__main__":
    unittest.main()
