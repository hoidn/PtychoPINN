"""Tests for the generate_test_index automation script."""

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class GenerateTestIndexTests(unittest.TestCase):
    """Validate metadata extraction helpers for the test index."""

    def test_get_module_docstring_handles_missing_docstring(self):
        """Files without docstrings should return a helpful placeholder."""
        module = REPO_ROOT / "tests" / "test_integration_workflow.py"
        from scripts.tools.generate_test_index import get_module_docstring

        doc = get_module_docstring(module)
        self.assertIn("No module docstring found", doc)

    def test_get_module_docstring_reads_existing_docstring(self):
        """Modules with docstrings should surface their descriptive text."""
        module = REPO_ROOT / "tests" / "test_raw_data_grouping.py"
        from scripts.tools.generate_test_index import get_module_docstring

        doc = get_module_docstring(module)
        self.assertIn("efficient coordinate grouping", doc)

    def test_get_test_functions_lists_key_tests(self):
        """Key tests should be enumerated from the module AST."""
        module = REPO_ROOT / "tests" / "image" / "test_registration.py"
        from scripts.tools.generate_test_index import get_test_functions

        test_names = get_test_functions(module)
        self.assertIn("`test_find_offset_known_shift_real`", test_names)
        self.assertIn("`test_apply_shift_and_crop_basic`", test_names)


if __name__ == "__main__":
    unittest.main()
