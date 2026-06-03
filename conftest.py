# conftest.py — pytest configuration
#
# Script-style tests that execute at module level are excluded from pytest
# collection. Run them directly when needed:
#   python tests/validation/test_decision_log_format.py

collect_ignore_glob = [
    "tests/validation/test_decision_log_format.py",
    "scripts/testing/*",
]
