# conftest.py — pytest configuration
#
# Script-style tests that execute at module level and call sys.exit()
# are excluded from pytest collection. Run them directly:
#   python tests/test_phase2_integration.py
#   python tests/test_phase3_dual_agent.py
#   python tests/validation/test_decision_log_format.py

collect_ignore_glob = [
    "tests/test_phase2_integration.py",
    "tests/test_phase3_dual_agent.py",
    "tests/test_phase1_fixes.py",
    "tests/validation/test_decision_log_format.py",
    "scripts/testing/*",
]

