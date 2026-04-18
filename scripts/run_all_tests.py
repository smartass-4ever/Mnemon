"""
run_all_tests.py
================
Runs the full test suite including @pytest.mark.slow tests.

Usage:
    python run_all_tests.py
    python run_all_tests.py -v
"""

import subprocess
import sys
import time

def main():
    args = sys.argv[1:]
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_full_suite.py",
        "--asyncio-mode=auto",
        "--tb=short",
        "-q",
    ] + args

    print("Running full test suite (fast + slow)...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    print(f"\nFull test suite completed in {elapsed:.1f}s")
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
