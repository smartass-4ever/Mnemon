"""
run_fast_tests.py
=================
Runs only the fast test suite (no @pytest.mark.slow tests).
All databases use :memory: — no file creation, no Windows file-lock issues.
Target: completes in under 60 seconds.

Usage:
    python run_fast_tests.py
    python run_fast_tests.py -v
"""

import subprocess
import sys
import time

def main():
    args = sys.argv[1:]
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_full_suite.py",
        "-m", "not slow",
        "--asyncio-mode=auto",
        "--tb=short",
        "-q",
    ] + args

    print("Running fast tests (not slow)...")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    print(f"\nFast tests completed in {elapsed:.1f}s")
    if elapsed > 60:
        print(f"WARNING: exceeded 60s target ({elapsed:.1f}s)")
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
