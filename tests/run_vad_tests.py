#!/usr/bin/env python
# run_vad_tests.py

import sys
import unittest

if __name__ == "__main__":
    # Load tests from the test_vad module
    test_suite = unittest.defaultTestLoader.discover("asr", pattern="test_vad.py")

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    result = test_runner.run(test_suite)

    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())
