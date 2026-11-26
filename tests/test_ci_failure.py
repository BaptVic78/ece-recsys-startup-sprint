import unittest


class CiFailureCheck(unittest.TestCase):
    def test_ci_reports_a_failure(self):
        self.fail("Temporary failure to verify the PR check")
