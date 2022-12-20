import math
import unittest

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../sicali")

from math_utils import wrap_to_pi


class TestTestMathUtils(unittest.TestCase):
    def test_wrap_to_pi(self):
        self.assertAlmostEqual(wrap_to_pi(0.0), 0.0)
        self.assertAlmostEqual(wrap_to_pi(-0.0), 0.0)

        self.assertAlmostEqual(wrap_to_pi(1.0), 1.0)
        self.assertAlmostEqual(wrap_to_pi(-1.0), -1.0)

        self.assertAlmostEqual(wrap_to_pi(math.pi), math.pi)
        self.assertAlmostEqual(wrap_to_pi(-math.pi), math.pi)

        self.assertAlmostEqual(wrap_to_pi(2 * math.pi), 0.0)
        self.assertAlmostEqual(wrap_to_pi(-2 * math.pi), 0.0)

        self.assertAlmostEqual(wrap_to_pi(6 * math.pi + 1.2), 1.2)
        self.assertAlmostEqual(wrap_to_pi(-6 * math.pi + 1.2), 1.2)


if __name__ == "__main__":
    unittest.main()
