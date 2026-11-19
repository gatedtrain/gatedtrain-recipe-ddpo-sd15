"""The adapter `ddpo_sd15` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "ddpo_sd15"


if __name__ == "__main__":
    unittest.main()
