import unittest

from filter_result_pair_manifest import filter_rows


class FilterResultPairManifestTest(unittest.TestCase):
    def test_filters_by_exact_pair_id(self):
        rows = [{"pair_id": "Pair 1"}, {"pair_id": "Pair 10"}]
        self.assertEqual(filter_rows(rows, {"Pair 1"}), [{"pair_id": "Pair 1"}])


if __name__ == "__main__":
    unittest.main()
