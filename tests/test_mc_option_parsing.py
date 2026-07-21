import unittest

from evaluate_cdh_bench import _candidate_answers, _get_options


class MCOptionParsingTest(unittest.TestCase):
    def test_splits_bullet_delimited_singleton(self):
        item = {
            "multiple_choice": {
                "options": ["A. 2 centers • B. 1 center • C. 3 centers • D. 4 centers"]
            }
        }
        self.assertEqual(
            _get_options(item),
            ["A. 2 centers", "B. 1 center", "C. 3 centers", "D. 4 centers"],
        )
        self.assertEqual(
            [value["key"] for value in _candidate_answers("mc", item)],
            ["A", "B", "C", "D"],
        )

    def test_splits_space_delimited_singleton(self):
        item = {
            "multiple_choice": {
                "options": ["A. Hot  B. Warm C. Cold D. Room temperature"]
            }
        }
        self.assertEqual(len(_get_options(item)), 4)

    def test_preserves_normal_option_list(self):
        options = ["A. red", "B. blue", "C. green", "D. black"]
        item = {"multiple_choice": {"options": options}}
        self.assertEqual(_get_options(item), options)


if __name__ == "__main__":
    unittest.main()
