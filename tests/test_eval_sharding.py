import unittest

from evaluate_cdh_bench import _evaluation_task_shard


class EvaluationShardingTest(unittest.TestCase):
    def test_two_shards_are_disjoint_complete_and_balanced(self):
        tasks = [
            (item, task, side)
            for item in range(7)
            for task in range(2)
            for side in range(2)
        ]
        shards = [
            {
                key
                for key in tasks
                if _evaluation_task_shard(*key, task_count=2, shard_count=2)
                == shard
            }
            for shard in range(2)
        ]
        self.assertFalse(shards[0] & shards[1])
        self.assertEqual(shards[0] | shards[1], set(tasks))
        self.assertEqual([len(values) for values in shards], [14, 14])

    def test_membership_does_not_depend_on_completed_rows(self):
        before = _evaluation_task_shard(5, 1, 0, task_count=2, shard_count=3)
        after_unrelated_rows_finish = _evaluation_task_shard(
            5, 1, 0, task_count=2, shard_count=3
        )
        self.assertEqual(before, after_unrelated_rows_finish)


if __name__ == "__main__":
    unittest.main()
