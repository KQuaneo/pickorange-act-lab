import unittest

from experiments.pick_orange_analysis import (
    align_paired_results,
    dataset_sampling_stats,
    exact_mcnemar,
    horizon_fairness,
    horizon_protocol_spec,
    matched_exposure_steps,
    normalized_l2,
    paired_bootstrap_difference,
    post_success_overrun,
)


class PickOrangeAnalysisTest(unittest.TestCase):
    def test_horizon_fairness_exposes_current_mismatch(self):
        audit = horizon_fairness()
        self.assertEqual(audit["a0"]["theoretical_duration_s"], 34.0)
        self.assertEqual(audit["a1"]["theoretical_duration_s"], 42.0)
        self.assertFalse(audit["equal_total_horizon"])
        self.assertEqual(audit["comparable_a0_policy_steps"], 1260)
        self.assertFalse(horizon_protocol_spec("native_horizon")["same_total_horizon"])
        self.assertTrue(horizon_protocol_spec("matched_horizon")["same_total_horizon"])

    def test_post_success_overrun_preserves_missing_as_null(self):
        self.assertIsNone(post_success_overrun(420, None))
        self.assertEqual(post_success_overrun(420, 358), 62)


    def test_dataset_exposure_and_matched_steps_use_frames_not_episode_ratio(self):
        stats = dataset_sampling_stats(2, 220, [100, 120], 100, 64, 1000)
        self.assertEqual(stats["sample_anchor_windows"], 220)
        self.assertEqual(stats["full_unpadded_windows"], 22)
        self.assertEqual(matched_exposure_steps(stats, 330), 1500)


    def test_paired_alignment_and_exact_mcnemar(self):
        left = [{"seed": 1, "episode": i, "initial_state_id": str(i), "success": i == 0} for i in range(3)]
        right = [{"seed": 1, "episode": i, "initial_state_id": str(i), "success": i != 0} for i in range(3)]
        aligned = align_paired_results(left, right)
        self.assertEqual(len(aligned["pairs"]), 3)
        result = exact_mcnemar([row[0]["success"] for row in aligned["pairs"]], [row[1]["success"] for row in aligned["pairs"]])
        self.assertEqual(result["discordant"], 3)
        self.assertEqual(result["exact_two_sided_p"], 1.0)


    def test_bootstrap_and_normalized_distance_are_deterministic(self):
        result = paired_bootstrap_difference([0, 0, 1], [0, 1, 1], samples=1000, seed=7)
        self.assertEqual(result["difference"], 1 / 3)
        self.assertEqual(result, paired_bootstrap_difference([0, 0, 1], [0, 1, 1], samples=1000, seed=7))
        self.assertEqual(normalized_l2([2, 4], [1, 2], [1, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
