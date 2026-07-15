import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "experiments/eval_pick_orange_temporal_aggregation.py"
RUNNER = ROOT / "experiments/run_pick_orange_temporal_aggregation.py"


def offline_ensemble(predictions, coeff=0.01):
    weights = [math.exp(-coeff * index) for index in range(len(predictions))]
    total = sum(weights)
    return sum(weight * value for weight, value in zip(weights, predictions, strict=True)) / total


class TemporalAggregationProtocolTest(unittest.TestCase):
    def test_oldest_prediction_gets_largest_weight_for_positive_coeff(self):
        weights = [math.exp(-0.01 * index) for index in range(100)]
        self.assertGreater(weights[0], weights[-1])
        self.assertAlmostEqual(offline_ensemble([0.0, 10.0]), 10.0 * weights[1] / (weights[0] + weights[1]))

    def test_overlap_size_saturates_at_chunk_size(self):
        sizes = [min(step + 1, 100) for step in range(420)]
        self.assertEqual(sizes[:3], [1, 2, 3])
        self.assertEqual(sizes[99], 100)
        self.assertTrue(all(size == 100 for size in sizes[99:]))

    def test_evaluator_uses_native_chunk_ensemble_not_action_moving_average(self):
        source = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler", source)
        self.assertIn("raw_chunk = bundle.policy.predict_action_chunk", source)
        self.assertIn("aggregated_raw = ensembler.update(raw_chunk)", source)
        self.assertIn("latest_raw = raw_chunk[:, 0]", source)
        self.assertNotIn("moving_average", source)
        self.assertNotIn("rolling_mean", source)

    def test_pairing_and_checkpoint_gates_precede_completed_marker(self):
        source = EVALUATOR.read_text(encoding="utf-8")
        pairing_gate = source.index("if not pair_ok:")
        hash_gate = source.index("if checkpoint_sha_after != checkpoint_sha_before:")
        completed = source.index('artifacts.directory / "COMPLETED"')
        self.assertLess(pairing_gate, completed)
        self.assertLess(hash_gate, completed)

    def test_runner_orders_smoke_before_formal(self):
        source = RUNNER.read_text(encoding="utf-8")
        smoke_loop = source.index("for episodes in (1, 3):")
        formal_run = source.index("formal_orchestrator.log")
        self.assertLess(smoke_loop, formal_run)
        self.assertIn("GPU became occupied after smoke; formal run stopped", source)
        self.assertIn("server deletion flow must be paused", source)


if __name__ == "__main__":
    unittest.main()
