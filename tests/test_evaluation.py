"""Tests for the evaluation module."""

import unittest

from ratings.methods.glicko import GlickoRating
from ratings.evaluation import (
    PairwiseResult,
    PairwisePrediction,
    extract_pairwise,
    predict_pairwise,
    pairwise_accuracy,
    backtest_glicko,
    grid_search_glicko,
)


class TestExtractPairwise(unittest.TestCase):
    """Test pairwise extraction from round results."""

    def test_two_players(self):
        """Two players should produce one pair."""
        results = [("alice", 100), ("bob", 90)]
        pairs = extract_pairwise(results)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].winner_id, "alice")
        self.assertEqual(pairs[0].loser_id, "bob")
        self.assertFalse(pairs[0].is_tie)

    def test_three_players(self):
        """Three players should produce three pairs."""
        results = [("alice", 100), ("bob", 90), ("charlie", 80)]
        pairs = extract_pairwise(results)
        self.assertEqual(len(pairs), 3)

        # Verify all expected matchups exist
        matchups = {(p.winner_id, p.loser_id) for p in pairs}
        self.assertIn(("alice", "bob"), matchups)
        self.assertIn(("alice", "charlie"), matchups)
        self.assertIn(("bob", "charlie"), matchups)

    def test_tie(self):
        """Tied scores should produce a tie result."""
        results = [("alice", 100), ("bob", 100)]
        pairs = extract_pairwise(results)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].is_tie)

    def test_pair_count_formula(self):
        """n players should produce n*(n-1)/2 pairs."""
        for n in [2, 5, 10]:
            results = [(f"player_{i}", 100 - i) for i in range(n)]
            pairs = extract_pairwise(results)
            expected_pairs = n * (n - 1) // 2
            self.assertEqual(len(pairs), expected_pairs)

    def test_empty_results(self):
        """Empty results should produce no pairs."""
        pairs = extract_pairwise([])
        self.assertEqual(len(pairs), 0)

    def test_single_player(self):
        """Single player should produce no pairs."""
        pairs = extract_pairwise([("alice", 100)])
        self.assertEqual(len(pairs), 0)


class TestPredictPairwise(unittest.TestCase):
    """Test pairwise prediction generation."""

    def test_higher_rated_favored(self):
        """Higher rated player should have >0.5 win probability."""
        pairs = [PairwiseResult("alice", "bob", False)]
        ratings = {
            "alice": GlickoRating(1600, 100),
            "bob": GlickoRating(1400, 100),
        }
        predictions = predict_pairwise(pairs, ratings)
        self.assertEqual(len(predictions), 1)
        self.assertGreater(predictions[0].prob_a_wins, 0.5)

    def test_equal_ratings(self):
        """Equal ratings should give 0.5 probability."""
        pairs = [PairwiseResult("alice", "bob", False)]
        ratings = {
            "alice": GlickoRating(1500, 100),
            "bob": GlickoRating(1500, 100),
        }
        predictions = predict_pairwise(pairs, ratings)
        self.assertAlmostEqual(predictions[0].prob_a_wins, 0.5, places=5)

    def test_unknown_player_gets_default(self):
        """Unknown players should get default rating."""
        pairs = [PairwiseResult("alice", "bob", False)]
        ratings = {"alice": GlickoRating(1600, 100)}  # bob unknown
        predictions = predict_pairwise(pairs, ratings)
        # alice rated higher than default (1500), should be favored
        self.assertGreater(predictions[0].prob_a_wins, 0.5)


class TestPairwiseAccuracy(unittest.TestCase):
    """Test accuracy calculation."""

    def test_perfect_predictions(self):
        """All correct predictions should give 1.0 accuracy."""
        predictions = [
            PairwisePrediction("alice", "bob", 0.7),
            PairwisePrediction("charlie", "dave", 0.8),
        ]
        actuals = [
            PairwiseResult("alice", "bob", False),
            PairwiseResult("charlie", "dave", False),
        ]
        self.assertEqual(pairwise_accuracy(predictions, actuals), 1.0)

    def test_all_wrong_predictions(self):
        """All wrong predictions should give 0.0 accuracy."""
        predictions = [
            PairwisePrediction("alice", "bob", 0.3),  # Predicted bob wins
            PairwisePrediction("charlie", "dave", 0.2),
        ]
        actuals = [
            PairwiseResult("alice", "bob", False),  # But alice won
            PairwiseResult("charlie", "dave", False),
        ]
        self.assertEqual(pairwise_accuracy(predictions, actuals), 0.0)

    def test_ties_count_as_half(self):
        """Actual ties should count as 0.5 correct."""
        predictions = [
            PairwisePrediction("alice", "bob", 0.7),
        ]
        actuals = [
            PairwiseResult("alice", "bob", True),  # Actual tie
        ]
        self.assertEqual(pairwise_accuracy(predictions, actuals), 0.5)

    def test_predicted_tie(self):
        """Predicted tie (0.5) on decisive result counts as 0.5."""
        predictions = [
            PairwisePrediction("alice", "bob", 0.5),
        ]
        actuals = [
            PairwiseResult("alice", "bob", False),
        ]
        self.assertEqual(pairwise_accuracy(predictions, actuals), 0.5)

    def test_mixed_results(self):
        """Mixed results should give appropriate accuracy."""
        predictions = [
            PairwisePrediction("a", "b", 0.7),  # Correct (a wins)
            PairwisePrediction("c", "d", 0.3),  # Wrong (c wins but predicted d)
        ]
        actuals = [
            PairwiseResult("a", "b", False),
            PairwiseResult("c", "d", False),
        ]
        self.assertEqual(pairwise_accuracy(predictions, actuals), 0.5)

    def test_empty_lists(self):
        """Empty lists should return 0.0."""
        self.assertEqual(pairwise_accuracy([], []), 0.0)


class TestBacktestGlicko(unittest.TestCase):
    """Test backtesting functionality."""

    def test_basic_backtest(self):
        """Backtest should run without errors."""
        rounds = [
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 95), ("bob", 85), ("charlie", 75)],
            [("alice", 90), ("bob", 88), ("charlie", 70)],
            [("alice", 92), ("bob", 80), ("charlie", 72)],
        ]
        result = backtest_glicko(rounds, train_rounds=2)
        self.assertEqual(result.rounds_evaluated, 2)
        self.assertGreater(result.total_pairs, 0)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)

    def test_consistent_winner_high_accuracy(self):
        """Consistent results should lead to high prediction accuracy."""
        # Alice always wins, bob always second, charlie always third
        rounds = [
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
        ]
        result = backtest_glicko(rounds, train_rounds=3)
        # After training, predictions should be accurate
        self.assertGreater(result.accuracy, 0.8)

    def test_final_ratings_populated(self):
        """Final ratings should include all participants."""
        rounds = [
            [("alice", 100), ("bob", 90)],
            [("alice", 95), ("bob", 85)],
        ]
        result = backtest_glicko(rounds, train_rounds=1)
        self.assertIn("alice", result.final_ratings)
        self.assertIn("bob", result.final_ratings)

    def test_train_rounds_validation(self):
        """Should raise error if train_rounds >= total rounds."""
        rounds = [
            [("alice", 100), ("bob", 90)],
            [("alice", 95), ("bob", 85)],
        ]
        with self.assertRaises(ValueError):
            backtest_glicko(rounds, train_rounds=2)


class TestGridSearchGlicko(unittest.TestCase):
    """Test parameter grid search."""

    def test_returns_best_params(self):
        """Grid search should return c value and accuracy."""
        rounds = [
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
            [("alice", 100), ("bob", 90), ("charlie", 80)],
        ]
        best_c, best_accuracy = grid_search_glicko(
            rounds, train_rounds=2, c_values=[30, 50, 70]
        )
        self.assertIn(best_c, [30, 50, 70])
        self.assertGreaterEqual(best_accuracy, 0.0)
        self.assertLessEqual(best_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
