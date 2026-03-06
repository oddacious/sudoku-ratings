"""Tests for ratings.methods.rating_tracker module.

These tests verify the RatingTracker produces consistent results and
correctly handles chronological ordering of competitions.
"""

import unittest

import polars as pl

from ratings.competitions import CompetitionIdentifier, get_competition_index
from ratings.leaderboard import INACTIVITY_THRESHOLD_YEARS
from ratings.methods.points_based import build_features_and_labels, build_features_with_prior
from ratings.methods.rating_tracker import RatingTracker


class TestGetCompetitionIndex(unittest.TestCase):
    """Test the get_competition_index function."""

    def test_returns_dict(self):
        """Test that it returns a dictionary."""
        result = get_competition_index()
        self.assertIsInstance(result, dict)

    def test_keys_are_tuples(self):
        """Test that keys are (year, round, event_type) tuples."""
        result = get_competition_index()
        for key in list(result.keys())[:5]:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 3)
            year, rnd, event_type = key
            self.assertIsInstance(year, int)
            self.assertIsInstance(rnd, int)
            self.assertIsInstance(event_type, str)

    def test_values_are_sequential(self):
        """Test that values are sequential integers starting from 0."""
        result = get_competition_index()
        values = sorted(result.values())
        expected = list(range(len(values)))
        self.assertEqual(values, expected)

    def test_gp_before_wsc_same_year(self):
        """Test that GP rounds come before WSC rounds in the same year."""
        result = get_competition_index()
        # Check 2014: GP should come before WSC
        gp_r1_idx = result.get((2014, 1, "GP"))
        wsc_r1_idx = result.get((2014, 1, "WSC"))
        if gp_r1_idx is not None and wsc_r1_idx is not None:
            self.assertLess(gp_r1_idx, wsc_r1_idx)

    def test_chronological_order(self):
        """Test that competitions are in chronological order."""
        result = get_competition_index()
        # 2014 GP R7 should come before 2014 WSC R5
        gp_r7_idx = result.get((2014, 7, "GP"))
        wsc_r5_idx = result.get((2014, 5, "WSC"))
        if gp_r7_idx is not None and wsc_r5_idx is not None:
            self.assertLess(gp_r7_idx, wsc_r5_idx,
                "GP R7 should be chronologically before WSC R5 despite round numbers")


class TestRatingTracker(unittest.TestCase):
    """Test the RatingTracker class."""

    def setUp(self):
        """Create sample adjusted data for testing."""
        # Create minimal test data with a few solvers across GP and WSC
        self.test_data = pl.DataFrame({
            "user_pseudo_id": [
                # 2014 GP rounds
                "Solver A", "Solver B", "Solver A", "Solver B",
                "Solver A", "Solver B", "Solver A", "Solver B",
                # 2014 WSC rounds
                "Solver A", "Solver B", "Solver A", "Solver B",
            ],
            "year": [2014] * 12,
            "round": [
                1, 1, 2, 2, 3, 3, 4, 4,  # GP R1-R4
                1, 1, 2, 2,  # WSC R1-R2
            ],
            "competition": [
                "GP", "GP", "GP", "GP", "GP", "GP", "GP", "GP",
                "WSC", "WSC", "WSC", "WSC",
            ],
            "points": [
                100, 90, 110, 85, 105, 95, 115, 80,  # GP
                120, 100, 110, 90,  # WSC
            ],
            "adjusted_points": [
                100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0,  # GP
                120.0, 100.0, 110.0, 90.0,  # WSC
            ],
        })

    def test_init(self):
        """Test tracker initialization."""
        tracker = RatingTracker(self.test_data, min_history=3, decay_rate=0.90)
        self.assertEqual(tracker.min_history, 3)
        self.assertEqual(tracker.decay_rate, 0.90)
        self.assertEqual(tracker.prior_k, 0)
        self.assertEqual(len(tracker.histories), 0)

    def test_advance_to_single_round(self):
        """Test advancing to a single round."""
        tracker = RatingTracker(self.test_data, min_history=1)
        comp = CompetitionIdentifier(2014, 1, "GP")
        tracker.advance_to(comp)

        # Both solvers should have history after GP R1
        self.assertIn("Solver A", tracker.histories)
        self.assertIn("Solver B", tracker.histories)
        self.assertEqual(len(tracker.histories["Solver A"]), 1)
        self.assertEqual(tracker.histories["Solver A"][0], 100.0)

    def test_advance_to_multiple_rounds(self):
        """Test advancing through multiple rounds."""
        tracker = RatingTracker(self.test_data, min_history=1)

        # Advance to GP R4
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))
        self.assertEqual(len(tracker.histories["Solver A"]), 4)

        # Advance to WSC R1 (should add one more)
        tracker.advance_to(CompetitionIdentifier(2014, 1, "WSC"))
        self.assertEqual(len(tracker.histories["Solver A"]), 5)

    def test_chronological_order_maintained(self):
        """Test that GP rounds are processed before WSC rounds."""
        tracker = RatingTracker(self.test_data, min_history=1)

        # Advance to WSC R1 - should include all GP rounds first
        tracker.advance_to(CompetitionIdentifier(2014, 1, "WSC"))

        # Solver A's history should be: GP R1-R4 + WSC R1
        expected = [100.0, 110.0, 105.0, 115.0, 120.0]
        self.assertEqual(tracker.histories["Solver A"], expected)

    def test_get_solver_rating_min_history(self):
        """Test that ratings require minimum history."""
        tracker = RatingTracker(self.test_data, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 2, "GP"))

        # With only 2 rounds, solver shouldn't have a rating yet
        rating = tracker.get_solver_rating("Solver A")
        self.assertIsNone(rating)

        # After 3 rounds, solver should have a rating
        tracker.advance_to(CompetitionIdentifier(2014, 3, "GP"))
        rating = tracker.get_solver_rating("Solver A")
        self.assertIsNotNone(rating)
        self.assertEqual(rating.n_rounds, 3)

    def test_get_leaderboard(self):
        """Test getting leaderboard."""
        tracker = RatingTracker(self.test_data, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        leaderboard = tracker.get_leaderboard(top_n=10, as_of_year=2014)

        # Both solvers should be on leaderboard
        self.assertEqual(len(leaderboard), 2)

        # Solver A has better scores, should be ranked higher
        self.assertEqual(leaderboard[0].solver_id, "Solver A")
        self.assertEqual(leaderboard[1].solver_id, "Solver B")

    def test_get_solver_rank(self):
        """Test getting a specific solver's rank."""
        tracker = RatingTracker(self.test_data, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        rank_a = tracker.get_solver_rank("Solver A", as_of_year=2014)
        rank_b = tracker.get_solver_rank("Solver B", as_of_year=2014)

        self.assertIsNotNone(rank_a)
        self.assertIsNotNone(rank_b)
        self.assertEqual(rank_a, (1, 2))  # Rank 1 of 2
        self.assertEqual(rank_b, (2, 2))  # Rank 2 of 2

    def test_prior_regularization(self):
        """Test that prior_k adds regularization."""
        tracker_no_prior = RatingTracker(self.test_data, min_history=3, prior_k=0)
        tracker_with_prior = RatingTracker(self.test_data, min_history=3, prior_k=3)

        tracker_no_prior.advance_to(CompetitionIdentifier(2014, 4, "GP"))
        tracker_with_prior.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        rating_no_prior = tracker_no_prior.get_solver_rating("Solver A")
        rating_with_prior = tracker_with_prior.get_solver_rating("Solver A")

        # Ratings should be different due to prior
        self.assertIsNotNone(rating_no_prior)
        self.assertIsNotNone(rating_with_prior)
        self.assertNotAlmostEqual(rating_no_prior.rating, rating_with_prior.rating)

    def test_inactivity_filtering(self):
        """Test that inactive solvers are filtered from leaderboard."""
        # Create data with solver active only in 2014
        old_data = pl.DataFrame({
            "user_pseudo_id": ["Old Solver"] * 4,
            "year": [2014] * 4,
            "round": [1, 2, 3, 4],
            "competition": ["GP"] * 4,
            "points": [100, 100, 100, 100],
            "adjusted_points": [100.0, 100.0, 100.0, 100.0],
        })

        tracker = RatingTracker(old_data, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        # In 2014, solver should appear
        lb_2014 = tracker.get_leaderboard(top_n=10, as_of_year=2014)
        self.assertEqual(len(lb_2014), 1)

        # In 2016 (after threshold), solver should be filtered out
        lb_2016 = tracker.get_leaderboard(top_n=10, as_of_year=2016)
        self.assertEqual(len(lb_2016), 0)

    def test_reset(self):
        """Test that reset clears state."""
        tracker = RatingTracker(self.test_data, min_history=1)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        self.assertGreater(len(tracker.histories), 0)

        tracker.reset()

        self.assertEqual(len(tracker.histories), 0)
        self.assertEqual(tracker.current_comp_idx, -1)
        self.assertEqual(tracker.prior_sum, 0.0)


class TestRatingTrackerConsistency(unittest.TestCase):
    """Test that RatingTracker produces consistent results.

    These tests are designed to catch bugs like the (year, round) sorting issue
    where GP R7 was incorrectly considered "after" WSC R5.
    """

    def test_wsc_includes_prior_gp_history(self):
        """Test that WSC ratings include GP history from same year."""
        # Solver performs in GP R1-R7, then WSC R1-R5
        data = pl.DataFrame({
            "user_pseudo_id": ["Solver"] * 12,
            "year": [2014] * 12,
            "round": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5],
            "competition": ["GP"] * 7 + ["WSC"] * 5,
            "points": [100] * 12,
            "adjusted_points": [100.0] * 12,
        })

        tracker = RatingTracker(data, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 5, "WSC"))

        rating = tracker.get_solver_rating("Solver")
        self.assertIsNotNone(rating)
        # Should have 12 rounds of history (7 GP + 5 WSC)
        self.assertEqual(rating.n_rounds, 12)

    def test_rating_monotonic_with_history(self):
        """Test that rating calculation is deterministic."""
        data = pl.DataFrame({
            "user_pseudo_id": ["Solver"] * 5,
            "year": [2014] * 5,
            "round": [1, 2, 3, 4, 5],
            "competition": ["GP"] * 5,
            "points": [100, 100, 100, 100, 100],
            "adjusted_points": [100.0, 100.0, 100.0, 100.0, 100.0],
        })

        # Run twice and verify same result
        tracker1 = RatingTracker(data, min_history=3, decay_rate=0.90)
        tracker2 = RatingTracker(data, min_history=3, decay_rate=0.90)

        tracker1.advance_to(CompetitionIdentifier(2014, 5, "GP"))
        tracker2.advance_to(CompetitionIdentifier(2014, 5, "GP"))

        rating1 = tracker1.get_solver_rating("Solver")
        rating2 = tracker2.get_solver_rating("Solver")

        self.assertEqual(rating1.rating, rating2.rating)
        self.assertEqual(rating1.n_rounds, rating2.n_rounds)


class TestCrossPathConsistency(unittest.TestCase):
    """Verify RatingTracker and build_features_and_labels produce identical ratings.

    These tests ensure the two code paths that compute ratings (the batch
    build_features_and_labels used for backtesting, and the incremental
    RatingTracker used for CLI commands) produce identical results. This
    guards against the kind of divergence that caused the original bug.
    """

    def test_no_prior_ratings_match(self):
        """RatingTracker (no prior) matches build_features_and_labels exp_weighted_mean."""
        data = pl.DataFrame({
            "user_pseudo_id": ["Solver"] * 6,
            "year": [2014] * 6,
            "round": [1, 2, 3, 4, 5, 6],
            "competition": ["GP"] * 6,
            "points": [100, 120, 90, 110, 105, 115],
            "adjusted_points": [100.0, 120.0, 90.0, 110.0, 105.0, 115.0],
        })

        decay_rate = 0.90
        min_history = 3

        # Path 1: batch feature computation
        features_df, _ = build_features_and_labels(
            data, min_history=min_history, decay_rate=decay_rate
        )

        # Path 2: incremental tracker
        tracker = RatingTracker(
            data, min_history=min_history, decay_rate=decay_rate, prior_k=0
        )

        solver_features = features_df.filter(
            pl.col("user_pseudo_id") == "Solver"
        ).sort("round")
        self.assertGreater(len(solver_features), 0)

        for row in solver_features.iter_rows(named=True):
            # Feature at round N uses history from rounds before N.
            # Advance tracker to round N-1 to get the same history.
            prev_round = row["round"] - 1
            tracker.advance_to(CompetitionIdentifier(2014, prev_round, "GP"))

            rating = tracker.get_solver_rating("Solver")
            self.assertIsNotNone(rating)
            self.assertAlmostEqual(
                rating.rating, row["exp_weighted_mean"],
                places=10,
                msg=f"Rating mismatch at round {row['round']}"
            )
            self.assertEqual(
                rating.n_rounds, row["n_rounds"],
                msg=f"Round count mismatch at round {row['round']}"
            )

    def test_with_prior_ratings_match(self):
        """RatingTracker (with prior) matches build_features_with_prior exp_weighted_mean."""
        data = pl.DataFrame({
            "user_pseudo_id": ["Solver"] * 6,
            "year": [2014] * 6,
            "round": [1, 2, 3, 4, 5, 6],
            "competition": ["GP"] * 6,
            "points": [100, 120, 90, 110, 105, 115],
            "adjusted_points": [100.0, 120.0, 90.0, 110.0, 105.0, 115.0],
        })

        decay_rate = 0.90
        min_history = 3
        prior_k = 3

        # Path 1: batch feature computation with prior
        features_df, _ = build_features_with_prior(
            data, min_history=min_history, decay_rate=decay_rate, prior_k=prior_k
        )

        # Path 2: incremental tracker with prior
        tracker = RatingTracker(
            data, min_history=min_history, decay_rate=decay_rate, prior_k=prior_k
        )

        solver_features = features_df.filter(
            pl.col("user_pseudo_id") == "Solver"
        ).sort("round")
        self.assertGreater(len(solver_features), 0)

        for row in solver_features.iter_rows(named=True):
            prev_round = row["round"] - 1
            tracker.advance_to(CompetitionIdentifier(2014, prev_round, "GP"))

            rating = tracker.get_solver_rating("Solver")
            self.assertIsNotNone(rating)
            self.assertAlmostEqual(
                rating.rating, row["exp_weighted_mean"],
                places=10,
                msg=f"Prior rating mismatch at round {row['round']}"
            )

    def test_multi_solver_prior_stays_in_sync(self):
        """Prior mean stays in sync between paths with multiple solvers."""
        data = pl.DataFrame({
            "user_pseudo_id": [
                "A", "B", "A", "B", "A", "B", "A", "B", "A", "B",
            ],
            "year": [2014] * 10,
            "round": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "competition": ["GP"] * 10,
            "points": [100, 80, 120, 95, 90, 85, 110, 100, 105, 90],
            "adjusted_points": [100.0, 80.0, 120.0, 95.0, 90.0, 85.0, 110.0, 100.0, 105.0, 90.0],
        })

        decay_rate = 0.90
        min_history = 3
        prior_k = 2

        features_df, _ = build_features_with_prior(
            data, min_history=min_history, decay_rate=decay_rate, prior_k=prior_k
        )

        tracker = RatingTracker(
            data, min_history=min_history, decay_rate=decay_rate, prior_k=prior_k
        )

        for row in features_df.sort(["round", "user_pseudo_id"]).iter_rows(named=True):
            solver_id = row["user_pseudo_id"]
            prev_round = row["round"] - 1
            tracker.advance_to(CompetitionIdentifier(2014, prev_round, "GP"))

            rating = tracker.get_solver_rating(solver_id)
            self.assertIsNotNone(rating, f"{solver_id} should have rating before round {row['round']}")
            self.assertAlmostEqual(
                rating.rating, row["exp_weighted_mean"],
                places=10,
                msg=f"{solver_id} rating mismatch at round {row['round']}"
            )

    def test_varying_scores_across_rounds(self):
        """Both paths handle non-uniform scores identically."""
        # Deliberately varied scores to stress-test the calculation
        data = pl.DataFrame({
            "user_pseudo_id": ["S"] * 7,
            "year": [2014] * 7,
            "round": [1, 2, 3, 4, 5, 6, 7],
            "competition": ["GP"] * 7,
            "points": [50, 200, 75, 300, 10, 150, 500],
            "adjusted_points": [50.0, 200.0, 75.0, 300.0, 10.0, 150.0, 500.0],
        })

        for decay_rate in [0.50, 0.90, 0.99]:
            features_df, _ = build_features_and_labels(
                data, min_history=3, decay_rate=decay_rate
            )
            tracker = RatingTracker(
                data, min_history=3, decay_rate=decay_rate, prior_k=0
            )

            for row in features_df.sort("round").iter_rows(named=True):
                tracker.advance_to(CompetitionIdentifier(2014, row["round"] - 1, "GP"))
                rating = tracker.get_solver_rating("S")
                self.assertAlmostEqual(
                    rating.rating, row["exp_weighted_mean"],
                    places=10,
                    msg=f"Mismatch at round {row['round']} with decay={decay_rate}"
                )
            # Reset tracker for next decay_rate iteration — create fresh
            tracker = None


if __name__ == "__main__":
    unittest.main()
