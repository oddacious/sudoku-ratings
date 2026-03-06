"""Tests for the percentile-based rating system."""

import unittest

import polars as pl

from ratings.methods.percentile_based import (
    compute_percentiles,
    estimate_field_strength,
    compute_adjusted_percentiles,
    compute_percentile_ratings,
    DEFAULT_PERCENTILE_RATING,
    REFERENCE_STRENGTH,
    FIELD_STRENGTH_K,
)


class TestComputePercentiles(unittest.TestCase):
    """Test the compute_percentiles function."""

    def test_basic_ranking(self):
        """Test basic ranking with clear winner/loser."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
            {"user_pseudo_id": "C", "year": 2022, "round": 1, "competition": "GP", "points": 25},
        ])

        result = compute_percentiles(data)

        # Winner should have percentile 1.0, last should have 0.0
        a_row = result.filter(pl.col("user_pseudo_id") == "A")
        b_row = result.filter(pl.col("user_pseudo_id") == "B")
        c_row = result.filter(pl.col("user_pseudo_id") == "C")

        self.assertEqual(a_row["raw_percentile"][0], 1.0)
        self.assertEqual(b_row["raw_percentile"][0], 0.5)
        self.assertEqual(c_row["raw_percentile"][0], 0.0)

    def test_handles_ties(self):
        """Test that ties are handled with average rank."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
            {"user_pseudo_id": "C", "year": 2022, "round": 1, "competition": "GP", "points": 50},
            {"user_pseudo_id": "D", "year": 2022, "round": 1, "competition": "GP", "points": 25},
        ])

        result = compute_percentiles(data)

        # B and C are tied for 2nd (ranks 2 and 3), so average rank = 2.5
        b_row = result.filter(pl.col("user_pseudo_id") == "B")
        c_row = result.filter(pl.col("user_pseudo_id") == "C")

        self.assertEqual(b_row["rank"][0], 2.5)
        self.assertEqual(c_row["rank"][0], 2.5)
        # Their percentiles should be equal
        self.assertEqual(b_row["raw_percentile"][0], c_row["raw_percentile"][0])
        # Percentile = 1 - (2.5 - 1) / (4 - 1) = 1 - 1.5/3 = 0.5
        self.assertAlmostEqual(b_row["raw_percentile"][0], 0.5, places=5)

    def test_handles_zeros(self):
        """Test that DNF/zero points are included with low percentile."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
            {"user_pseudo_id": "C", "year": 2022, "round": 1, "competition": "GP", "points": 0},
        ])

        result = compute_percentiles(data)

        # Zero points should still be ranked (last place)
        c_row = result.filter(pl.col("user_pseudo_id") == "C")
        self.assertEqual(c_row["rank"][0], 3.0)
        self.assertEqual(c_row["raw_percentile"][0], 0.0)

    def test_single_participant(self):
        """Test that a single participant gets percentile 1.0."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
        ])

        result = compute_percentiles(data)

        self.assertEqual(result["raw_percentile"][0], 1.0)

    def test_multiple_rounds(self):
        """Test that rankings are computed per round."""
        data = pl.DataFrame([
            # Round 1: A wins
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
            # Round 2: B wins
            {"user_pseudo_id": "A", "year": 2022, "round": 2, "competition": "GP", "points": 40},
            {"user_pseudo_id": "B", "year": 2022, "round": 2, "competition": "GP", "points": 80},
        ])

        result = compute_percentiles(data)

        # A wins round 1
        a_r1 = result.filter(
            (pl.col("user_pseudo_id") == "A") & (pl.col("round") == 1)
        )
        self.assertEqual(a_r1["raw_percentile"][0], 1.0)

        # B wins round 2
        b_r2 = result.filter(
            (pl.col("user_pseudo_id") == "B") & (pl.col("round") == 2)
        )
        self.assertEqual(b_r2["raw_percentile"][0], 1.0)


class TestFieldStrengthEstimation(unittest.TestCase):
    """Test the estimate_field_strength function."""

    def test_with_known_ratings(self):
        """Test field strength with known solver ratings."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
        ])

        ratings = {"A": 0.8, "B": 0.6}
        data = compute_percentiles(data)

        field_strengths = estimate_field_strength(data, ratings)

        # Mean of 0.8 and 0.6 = 0.7
        self.assertAlmostEqual(
            field_strengths[(2022, 1, "GP")], 0.7, places=5
        )

    def test_with_unknown_solvers(self):
        """Test that unknown solvers get default rating."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
        ])

        # Only A has a rating
        ratings = {"A": 0.9}
        data = compute_percentiles(data)

        field_strengths = estimate_field_strength(data, ratings)

        # Mean of 0.9 and DEFAULT_PERCENTILE_RATING (0.5)
        expected = (0.9 + DEFAULT_PERCENTILE_RATING) / 2
        self.assertAlmostEqual(
            field_strengths[(2022, 1, "GP")], expected, places=5
        )


class TestAdjustedPercentiles(unittest.TestCase):
    """Test the compute_adjusted_percentiles function."""

    def test_basic_adjustment(self):
        """Test that field strength adjusts percentiles correctly."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
        ])

        data = compute_percentiles(data)

        # Higher field strength should increase percentiles
        field_strengths = {(2022, 1, "GP"): 0.7}  # Above reference (0.5)

        result = compute_adjusted_percentiles(
            data, field_strengths, reference_strength=0.5, k=0.3
        )

        # Adjustment = 0.3 * (0.7 - 0.5) = 0.06
        # A: raw=1.0 -> adj=1.0 (capped)
        # B: raw=0.0 -> adj=0.06
        a_row = result.filter(pl.col("user_pseudo_id") == "A")
        b_row = result.filter(pl.col("user_pseudo_id") == "B")

        self.assertEqual(a_row["adjusted_percentile"][0], 1.0)
        self.assertAlmostEqual(b_row["adjusted_percentile"][0], 0.06, places=5)

    def test_capped_to_zero(self):
        """Test that adjusted percentiles don't go below 0."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
        ])

        data = compute_percentiles(data)

        # Very low field strength would push B's percentile negative
        field_strengths = {(2022, 1, "GP"): 0.1}

        result = compute_adjusted_percentiles(
            data, field_strengths, reference_strength=0.5, k=1.0
        )

        # Adjustment = 1.0 * (0.1 - 0.5) = -0.4
        # B: raw=0.0 -> adj=max(0, 0.0-0.4) = 0.0
        b_row = result.filter(pl.col("user_pseudo_id") == "B")
        self.assertEqual(b_row["adjusted_percentile"][0], 0.0)

    def test_capped_to_one(self):
        """Test that adjusted percentiles don't exceed 1."""
        data = pl.DataFrame([
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
        ])

        data = compute_percentiles(data)

        # Very high field strength
        field_strengths = {(2022, 1, "GP"): 0.9}

        result = compute_adjusted_percentiles(
            data, field_strengths, reference_strength=0.5, k=1.0
        )

        # Adjustment = 1.0 * (0.9 - 0.5) = 0.4
        # A: raw=1.0 -> adj=min(1.0, 1.0+0.4) = 1.0
        a_row = result.filter(pl.col("user_pseudo_id") == "A")
        self.assertEqual(a_row["adjusted_percentile"][0], 1.0)


class TestComputePercentileRatings(unittest.TestCase):
    """Test the compute_percentile_ratings function."""

    def test_basic_iteration(self):
        """Test that ratings converge with iteration."""
        data = pl.DataFrame([
            # Round 1
            {"user_pseudo_id": "A", "year": 2022, "round": 1, "competition": "GP", "points": 100},
            {"user_pseudo_id": "B", "year": 2022, "round": 1, "competition": "GP", "points": 50},
            # Round 2
            {"user_pseudo_id": "A", "year": 2022, "round": 2, "competition": "GP", "points": 90},
            {"user_pseudo_id": "B", "year": 2022, "round": 2, "competition": "GP", "points": 60},
            # Round 3
            {"user_pseudo_id": "A", "year": 2022, "round": 3, "competition": "GP", "points": 95},
            {"user_pseudo_id": "B", "year": 2022, "round": 3, "competition": "GP", "points": 55},
        ])

        ratings, final_data = compute_percentile_ratings(data, n_iterations=3, min_history=1)

        # A consistently wins, should have higher rating
        self.assertGreater(ratings["A"], ratings["B"])

        # Ratings should be in [0, 1] range
        self.assertGreaterEqual(ratings["A"], 0.0)
        self.assertLessEqual(ratings["A"], 1.0)
        self.assertGreaterEqual(ratings["B"], 0.0)
        self.assertLessEqual(ratings["B"], 1.0)


if __name__ == "__main__":
    unittest.main()
