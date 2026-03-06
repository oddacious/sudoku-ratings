"""Tests for leaderboard.py ranking logic."""

import unittest

import polars as pl

from ratings.leaderboard import (
    INACTIVITY_THRESHOLD_YEARS,
    LeaderboardEntry,
    PercentileLeaderboardEntry,
    generate_leaderboard,
    generate_leaderboard_after_round,
    generate_percentile_leaderboard,
)
from ratings.competitions import CompetitionIdentifier


def create_mock_features_df(rows: list[dict]) -> pl.DataFrame:
    """Create a mock features DataFrame for testing.

    Args:
        rows: List of dicts with keys:
            - user_pseudo_id
            - year
            - round
            - competition (event_type)
            - exp_weighted_mean
            - mean_adj_points
            - n_rounds
    """
    return pl.DataFrame(rows)


def create_mock_percentile_features_df(rows: list[dict]) -> pl.DataFrame:
    """Create a mock percentile features DataFrame for testing.

    Args:
        rows: List of dicts with keys:
            - user_pseudo_id
            - year
            - round
            - competition (event_type)
            - exp_weighted_pct
            - mean_adj_percentile
            - n_rounds
    """
    return pl.DataFrame(rows)


class TestGenerateLeaderboard(unittest.TestCase):
    """Test the generate_leaderboard function."""

    def test_basic_ranking(self):
        """Test basic ranking by exp_weighted_mean."""
        features = create_mock_features_df([
            {"user_pseudo_id": "Alice", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 900.0, "mean_adj_points": 850.0, "n_rounds": 20},
            {"user_pseudo_id": "Bob", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 850.0, "mean_adj_points": 800.0, "n_rounds": 15},
            {"user_pseudo_id": "Charlie", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 950.0, "mean_adj_points": 920.0, "n_rounds": 25},
        ])

        result = generate_leaderboard(features, top_n=3, as_of_year=2024)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].rank, 1)
        self.assertEqual(result[0].solver_id, "Charlie")  # Highest exp_weighted_mean
        self.assertEqual(result[1].solver_id, "Alice")
        self.assertEqual(result[2].solver_id, "Bob")

    def test_top_n_limits_results(self):
        """Test that top_n limits the number of results."""
        features = create_mock_features_df([
            {"user_pseudo_id": f"Solver{i}", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 1000 - i * 10, "mean_adj_points": 900.0, "n_rounds": 10}
            for i in range(10)
        ])

        result = generate_leaderboard(features, top_n=3, as_of_year=2024)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].solver_id, "Solver0")  # Highest

    def test_inactivity_filter(self):
        """Test that inactive solvers are filtered out."""
        # 2024 leaderboard should exclude solvers last active before 2023
        features = create_mock_features_df([
            {"user_pseudo_id": "Active", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 800.0, "mean_adj_points": 750.0, "n_rounds": 10},
            {"user_pseudo_id": "Recent", "year": 2023, "round": 8, "competition": "GP",
             "exp_weighted_mean": 900.0, "mean_adj_points": 850.0, "n_rounds": 20},
            {"user_pseudo_id": "Inactive", "year": 2022, "round": 8, "competition": "GP",
             "exp_weighted_mean": 950.0, "mean_adj_points": 920.0, "n_rounds": 50},
        ])

        result = generate_leaderboard(features, top_n=10, as_of_year=2024)

        solver_ids = [e.solver_id for e in result]
        self.assertIn("Active", solver_ids)
        self.assertIn("Recent", solver_ids)
        # Inactive should be excluded (2022 is more than INACTIVITY_THRESHOLD_YEARS ago)
        self.assertNotIn("Inactive", solver_ids)

    def test_uses_latest_record_per_solver(self):
        """Test that the latest record is used for each solver."""
        features = create_mock_features_df([
            {"user_pseudo_id": "Alice", "year": 2023, "round": 1, "competition": "GP",
             "exp_weighted_mean": 700.0, "mean_adj_points": 650.0, "n_rounds": 5},
            {"user_pseudo_id": "Alice", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 850.0, "mean_adj_points": 800.0, "n_rounds": 10},
            {"user_pseudo_id": "Alice", "year": 2024, "round": 3, "competition": "GP",
             "exp_weighted_mean": 900.0, "mean_adj_points": 850.0, "n_rounds": 12},
        ])

        result = generate_leaderboard(features, top_n=1, as_of_year=2024)

        self.assertEqual(len(result), 1)
        # Should use the 2024 R3 record (latest)
        self.assertEqual(result[0].exp_weighted_mean, 900.0)
        self.assertEqual(result[0].n_rounds, 12)

    def test_as_of_year_filters_future_data(self):
        """Test that as_of_year prevents future data leakage."""
        features = create_mock_features_df([
            {"user_pseudo_id": "Alice", "year": 2022, "round": 8, "competition": "GP",
             "exp_weighted_mean": 800.0, "mean_adj_points": 750.0, "n_rounds": 10},
            {"user_pseudo_id": "Alice", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 950.0, "mean_adj_points": 900.0, "n_rounds": 20},
        ])

        result = generate_leaderboard(features, top_n=1, as_of_year=2022)

        # Should use 2022 data, not 2024
        self.assertEqual(result[0].exp_weighted_mean, 800.0)

    def test_leaderboard_entry_fields(self):
        """Test that LeaderboardEntry has correct fields."""
        features = create_mock_features_df([
            {"user_pseudo_id": "Alice", "year": 2024, "round": 5, "competition": "GP",
             "exp_weighted_mean": 900.0, "mean_adj_points": 850.0, "n_rounds": 20},
        ])

        result = generate_leaderboard(features, top_n=1, as_of_year=2024)
        entry = result[0]

        self.assertEqual(entry.rank, 1)
        self.assertEqual(entry.solver_id, "Alice")
        self.assertEqual(entry.exp_weighted_mean, 900.0)
        self.assertEqual(entry.mean_adj_points, 850.0)
        self.assertEqual(entry.n_rounds, 20)
        self.assertEqual(entry.last_active_year, 2024)


class TestGenerateLeaderboardAfterRound(unittest.TestCase):
    """Test the generate_leaderboard_after_round function."""

    def test_limits_to_competition(self):
        """Test that only data up to specified competition is included."""
        features = create_mock_features_df([
            {"user_pseudo_id": "Alice", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 800.0, "mean_adj_points": 750.0, "n_rounds": 10},
            {"user_pseudo_id": "Alice", "year": 2024, "round": 3, "competition": "GP",
             "exp_weighted_mean": 850.0, "mean_adj_points": 800.0, "n_rounds": 12},
            {"user_pseudo_id": "Alice", "year": 2024, "round": 5, "competition": "GP",
             "exp_weighted_mean": 900.0, "mean_adj_points": 850.0, "n_rounds": 14},
        ])

        comp = CompetitionIdentifier(2024, 3, "GP")
        result = generate_leaderboard_after_round(features, comp, top_n=1)

        # Should use R3 data, not R5
        if result:  # Only if competition exists in schedule
            self.assertLessEqual(result[0].exp_weighted_mean, 850.0)

    def test_returns_empty_for_invalid_competition(self):
        """Test that invalid competition returns empty list."""
        features = create_mock_features_df([
            {"user_pseudo_id": "Alice", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 800.0, "mean_adj_points": 750.0, "n_rounds": 10},
        ])

        # Nonexistent competition
        comp = CompetitionIdentifier(1990, 1, "GP")
        result = generate_leaderboard_after_round(features, comp, top_n=1)

        self.assertEqual(result, [])


class TestGeneratePercentileLeaderboard(unittest.TestCase):
    """Test the generate_percentile_leaderboard function."""

    def test_basic_percentile_ranking(self):
        """Test basic ranking by exp_weighted_pct."""
        features = create_mock_percentile_features_df([
            {"user_pseudo_id": "Alice", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_pct": 0.85, "mean_adj_percentile": 0.80, "n_rounds": 20},
            {"user_pseudo_id": "Bob", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_pct": 0.92, "mean_adj_percentile": 0.88, "n_rounds": 15},
            {"user_pseudo_id": "Charlie", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_pct": 0.78, "mean_adj_percentile": 0.75, "n_rounds": 25},
        ])

        result = generate_percentile_leaderboard(features, top_n=3, as_of_year=2024)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].rank, 1)
        self.assertEqual(result[0].solver_id, "Bob")  # Highest exp_weighted_pct
        self.assertEqual(result[1].solver_id, "Alice")
        self.assertEqual(result[2].solver_id, "Charlie")

    def test_percentile_entry_fields(self):
        """Test that PercentileLeaderboardEntry has correct fields."""
        features = create_mock_percentile_features_df([
            {"user_pseudo_id": "Alice", "year": 2024, "round": 5, "competition": "GP",
             "exp_weighted_pct": 0.92, "mean_adj_percentile": 0.88, "n_rounds": 20},
        ])

        result = generate_percentile_leaderboard(features, top_n=1, as_of_year=2024)
        entry = result[0]

        self.assertIsInstance(entry, PercentileLeaderboardEntry)
        self.assertEqual(entry.rank, 1)
        self.assertEqual(entry.solver_id, "Alice")
        self.assertAlmostEqual(entry.exp_weighted_pct, 0.92)
        self.assertAlmostEqual(entry.mean_adj_percentile, 0.88)
        self.assertEqual(entry.n_rounds, 20)
        self.assertEqual(entry.last_active_year, 2024)

    def test_inactivity_filter_percentile(self):
        """Test that inactive solvers are filtered from percentile leaderboard."""
        features = create_mock_percentile_features_df([
            {"user_pseudo_id": "Active", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_pct": 0.80, "mean_adj_percentile": 0.75, "n_rounds": 10},
            {"user_pseudo_id": "Inactive", "year": 2020, "round": 8, "competition": "GP",
             "exp_weighted_pct": 0.95, "mean_adj_percentile": 0.92, "n_rounds": 50},
        ])

        result = generate_percentile_leaderboard(features, top_n=10, as_of_year=2024)

        solver_ids = [e.solver_id for e in result]
        self.assertIn("Active", solver_ids)
        self.assertNotIn("Inactive", solver_ids)


class TestInactivityThreshold(unittest.TestCase):
    """Test the inactivity threshold constant."""

    def test_threshold_value(self):
        """Test that the inactivity threshold is set correctly."""
        # Based on CLAUDE.md, this should be 1 year
        self.assertEqual(INACTIVITY_THRESHOLD_YEARS, 1)

    def test_threshold_boundary(self):
        """Test solvers exactly at the inactivity boundary."""
        # For 2024 leaderboard with threshold=1:
        # - 2024 active: included
        # - 2023 active: included (within 1 year)
        # - 2022 active: excluded (more than 1 year)
        features = create_mock_features_df([
            {"user_pseudo_id": "Y2024", "year": 2024, "round": 1, "competition": "GP",
             "exp_weighted_mean": 800.0, "mean_adj_points": 750.0, "n_rounds": 10},
            {"user_pseudo_id": "Y2023", "year": 2023, "round": 8, "competition": "GP",
             "exp_weighted_mean": 850.0, "mean_adj_points": 800.0, "n_rounds": 15},
            {"user_pseudo_id": "Y2022", "year": 2022, "round": 8, "competition": "GP",
             "exp_weighted_mean": 900.0, "mean_adj_points": 850.0, "n_rounds": 20},
        ])

        result = generate_leaderboard(features, top_n=10, as_of_year=2024)
        solver_ids = [e.solver_id for e in result]

        self.assertIn("Y2024", solver_ids)
        self.assertIn("Y2023", solver_ids)
        self.assertNotIn("Y2022", solver_ids)


if __name__ == "__main__":
    unittest.main()
