"""Tests for competition_results.py data normalization functions."""

import unittest

import polars as pl

from ratings.competition_results import (
    GP_2014_2015_SCALE_FACTOR,
    melt_by_columns,
    normalize_table_gp,
    normalize_table_wsc,
    normalize_gp_scoring_scale,
    normalize_all_tables,
    fetch_participant_records,
)
from ratings.competitions import CompetitionIdentifier, all_gp_round_names, all_wsc_round_names


def create_full_gp_table(rows: list[dict]) -> pl.DataFrame:
    """Create a GP table with all required round columns.

    Args:
        rows: List of dicts with user_pseudo_id, year, and optional round points.
              Round points should be keyed like "GP_t1 points", "GP_t2 points", etc.
    """
    all_round_cols = all_gp_round_names()
    base_data = {"user_pseudo_id": [], "year": []}
    for col in all_round_cols:
        base_data[col] = []

    for row in rows:
        base_data["user_pseudo_id"].append(row["user_pseudo_id"])
        base_data["year"].append(row["year"])
        for col in all_round_cols:
            base_data[col].append(row.get(col, None))

    return pl.DataFrame(base_data)


def create_full_wsc_table(rows: list[dict]) -> pl.DataFrame:
    """Create a WSC table with all required round columns.

    Args:
        rows: List of dicts with user_pseudo_id, year, and optional round points.
              Round points should be keyed like "WSC_t1 points", "WSC_t2 points", etc.
    """
    all_round_cols = all_wsc_round_names()
    base_data = {"user_pseudo_id": [], "year": []}
    for col in all_round_cols:
        base_data[col] = []

    for row in rows:
        base_data["user_pseudo_id"].append(row["user_pseudo_id"])
        base_data["year"].append(row["year"])
        for col in all_round_cols:
            base_data[col].append(row.get(col, None))

    return pl.DataFrame(base_data)


class TestMeltByColumns(unittest.TestCase):
    """Test the melt_by_columns function."""

    def test_gp_basic_melt(self):
        """Test basic GP table melting."""
        gp_table = create_full_gp_table([
            {"user_pseudo_id": "Alice", "year": 2022, "GP_t1 points": 100.0, "GP_t2 points": 90.0},
            {"user_pseudo_id": "Bob", "year": 2022, "GP_t1 points": 80.0, "GP_t2 points": 85.0},
        ])

        result = melt_by_columns(gp_table, "GP")

        # Should have 4 rows (2 solvers x 2 rounds with data)
        self.assertEqual(len(result), 4)

        # Check columns
        self.assertIn("user_pseudo_id", result.columns)
        self.assertIn("year", result.columns)
        self.assertIn("round", result.columns)
        self.assertIn("points", result.columns)
        self.assertIn("competition", result.columns)

        # All competition values should be "GP"
        self.assertTrue(all(c == "GP" for c in result["competition"].to_list()))

        # Check round extraction
        rounds = sorted(result["round"].unique().to_list())
        self.assertEqual(rounds, [1, 2])

    def test_wsc_basic_melt(self):
        """Test basic WSC table melting."""
        wsc_table = create_full_wsc_table([
            {"user_pseudo_id": "Alice", "year": 2022, "WSC_t1 points": 200.0, "WSC_t2 points": 190.0},
            {"user_pseudo_id": "Bob", "year": 2022, "WSC_t1 points": 180.0},  # Bob didn't participate in R2
        ])

        result = melt_by_columns(wsc_table, "WSC")

        # Should have 3 rows (Bob's R2 is filtered out due to null)
        self.assertEqual(len(result), 3)

        # All competition values should be "WSC"
        self.assertTrue(all(c == "WSC" for c in result["competition"].to_list()))

    def test_null_values_filtered(self):
        """Test that null point values are filtered out."""
        table = create_full_gp_table([
            {"user_pseudo_id": "Alice", "year": 2022, "GP_t1 points": 100.0},  # Only R1
            {"user_pseudo_id": "Bob", "year": 2022},  # No rounds
            {"user_pseudo_id": "Charlie", "year": 2022, "GP_t1 points": 80.0, "GP_t2 points": 70.0},
        ])

        result = melt_by_columns(table, "GP")

        # Should filter out null values
        # Alice R1 (100), Charlie R1 (80), Charlie R2 (70)
        self.assertEqual(len(result), 3)

        # Verify Alice only has R1
        alice_rounds = result.filter(pl.col("user_pseudo_id") == "Alice")["round"].to_list()
        self.assertEqual(alice_rounds, [1])


class TestNormalizeTableGP(unittest.TestCase):
    """Test the normalize_table_gp function."""

    def test_adds_gp_competition_tag(self):
        """Test that GP tables get the GP competition tag."""
        gp_table = create_full_gp_table([
            {"user_pseudo_id": "Alice", "year": 2022, "GP_t1 points": 100.0},
        ])

        result = normalize_table_gp(gp_table)

        self.assertEqual(result["competition"].to_list()[0], "GP")


class TestNormalizeTableWSC(unittest.TestCase):
    """Test the normalize_table_wsc function."""

    def test_adds_wsc_competition_tag(self):
        """Test that WSC tables get the WSC competition tag."""
        wsc_table = create_full_wsc_table([
            {"user_pseudo_id": "Alice", "year": 2022, "WSC_t1 points": 200.0},
        ])

        result = normalize_table_wsc(wsc_table)

        self.assertEqual(result["competition"].to_list()[0], "WSC")


class TestNormalizeGPScoringScale(unittest.TestCase):
    """Test the normalize_gp_scoring_scale function."""

    def test_scales_2014_gp_scores(self):
        """Test that 2014 GP scores are scaled up."""
        df = pl.DataFrame({
            "year": [2014, 2014],
            "competition": ["GP", "GP"],
            "points": [100.0, 50.0],
        })

        result = normalize_gp_scoring_scale(df)
        points = result["points"].to_list()

        self.assertAlmostEqual(points[0], 100.0 * GP_2014_2015_SCALE_FACTOR)
        self.assertAlmostEqual(points[1], 50.0 * GP_2014_2015_SCALE_FACTOR)

    def test_scales_2015_gp_scores(self):
        """Test that 2015 GP scores are scaled up."""
        df = pl.DataFrame({
            "year": [2015],
            "competition": ["GP"],
            "points": [80.0],
        })

        result = normalize_gp_scoring_scale(df)

        self.assertAlmostEqual(result["points"][0], 80.0 * GP_2014_2015_SCALE_FACTOR)

    def test_does_not_scale_2016_plus(self):
        """Test that 2016+ GP scores are not scaled."""
        df = pl.DataFrame({
            "year": [2016, 2020, 2024],
            "competition": ["GP", "GP", "GP"],
            "points": [800.0, 750.0, 900.0],
        })

        result = normalize_gp_scoring_scale(df)
        points = result["points"].to_list()

        self.assertEqual(points, [800.0, 750.0, 900.0])

    def test_does_not_scale_wsc_scores(self):
        """Test that WSC scores are never scaled regardless of year."""
        df = pl.DataFrame({
            "year": [2014, 2015, 2022],
            "competition": ["WSC", "WSC", "WSC"],
            "points": [100.0, 200.0, 300.0],
        })

        result = normalize_gp_scoring_scale(df)
        points = result["points"].to_list()

        self.assertEqual(points, [100.0, 200.0, 300.0])

    def test_mixed_data(self):
        """Test with mixed GP and WSC data across years."""
        df = pl.DataFrame({
            "year": [2014, 2014, 2016, 2016],
            "competition": ["GP", "WSC", "GP", "WSC"],
            "points": [100.0, 100.0, 800.0, 400.0],
        })

        result = normalize_gp_scoring_scale(df)
        points = result["points"].to_list()

        # 2014 GP scaled, 2014 WSC not, 2016 GP not, 2016 WSC not
        expected = [
            100.0 * GP_2014_2015_SCALE_FACTOR,
            100.0,
            800.0,
            400.0,
        ]
        self.assertEqual(points, expected)


class TestNormalizeAllTables(unittest.TestCase):
    """Test the normalize_all_tables function."""

    def test_combines_gp_and_wsc(self):
        """Test that GP and WSC tables are combined correctly."""
        gp_table = create_full_gp_table([
            {"user_pseudo_id": "Alice", "year": 2022, "GP_t1 points": 800.0, "GP_t2 points": 850.0},
            {"user_pseudo_id": "Bob", "year": 2022, "GP_t1 points": 700.0, "GP_t2 points": 750.0},
        ])

        wsc_table = create_full_wsc_table([
            {"user_pseudo_id": "Alice", "year": 2022, "WSC_t1 points": 400.0},
            {"user_pseudo_id": "Charlie", "year": 2022, "WSC_t1 points": 350.0},
        ])

        result = normalize_all_tables(gp_table, wsc_table)

        # GP: 2 solvers x 2 rounds = 4
        # WSC: 2 solvers x 1 round = 2
        # Total: 6
        self.assertEqual(len(result), 6)

        # Check competition types
        gp_rows = result.filter(pl.col("competition") == "GP")
        wsc_rows = result.filter(pl.col("competition") == "WSC")
        self.assertEqual(len(gp_rows), 4)
        self.assertEqual(len(wsc_rows), 2)

    def test_applies_scoring_scale(self):
        """Test that 2014-2015 GP scoring scale is applied."""
        gp_table = create_full_gp_table([
            {"user_pseudo_id": "Alice", "year": 2014, "GP_t1 points": 100.0},
        ])

        wsc_table = create_full_wsc_table([
            {"user_pseudo_id": "Alice", "year": 2014, "WSC_t1 points": 100.0},
        ])

        result = normalize_all_tables(gp_table, wsc_table)

        gp_points = result.filter(pl.col("competition") == "GP")["points"][0]
        wsc_points = result.filter(pl.col("competition") == "WSC")["points"][0]

        # GP should be scaled, WSC should not
        self.assertAlmostEqual(gp_points, 100.0 * GP_2014_2015_SCALE_FACTOR)
        self.assertEqual(wsc_points, 100.0)


class TestFetchParticipantRecords(unittest.TestCase):
    """Test the fetch_participant_records function."""

    def setUp(self):
        """Create a test DataFrame with normalized results."""
        self.result_table = pl.DataFrame({
            "user_pseudo_id": [
                "Alice", "Bob", "Alice", "Bob",  # 2022 GP R1, R2
                "Alice", "Charlie",  # 2022 WSC R1
                "Alice", "Bob",  # 2023 GP R1
            ],
            "year": [2022, 2022, 2022, 2022, 2022, 2022, 2023, 2023],
            "round": [1, 1, 2, 2, 1, 1, 1, 1],
            "competition": ["GP", "GP", "GP", "GP", "WSC", "WSC", "GP", "GP"],
            "points": [800.0, 700.0, 850.0, 750.0, 400.0, 350.0, 900.0, 800.0],
        })

    def test_fetch_single_competition(self):
        """Test fetching records for a single competition."""
        events = [CompetitionIdentifier(2022, 1, "GP")]
        result = fetch_participant_records(events, self.result_table)

        self.assertEqual(len(result), 2)  # Alice and Bob
        self.assertTrue(all(r == 1 for r in result["round"].to_list()))
        self.assertTrue(all(c == "GP" for c in result["competition"].to_list()))

    def test_fetch_multiple_competitions(self):
        """Test fetching records for multiple competitions."""
        events = [
            CompetitionIdentifier(2022, 1, "GP"),
            CompetitionIdentifier(2022, 2, "GP"),
        ]
        result = fetch_participant_records(events, self.result_table)

        self.assertEqual(len(result), 4)  # 2 solvers x 2 rounds

    def test_fetch_cross_event_type(self):
        """Test fetching records across GP and WSC."""
        events = [
            CompetitionIdentifier(2022, 1, "GP"),
            CompetitionIdentifier(2022, 1, "WSC"),
        ]
        result = fetch_participant_records(events, self.result_table)

        self.assertEqual(len(result), 4)  # GP: Alice+Bob, WSC: Alice+Charlie

        gp_rows = result.filter(pl.col("competition") == "GP")
        wsc_rows = result.filter(pl.col("competition") == "WSC")
        self.assertEqual(len(gp_rows), 2)
        self.assertEqual(len(wsc_rows), 2)

    def test_fetch_nonexistent_competition(self):
        """Test fetching a competition that doesn't exist in the data."""
        events = [CompetitionIdentifier(2025, 1, "GP")]
        result = fetch_participant_records(events, self.result_table)

        # Should return an empty DataFrame (or None, depending on implementation)
        # The current implementation returns None if no records found
        self.assertTrue(result is None or len(result) == 0)

    def test_preserves_all_columns(self):
        """Test that all columns from the input are preserved."""
        events = [CompetitionIdentifier(2022, 1, "GP")]
        result = fetch_participant_records(events, self.result_table)

        for col in ["user_pseudo_id", "year", "round", "competition", "points"]:
            self.assertIn(col, result.columns)


if __name__ == "__main__":
    unittest.main()
