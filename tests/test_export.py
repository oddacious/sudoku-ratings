"""Tests for ratings.export module.

These tests verify the export functions produce correct output schemas
and consistent data.
"""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from ratings.competitions import CompetitionIdentifier
from ratings.export import (
    export_timeseries,
    export_current_leaderboard,
    export_alltime_leaderboard,
    export_records,
    write_metadata,
    write_dataframe,
    run_export,
)
from ratings.methods.rating_tracker import RatingTracker


class TestExportTimeseries(unittest.TestCase):
    """Test the export_timeseries function."""

    def setUp(self):
        """Create sample data for testing."""
        # Create minimal test data with a few solvers
        self.adjusted = pl.DataFrame({
            "user_pseudo_id": [
                "Solver A", "Solver B", "Solver A", "Solver B",
                "Solver A", "Solver B", "Solver A", "Solver B",
            ],
            "year": [2014] * 8,
            "round": [1, 1, 2, 2, 3, 3, 4, 4],
            "competition": ["GP"] * 8,
            "points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
            "adjusted_points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
        })
        self.normalized = self.adjusted.select([
            "user_pseudo_id", "year", "round", "competition", "points"
        ])

    def test_returns_dataframe(self):
        """Test that export_timeseries returns a DataFrame."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_timeseries(tracker, self.normalized, self.adjusted)

        self.assertIsInstance(result, pl.DataFrame)

    def test_output_columns(self):
        """Test that output has expected columns."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_timeseries(tracker, self.normalized, self.adjusted)

        expected_columns = {
            'user_pseudo_id', 'year', 'round', 'competition', 'comp_idx',
            'rating', 'n_rounds', 'rank', 'rank_total', 'raw_points', 'adjusted_points'
        }
        self.assertEqual(set(result.columns), expected_columns)

    def test_only_rated_solvers_included(self):
        """Test that only solvers with min_history are included."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_timeseries(tracker, self.normalized, self.adjusted)

        # With min_history=3, first entries should appear at round 3
        min_round = result['round'].min()
        self.assertGreaterEqual(min_round, 3)

    def test_ranks_are_valid(self):
        """Test that ranks are positive integers."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_timeseries(tracker, self.normalized, self.adjusted)

        # All ranks should be positive
        self.assertTrue((result['rank'] > 0).all())
        # Rank should not exceed rank_total
        self.assertTrue((result['rank'] <= result['rank_total']).all())


class TestExportCurrentLeaderboard(unittest.TestCase):
    """Test the export_current_leaderboard function."""

    def setUp(self):
        """Create sample data for testing."""
        self.adjusted = pl.DataFrame({
            "user_pseudo_id": [
                "Solver A", "Solver B", "Solver A", "Solver B",
                "Solver A", "Solver B", "Solver A", "Solver B",
            ],
            "year": [2014] * 8,
            "round": [1, 1, 2, 2, 3, 3, 4, 4],
            "competition": ["GP"] * 8,
            "points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
            "adjusted_points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
        })
        self.normalized = self.adjusted.select([
            "user_pseudo_id", "year", "round", "competition", "points"
        ])

    def test_returns_dataframe(self):
        """Test that export_current_leaderboard returns a DataFrame."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_current_leaderboard(tracker, self.normalized, 2014)

        self.assertIsInstance(result, pl.DataFrame)

    def test_output_columns(self):
        """Test that output has expected columns."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_current_leaderboard(tracker, self.normalized, 2014)

        expected_columns = {
            'rank', 'user_pseudo_id', 'rating', 'mean_adj_points',
            'n_rounds', 'last_year', 'last_place', 'last_round_size'
        }
        self.assertEqual(set(result.columns), expected_columns)

    def test_ranks_are_sequential(self):
        """Test that ranks are sequential starting from 1."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_current_leaderboard(tracker, self.normalized, 2014)

        if len(result) > 0:
            expected_ranks = list(range(1, len(result) + 1))
            actual_ranks = result['rank'].to_list()
            self.assertEqual(actual_ranks, expected_ranks)

    def test_sorted_by_rating(self):
        """Test that leaderboard is sorted by rating descending."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_current_leaderboard(tracker, self.normalized, 2014)

        if len(result) > 1:
            ratings = result['rating'].to_list()
            self.assertEqual(ratings, sorted(ratings, reverse=True))


class TestExportAlltimeLeaderboard(unittest.TestCase):
    """Test the export_alltime_leaderboard function."""

    def setUp(self):
        """Create sample data for testing."""
        self.adjusted = pl.DataFrame({
            "user_pseudo_id": [
                "Solver A", "Solver B", "Solver A", "Solver B",
                "Solver A", "Solver B", "Solver A", "Solver B",
            ],
            "year": [2014] * 8,
            "round": [1, 1, 2, 2, 3, 3, 4, 4],
            "competition": ["GP"] * 8,
            "points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
            "adjusted_points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
        })

    def test_returns_dataframe(self):
        """Test that export_alltime_leaderboard returns a DataFrame."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_alltime_leaderboard(tracker)

        self.assertIsInstance(result, pl.DataFrame)

    def test_output_columns(self):
        """Test that output has expected columns."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_alltime_leaderboard(tracker)

        expected_columns = {
            'rank', 'user_pseudo_id', 'peak_rating', 'peak_year',
            'peak_round', 'peak_competition', 'n_rounds'
        }
        self.assertEqual(set(result.columns), expected_columns)

    def test_one_row_per_solver(self):
        """Test that there is exactly one row per solver."""
        tracker = RatingTracker(self.adjusted, min_history=3)
        tracker.advance_to(CompetitionIdentifier(2014, 4, "GP"))

        result = export_alltime_leaderboard(tracker)

        solver_counts = result.group_by('user_pseudo_id').len()
        self.assertTrue((solver_counts['len'] == 1).all())


class TestExportRecords(unittest.TestCase):
    """Test the export_records function."""

    def setUp(self):
        """Create sample data for testing."""
        self.adjusted = pl.DataFrame({
            "user_pseudo_id": [
                "Solver A", "Solver B", "Solver A", "Solver B",
                "Solver A", "Solver B", "Solver A", "Solver B",
            ],
            "year": [2014] * 8,
            "round": [1, 1, 2, 2, 3, 3, 4, 4],
            "competition": ["GP"] * 8,
            "points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
            "adjusted_points": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0, 115.0, 80.0],
        })
        self.normalized = self.adjusted.select([
            "user_pseudo_id", "year", "round", "competition", "points"
        ])

    def test_returns_dataframe(self):
        """Test that export_records returns a DataFrame."""
        tracker = RatingTracker(self.adjusted, min_history=3)

        result = export_records(tracker, self.normalized, self.adjusted)

        self.assertIsInstance(result, pl.DataFrame)

    def test_output_columns(self):
        """Test that output has expected columns."""
        tracker = RatingTracker(self.adjusted, min_history=3)

        result = export_records(tracker, self.normalized, self.adjusted)

        expected_columns = {
            'user_pseudo_id', 'ones_count', 'best_streak', 'wins_count',
            'total_adj_points', 'total_raw_points', 'total_rounds'
        }
        self.assertEqual(set(result.columns), expected_columns)

    def test_excludes_low_round_solvers(self):
        """Test that solvers with < 3 rounds are excluded."""
        # Create data with a solver who has only 2 rounds
        data = pl.DataFrame({
            "user_pseudo_id": ["A", "A", "B", "B", "B", "B"],
            "year": [2014] * 6,
            "round": [1, 2, 1, 2, 3, 4],
            "competition": ["GP"] * 6,
            "points": [100.0, 100.0, 90.0, 90.0, 90.0, 90.0],
            "adjusted_points": [100.0, 100.0, 90.0, 90.0, 90.0, 90.0],
        })
        normalized = data.select(["user_pseudo_id", "year", "round", "competition", "points"])

        tracker = RatingTracker(data, min_history=3)
        result = export_records(tracker, normalized, data)

        # Solver A has only 2 rounds, should be excluded
        solvers = result['user_pseudo_id'].to_list()
        self.assertNotIn("A", solvers)
        self.assertIn("B", solvers)


class TestWriteMetadata(unittest.TestCase):
    """Test the write_metadata function."""

    def test_creates_json_file(self):
        """Test that write_metadata creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_metadata(output_dir, "prior", 100, "2024 GP R8")

            metadata_path = output_dir / "metadata.json"
            self.assertTrue(metadata_path.exists())

    def test_metadata_contents(self):
        """Test that metadata has expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_metadata(output_dir, "prior", 100, "2024 GP R8")

            with open(output_dir / "metadata.json") as f:
                metadata = json.load(f)

            self.assertEqual(metadata['version'], '1.0')
            self.assertEqual(metadata['method'], 'prior')
            self.assertEqual(metadata['total_solvers'], 100)
            self.assertEqual(metadata['data_through'], '2024 GP R8')
            self.assertIn('generated_at', metadata)


class TestWriteDataframe(unittest.TestCase):
    """Test the write_dataframe function."""

    def test_write_parquet(self):
        """Test writing only parquet."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_dataframe(df, output_dir, "test", "parquet")

            self.assertTrue((output_dir / "test.parquet").exists())
            self.assertFalse((output_dir / "test.csv").exists())

    def test_write_csv(self):
        """Test writing only csv."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_dataframe(df, output_dir, "test", "csv")

            self.assertFalse((output_dir / "test.parquet").exists())
            self.assertTrue((output_dir / "test.csv").exists())

    def test_write_both(self):
        """Test writing both formats."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_dataframe(df, output_dir, "test", "both")

            self.assertTrue((output_dir / "test.parquet").exists())
            self.assertTrue((output_dir / "test.csv").exists())

    def test_roundtrip_parquet(self):
        """Test that parquet data survives round-trip."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_dataframe(df, output_dir, "test", "parquet")

            loaded = pl.read_parquet(output_dir / "test.parquet")
            self.assertTrue(df.equals(loaded))


class TestExportArgumentParsing(unittest.TestCase):
    """Test export command argument parsing."""

    def _make_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        export_parser = subparsers.add_parser('export')
        export_parser.add_argument('--output-dir', type=str, default='./export/')
        export_parser.add_argument('--format', type=str,
                                   choices=['parquet', 'csv', 'both'],
                                   default='both')
        export_parser.add_argument('--method', type=str,
                                   choices=['prior', 'no-prior'],
                                   default='prior')
        return parser

    def test_export_defaults(self):
        """Test export command default arguments."""
        parser = self._make_parser()
        args = parser.parse_args(['export'])

        self.assertEqual(args.command, 'export')
        self.assertEqual(args.output_dir, './export/')
        self.assertEqual(args.format, 'both')
        self.assertEqual(args.method, 'prior')

    def test_export_output_dir(self):
        """Test custom output directory."""
        parser = self._make_parser()
        args = parser.parse_args(['export', '--output-dir', '/tmp/ratings'])

        self.assertEqual(args.output_dir, '/tmp/ratings')

    def test_export_format_options(self):
        """Test each format option parses correctly."""
        parser = self._make_parser()

        for fmt in ['parquet', 'csv', 'both']:
            args = parser.parse_args(['export', '--format', fmt])
            self.assertEqual(args.format, fmt)

    def test_export_method_options(self):
        """Test method options."""
        parser = self._make_parser()

        args = parser.parse_args(['export', '--method', 'no-prior'])
        self.assertEqual(args.method, 'no-prior')


class TestRunExportReturnsStats(unittest.TestCase):
    """Test that run_export returns expected statistics."""

    def test_returns_dict_with_expected_keys(self):
        """Test that run_export returns dict with expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use actual data for this integration test
            stats = run_export(output_dir=tmpdir, fmt='parquet', method='prior')

            expected_keys = {
                'output_dir', 'method', 'format', 'total_solvers',
                'timeseries_rows', 'current_leaderboard_rows',
                'alltime_leaderboard_rows', 'records_rows', 'data_through'
            }
            self.assertEqual(set(stats.keys()), expected_keys)

            # Verify positive counts
            self.assertGreater(stats['total_solvers'], 0)
            self.assertGreater(stats['timeseries_rows'], 0)
            self.assertGreater(stats['current_leaderboard_rows'], 0)
            self.assertGreater(stats['records_rows'], 0)


if __name__ == "__main__":
    unittest.main()
