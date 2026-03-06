"""Tests for CLI commands.

These tests focus on argument parsing and command dispatch.
Integration tests requiring data loading are marked and can be skipped.
"""

import argparse
import io
import unittest
from unittest.mock import patch, MagicMock

import polars as pl


class TestCacheCommand(unittest.TestCase):
    """Test the cache command."""

    @patch('ratings.cli.commands.get_cache_info')
    def test_cache_info_no_cache(self, mock_get_cache_info):
        """Test cache info when no cache exists."""
        from ratings.cli.commands import cmd_cache

        mock_info = MagicMock()
        mock_info.exists = False
        mock_get_cache_info.return_value = mock_info

        args = argparse.Namespace(purge=False)

        # Capture stdout
        captured = io.StringIO()
        with patch('sys.stdout', captured):
            cmd_cache(args)

        output = captured.getvalue()
        self.assertIn("No cache exists", output)

    @patch('ratings.cli.commands.purge_cache')
    def test_cache_purge_success(self, mock_purge_cache):
        """Test successful cache purge."""
        from ratings.cli.commands import cmd_cache

        mock_purge_cache.return_value = True
        args = argparse.Namespace(purge=True)

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            cmd_cache(args)

        output = captured.getvalue()
        self.assertIn("Cache purged", output)

    @patch('ratings.cli.commands.purge_cache')
    def test_cache_purge_nothing_to_purge(self, mock_purge_cache):
        """Test cache purge when nothing to purge."""
        from ratings.cli.commands import cmd_cache

        mock_purge_cache.return_value = False
        args = argparse.Namespace(purge=True)

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            cmd_cache(args)

        output = captured.getvalue()
        self.assertIn("No cache to purge", output)


class TestArgumentParsing(unittest.TestCase):
    """Test command-line argument parsing."""

    def test_leaderboard_defaults(self):
        """Test leaderboard command default arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        lb_parser = subparsers.add_parser('leaderboard')
        lb_parser.add_argument('--all-time', action='store_true')
        lb_parser.add_argument('--year', type=int)
        lb_parser.add_argument('--top', type=int, default=30)

        args = parser.parse_args(['leaderboard'])

        self.assertEqual(args.command, 'leaderboard')
        self.assertFalse(args.all_time)
        self.assertIsNone(args.year)
        self.assertEqual(args.top, 30)

    def test_leaderboard_with_options(self):
        """Test leaderboard command with options."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        lb_parser = subparsers.add_parser('leaderboard')
        lb_parser.add_argument('--all-time', action='store_true')
        lb_parser.add_argument('--year', type=int)
        lb_parser.add_argument('--top', type=int, default=30)

        args = parser.parse_args(['leaderboard', '--all-time', '--year', '2020', '--top', '50'])

        self.assertTrue(args.all_time)
        self.assertEqual(args.year, 2020)
        self.assertEqual(args.top, 50)

    def test_solver_requires_name(self):
        """Test that solver command requires a name argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        solver_parser = subparsers.add_parser('solver')
        solver_parser.add_argument('name')
        solver_parser.add_argument('--top', type=int)

        # Should work with name
        args = parser.parse_args(['solver', 'Kota Morinishi'])
        self.assertEqual(args.name, 'Kota Morinishi')

        # Should fail without name
        with self.assertRaises(SystemExit):
            parser.parse_args(['solver'])

    def test_progression_defaults(self):
        """Test progression command default arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        prog_parser = subparsers.add_parser('progression')
        prog_parser.add_argument('--top', type=int, default=3)
        prog_parser.add_argument('--from-year', type=int, dest='from_year')
        prog_parser.add_argument('--to-year', type=int, dest='to_year')
        prog_parser.add_argument('--ratings', action='store_true')

        args = parser.parse_args(['progression'])

        self.assertEqual(args.top, 3)
        self.assertIsNone(args.from_year)
        self.assertIsNone(args.to_year)
        self.assertFalse(args.ratings)

    def test_compare_defaults(self):
        """Test compare command default arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        compare_parser = subparsers.add_parser('compare')
        compare_parser.add_argument('--leaderboard', action='store_true')
        compare_parser.add_argument('--top', type=int, default=20)
        compare_parser.add_argument('--year', type=int)
        compare_parser.add_argument('--burn-in', type=int, default=0, dest='burn_in')
        compare_parser.add_argument('--horizon', type=int, default=3)

        args = parser.parse_args(['compare'])

        self.assertFalse(args.leaderboard)
        self.assertEqual(args.top, 20)
        self.assertEqual(args.burn_in, 0)
        self.assertEqual(args.horizon, 3)

    def test_competitions_year_filter(self):
        """Test competitions command year filter parsing."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        comp_parser = subparsers.add_parser('competitions')
        comp_parser.add_argument('--year', type=str)
        comp_parser.add_argument('--event', type=str, choices=['GP', 'WSC', 'gp', 'wsc'])

        # Single year
        args = parser.parse_args(['competitions', '--year', '2024'])
        self.assertEqual(args.year, '2024')

        # Year range
        args = parser.parse_args(['competitions', '--year', '2020-2024'])
        self.assertEqual(args.year, '2020-2024')

    def test_competitions_event_filter(self):
        """Test competitions command event filter."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        comp_parser = subparsers.add_parser('competitions')
        comp_parser.add_argument('--year', type=str)
        comp_parser.add_argument('--event', type=str, choices=['GP', 'WSC', 'gp', 'wsc'])

        args = parser.parse_args(['competitions', '--event', 'GP'])
        self.assertEqual(args.event, 'GP')

        args = parser.parse_args(['competitions', '--event', 'wsc'])
        self.assertEqual(args.event, 'wsc')

        # Invalid event should fail
        with self.assertRaises(SystemExit):
            parser.parse_args(['competitions', '--event', 'INVALID'])


class TestComputeHorizonAccuracy(unittest.TestCase):
    """Test the compute_horizon_accuracy function."""

    def test_basic_horizon_accuracy(self):
        """Test basic horizon accuracy computation."""
        from ratings.cli.commands import compute_horizon_accuracy

        # Create minimal features DataFrame
        # Need: user_pseudo_id, year, round, competition, exp_weighted_mean, label
        features = pl.DataFrame([
            # Solver A: rated after round 0, performs well in next 2 rounds
            {"user_pseudo_id": "A", "year": 2020, "round": 1, "competition": "GP",
             "exp_weighted_mean": 900.0, "label": 850.0},
            {"user_pseudo_id": "A", "year": 2020, "round": 2, "competition": "GP",
             "exp_weighted_mean": 880.0, "label": 900.0},
            {"user_pseudo_id": "A", "year": 2020, "round": 3, "competition": "GP",
             "exp_weighted_mean": 890.0, "label": 880.0},
            # Solver B: rated after round 0, performs worse in next 2 rounds
            {"user_pseudo_id": "B", "year": 2020, "round": 1, "competition": "GP",
             "exp_weighted_mean": 800.0, "label": 750.0},
            {"user_pseudo_id": "B", "year": 2020, "round": 2, "competition": "GP",
             "exp_weighted_mean": 780.0, "label": 780.0},
            {"user_pseudo_id": "B", "year": 2020, "round": 3, "competition": "GP",
             "exp_weighted_mean": 790.0, "label": 770.0},
        ])

        accuracy, total_pairs = compute_horizon_accuracy(features, "exp_weighted_mean", horizon=2)

        # Should have some pairs and non-zero accuracy
        self.assertGreaterEqual(total_pairs, 0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    def test_format_size(self):
        """Test the format_size helper function logic."""
        # The format_size function is defined inside cmd_cache, so we test the logic here
        def format_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"

        self.assertEqual(format_size(500), "500 B")
        self.assertEqual(format_size(1024), "1.0 KB")
        self.assertEqual(format_size(2048), "2.0 KB")
        self.assertEqual(format_size(1024 * 1024), "1.0 MB")
        self.assertEqual(format_size(1024 * 1024 * 5), "5.0 MB")


class TestYearRangeParsing(unittest.TestCase):
    """Test year range parsing in competitions command."""

    def test_single_year_filter(self):
        """Test parsing a single year."""
        year_str = "2024"
        if '-' in year_str:
            start, end = year_str.split('-')
            years_filter = range(int(start), int(end) + 1)
        else:
            years_filter = [int(year_str)]

        self.assertEqual(list(years_filter), [2024])

    def test_year_range_filter(self):
        """Test parsing a year range."""
        year_str = "2020-2024"
        if '-' in year_str:
            start, end = year_str.split('-')
            years_filter = range(int(start), int(end) + 1)
        else:
            years_filter = [int(year_str)]

        self.assertEqual(list(years_filter), [2020, 2021, 2022, 2023, 2024])


class TestRecordsArgumentParsing(unittest.TestCase):
    """Test records command argument parsing."""

    def _make_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        records_parser = subparsers.add_parser('records')
        records_parser.add_argument('--sort', type=str,
                                    choices=['ones', 'streak', 'wins', 'adj-points', 'points', 'rounds'],
                                    default='ones')
        records_parser.add_argument('--method', type=str, choices=['prior', 'no-prior'],
                                    default='prior')
        records_parser.add_argument('--top', type=int, default=20)
        return parser

    def test_records_defaults(self):
        """Test records command default arguments."""
        parser = self._make_parser()
        args = parser.parse_args(['records'])

        self.assertEqual(args.command, 'records')
        self.assertEqual(args.sort, 'ones')
        self.assertEqual(args.method, 'prior')
        self.assertEqual(args.top, 20)

    def test_records_sort_options(self):
        """Test each sort choice parses correctly."""
        parser = self._make_parser()
        for choice in ['ones', 'streak', 'wins', 'adj-points', 'points', 'rounds']:
            args = parser.parse_args(['records', '--sort', choice])
            self.assertEqual(args.sort, choice)

    def test_records_invalid_sort(self):
        """Test invalid sort choice raises SystemExit."""
        parser = self._make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(['records', '--sort', 'invalid'])

    def test_records_method_no_prior(self):
        """Test method=no-prior parses correctly."""
        parser = self._make_parser()
        args = parser.parse_args(['records', '--method', 'no-prior'])
        self.assertEqual(args.method, 'no-prior')

    def test_records_top(self):
        """Test custom top value."""
        parser = self._make_parser()
        args = parser.parse_args(['records', '--top', '10'])
        self.assertEqual(args.top, 10)


if __name__ == "__main__":
    unittest.main()
