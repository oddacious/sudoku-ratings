#!/usr/bin/env python3
"""Command-line interface for sudoku ratings.

This module handles argument parsing and routes to command implementations
in ratings.cli.commands. Run `python main.py --help` for usage.
"""

import argparse
import sys

from ratings.cli import (
    cmd_leaderboard,
    cmd_progression,
    cmd_solver,
    cmd_compare,
    cmd_cache,
    cmd_competitions,
    cmd_records,
    cmd_export,
)


def main():
    parser = argparse.ArgumentParser(
        description="Sudoku competition ratings CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py leaderboard                    Show current leaderboard
  python main.py leaderboard --all-time         Show all-time peak ratings
  python main.py leaderboard --year 2018        Show leaderboard as of 2018
  python main.py leaderboard --method no-prior  Disable prior regularization
  python main.py solver "Kota Morinishi"        Show rating progression
  python main.py solver "Tantan"                Partial name matching works
  python main.py solver "Kota" --method no-prior  Show progression without prior
  python main.py progression                    Show top 3 after each round
  python main.py progression --from-year 2020   Show progression from 2020 onward
  python main.py progression --ratings          Include ratings in output
  python main.py progression --method no-prior  Disable prior regularization
  python main.py compare                        Compare rating methods by accuracy
  python main.py compare --leaderboard          Compare leaderboards across methods
  python main.py competitions                   Show competition statistics
  python main.py competitions --year 2024       Stats for a specific year
  python main.py competitions --year 2020-2024  Stats for a year range
  python main.py competitions --event GP        Stats for GP only
  python main.py records                         Show career records sorted by #1 count
  python main.py records --sort wins             Sort by round wins
  python main.py records --sort adj-points       Sort by total adjusted points
  python main.py records --method no-prior       Disable prior regularization
  python main.py records --top 10                Limit to 10 rows
  python main.py export                          Export rating data files
  python main.py export --output-dir ./data/     Export to specific directory
  python main.py export --format parquet         Export only Parquet files
  python main.py cache                           Show cache info
  python main.py cache --purge                   Clear the cache
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Leaderboard command
    lb_parser = subparsers.add_parser('leaderboard', help='Generate leaderboard')
    lb_parser.add_argument('--all-time', action='store_true',
                          help='Show all-time leaderboard (no inactivity filter)')
    lb_parser.add_argument('--year', type=int,
                          help='Show leaderboard as of this year')
    lb_parser.add_argument('--top', type=int, default=30,
                          help='Number of entries to show (default: 30)')
    lb_parser.add_argument('--method', type=str, choices=['prior', 'no-prior'],
                          default='prior',
                          help='Rating method: prior (default, with regularization) or no-prior')

    # Solver command
    solver_parser = subparsers.add_parser('solver', help='Show solver rating progression')
    solver_parser.add_argument('name', help='Solver name (partial match supported)')
    solver_parser.add_argument('--top', type=int,
                              help='Limit to N most recent rounds')
    solver_parser.add_argument('--method', type=str, choices=['prior', 'no-prior'],
                              default='prior',
                              help='Rating method: prior (default, with regularization) or no-prior')

    # Progression command
    prog_parser = subparsers.add_parser('progression', help='Show leadership progression over time')
    prog_parser.add_argument('--top', type=int, default=3,
                            help='Number of leaders to show per round (default: 3)')
    prog_parser.add_argument('--from-year', type=int, dest='from_year',
                            help='Start from this year')
    prog_parser.add_argument('--to-year', type=int, dest='to_year',
                            help='End at this year')
    prog_parser.add_argument('--ratings', action='store_true',
                            help='Show ratings alongside names')
    prog_parser.add_argument('--method', type=str, choices=['prior', 'no-prior'],
                            default='prior',
                            help='Rating method: prior (default, with regularization) or no-prior')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare rating methods')
    compare_parser.add_argument('--leaderboard', action='store_true',
                               help='Compare leaderboards instead of accuracy')
    compare_parser.add_argument('--top', type=int, default=20,
                               help='Number of entries for leaderboard comparison (default: 20)')
    compare_parser.add_argument('--year', type=int,
                               help='Compare as of this year')
    compare_parser.add_argument('--burn-in', type=int, default=0, dest='burn_in',
                               help='Skip first N rounds before measuring accuracy (default: 0)')
    compare_parser.add_argument('--horizon', type=int, default=3,
                               help='Number of future rounds for horizon accuracy (default: 3)')

    # Competitions command
    comp_parser = subparsers.add_parser('competitions', help='Show competition statistics')
    comp_parser.add_argument('--year', type=str,
                            help='Filter by year (e.g., 2024) or range (e.g., 2020-2024)')
    comp_parser.add_argument('--event', type=str, choices=['GP', 'WSC', 'gp', 'wsc'],
                            help='Filter by event type (GP or WSC)')

    # Records command
    records_parser = subparsers.add_parser('records', help='Show career records')
    records_parser.add_argument('--sort', type=str,
                               choices=['ones', 'streak', 'wins', 'adj-points', 'points', 'rounds'],
                               default='ones',
                               help='Sort by column (default: ones)')
    records_parser.add_argument('--method', type=str, choices=['prior', 'no-prior'],
                               default='prior',
                               help='Rating method: prior (default, with regularization) or no-prior')
    records_parser.add_argument('--top', type=int, default=20,
                               help='Number of entries to show (default: 20)')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export rating data for sudokudos website')
    export_parser.add_argument('--output-dir', type=str, default='./export/',
                              help='Output directory (default: ./export/)')
    export_parser.add_argument('--format', type=str, choices=['parquet', 'csv', 'both'],
                              default='both',
                              help='Output format (default: both)')
    export_parser.add_argument('--method', type=str, choices=['prior', 'no-prior'],
                              default='prior',
                              help='Rating method: prior (default, with regularization) or no-prior')

    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Show or manage data cache')
    cache_parser.add_argument('--purge', action='store_true',
                             help='Delete all cached data')

    args = parser.parse_args()

    if args.command == 'leaderboard':
        cmd_leaderboard(args)
    elif args.command == 'solver':
        cmd_solver(args)
    elif args.command == 'progression':
        cmd_progression(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'competitions':
        cmd_competitions(args)
    elif args.command == 'records':
        cmd_records(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'cache':
        cmd_cache(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
