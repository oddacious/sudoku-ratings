"""CLI command implementations for sudoku ratings.

This package contains the business logic for CLI commands,
separated from the argument parsing in main.py.
"""

from ratings.cli.commands import (
    load_features,
    cmd_leaderboard,
    cmd_progression,
    cmd_solver,
    cmd_compare,
    cmd_cache,
    cmd_competitions,
    cmd_records,
    cmd_export,
    compute_horizon_accuracy,
)

__all__ = [
    'load_features',
    'cmd_leaderboard',
    'cmd_progression',
    'cmd_solver',
    'cmd_compare',
    'cmd_cache',
    'cmd_competitions',
    'cmd_records',
    'cmd_export',
    'compute_horizon_accuracy',
]
