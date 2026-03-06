"""Leaderboard generation and display utilities.

Provides functions for generating leaderboards from both
points-based and percentile-based rating systems.
"""

from dataclasses import dataclass
from typing import Optional, Union

import polars as pl

from ratings.competitions import CompetitionIdentifier, get_all_competitions, get_competition_index


# Solvers are dropped from the leaderboard after this many full calendar years of inactivity.
# A solver active in 2022 remains on the 2023 leaderboard but drops off in 2024.
INACTIVITY_THRESHOLD_YEARS = 1


@dataclass
class LeaderboardEntry:
    """A single entry in the points-based leaderboard."""
    rank: int
    solver_id: str
    exp_weighted_mean: float
    mean_adj_points: float
    n_rounds: int
    last_active_year: int
    # Place in last round (optional, populated when normalized_df provided)
    last_place: Optional[int] = None
    last_round_size: Optional[int] = None
    last_round_label: Optional[str] = None


@dataclass
class PercentileLeaderboardEntry:
    """A single entry in the percentile-based leaderboard."""
    rank: int
    solver_id: str
    exp_weighted_pct: float
    mean_adj_percentile: float
    n_rounds: int
    last_active_year: int


def generate_leaderboard(
    features_df: pl.DataFrame,
    top_n: int = 30,
    as_of_year: Optional[int] = None,
    normalized_df: Optional[pl.DataFrame] = None
) -> list[LeaderboardEntry]:
    """Generate current leaderboard based on exponentially weighted mean.

    Uses each solver's most recent feature snapshot as of the specified year.
    Excludes solvers inactive for more than INACTIVITY_THRESHOLD_YEARS.

    Args:
        features_df: DataFrame from build_features_and_labels()
        top_n: Number of top solvers to return
        as_of_year: Year to calculate leaderboard for (default: max year in data).
            Only data up to this year is considered (no future data leakage).
            Solvers must have competed in as_of_year or as_of_year-1 to be included.
        normalized_df: Optional normalized data for computing place in last round.
            If provided, entries will include last_place and last_round_size.

    Returns:
        List of LeaderboardEntry sorted by exp_weighted_mean descending
    """
    # Determine the reference year
    if as_of_year is None:
        as_of_year = features_df['year'].max()

    # Filter to only include data up to as_of_year (prevent future data leakage)
    historical_df = features_df.filter(pl.col('year') <= as_of_year)

    # Minimum year for activity (must have competed in this year or later)
    min_active_year = as_of_year - INACTIVITY_THRESHOLD_YEARS

    # Add chronological index for proper sorting across GP/WSC
    comp_to_idx = get_competition_index()
    historical_df = historical_df.with_columns(
        pl.struct(["year", "round", "competition"]).map_elements(
            lambda x: comp_to_idx.get((x["year"], x["round"], x["competition"]), -1),
            return_dtype=pl.Int64
        ).alias("comp_idx")
    )

    # Get latest record per solver from historical data only
    latest = (
        historical_df
        .sort(['user_pseudo_id', 'comp_idx'], descending=[False, True])
        .group_by('user_pseudo_id')
        .first()
    )

    # Filter to only active solvers
    latest = latest.filter(pl.col('year') >= min_active_year)

    # Sort and limit
    latest = (
        latest
        .sort('exp_weighted_mean', descending=True)
        .head(top_n)
    )

    # Pre-compute place info if normalized_df provided
    place_info = {}
    if normalized_df is not None:
        # Get unique rounds each solver's last round falls into
        for row in latest.iter_rows(named=True):
            solver_id = row['user_pseudo_id']
            year = row['year']
            rnd = row['round']
            comp = row['competition']

            # Get all participants in this round
            round_data = normalized_df.filter(
                (pl.col('year') == year) &
                (pl.col('round') == rnd) &
                (pl.col('competition') == comp)
            ).sort('points', descending=True)

            if len(round_data) > 0:
                # Find this solver's place
                participants = round_data['user_pseudo_id'].to_list()
                total = len(participants)
                if solver_id in participants:
                    place = participants.index(solver_id) + 1
                    label = f"{year} {comp} R{rnd}"
                    place_info[solver_id] = (place, total, label)

    entries = []
    for i, row in enumerate(latest.iter_rows(named=True), 1):
        solver_id = row['user_pseudo_id']
        place, total, label = place_info.get(solver_id, (None, None, None))
        entries.append(LeaderboardEntry(
            rank=i,
            solver_id=row['user_pseudo_id'],
            exp_weighted_mean=row['exp_weighted_mean'],
            mean_adj_points=row['mean_adj_points'],
            n_rounds=row['n_rounds'],
            last_active_year=row['year'],
            last_place=place,
            last_round_size=total,
            last_round_label=label
        ))

    return entries


def print_leaderboard(entries: list[LeaderboardEntry]) -> None:
    """Print leaderboard in formatted table."""
    print(f"{'Rank':<5} {'Solver':<40} {'Exp Wtd':>10} {'Mean':>10} {'Rounds':>8} {'Last':>6}")
    print("-" * 85)
    for entry in entries:
        print(f"{entry.rank:<5} {entry.solver_id:<40} "
              f"{entry.exp_weighted_mean:>10.1f} {entry.mean_adj_points:>10.1f} "
              f"{entry.n_rounds:>8} {entry.last_active_year:>6}")


def generate_leaderboard_after_round(
    features_df: pl.DataFrame,
    competition: CompetitionIdentifier,
    top_n: int = 3
) -> list[LeaderboardEntry]:
    """Generate leaderboard state immediately after a specific round.

    Note: This uses pre-round features from features_df, so the rating shown
    for each solver is based on their history before the round they last
    participated in. For post-round ratings, use RatingTracker instead.

    Args:
        features_df: DataFrame from build_features_and_labels()
        competition: The round after which to generate the leaderboard
        top_n: Number of top solvers to return

    Returns:
        List of LeaderboardEntry sorted by exp_weighted_mean descending
    """
    comp_to_idx = get_competition_index()
    target_idx = comp_to_idx.get(
        (competition.year, competition.round, competition.event_type)
    )
    if target_idx is None:
        return []

    # Add chronological index and filter to rounds up to this competition
    historical_df = features_df.with_columns(
        pl.struct(["year", "round", "competition"]).map_elements(
            lambda x: comp_to_idx.get((x["year"], x["round"], x["competition"]), -1),
            return_dtype=pl.Int64
        ).alias("comp_idx")
    ).filter(pl.col("comp_idx") <= target_idx)

    if len(historical_df) == 0:
        return []

    # Apply inactivity threshold based on the competition year
    min_active_year = competition.year - INACTIVITY_THRESHOLD_YEARS

    # Get latest record per solver by chronological order
    latest = (
        historical_df
        .sort(['user_pseudo_id', 'comp_idx'], descending=[False, True])
        .group_by('user_pseudo_id')
        .first()
    )

    # Filter to active solvers
    latest = latest.filter(pl.col('year') >= min_active_year)

    # Sort and limit
    latest = (
        latest
        .sort('exp_weighted_mean', descending=True)
        .head(top_n)
    )

    entries = []
    for i, row in enumerate(latest.iter_rows(named=True), 1):
        entries.append(LeaderboardEntry(
            rank=i,
            solver_id=row['user_pseudo_id'],
            exp_weighted_mean=row['exp_weighted_mean'],
            mean_adj_points=row['mean_adj_points'],
            n_rounds=row['n_rounds'],
            last_active_year=row['year']
        ))

    return entries


# =============================================================================
# Percentile-based leaderboard functions
# =============================================================================


def generate_percentile_leaderboard(
    features_df: pl.DataFrame,
    top_n: int = 30,
    as_of_year: Optional[int] = None
) -> list[PercentileLeaderboardEntry]:
    """Generate current leaderboard based on exponentially weighted percentile.

    Uses each solver's most recent feature snapshot as of the specified year.
    Excludes solvers inactive for more than INACTIVITY_THRESHOLD_YEARS.

    Args:
        features_df: DataFrame from build_percentile_features()
        top_n: Number of top solvers to return
        as_of_year: Year to calculate leaderboard for (default: max year in data).
            Only data up to this year is considered (no future data leakage).

    Returns:
        List of PercentileLeaderboardEntry sorted by exp_weighted_pct descending
    """
    if as_of_year is None:
        as_of_year = features_df['year'].max()

    # Filter to only include data up to as_of_year (prevent future data leakage)
    historical_df = features_df.filter(pl.col('year') <= as_of_year)

    min_active_year = as_of_year - INACTIVITY_THRESHOLD_YEARS

    # Add chronological index for proper sorting across GP/WSC
    comp_to_idx = get_competition_index()
    historical_df = historical_df.with_columns(
        pl.struct(["year", "round", "competition"]).map_elements(
            lambda x: comp_to_idx.get((x["year"], x["round"], x["competition"]), -1),
            return_dtype=pl.Int64
        ).alias("comp_idx")
    )

    # Get latest record per solver from historical data only
    latest = (
        historical_df
        .sort(['user_pseudo_id', 'comp_idx'], descending=[False, True])
        .group_by('user_pseudo_id')
        .first()
    )

    # Filter to only active solvers
    latest = latest.filter(pl.col('year') >= min_active_year)

    # Sort and limit
    latest = (
        latest
        .sort('exp_weighted_pct', descending=True)
        .head(top_n)
    )

    entries = []
    for i, row in enumerate(latest.iter_rows(named=True), 1):
        entries.append(PercentileLeaderboardEntry(
            rank=i,
            solver_id=row['user_pseudo_id'],
            exp_weighted_pct=row['exp_weighted_pct'],
            mean_adj_percentile=row['mean_adj_percentile'],
            n_rounds=row['n_rounds'],
            last_active_year=row['year']
        ))

    return entries


def print_percentile_leaderboard(entries: list[PercentileLeaderboardEntry]) -> None:
    """Print percentile leaderboard in formatted table."""
    print(f"{'Rank':<5} {'Solver':<40} {'Exp Wtd':>10} {'Mean':>10} {'Rounds':>8} {'Last':>6}")
    print("-" * 85)
    for entry in entries:
        print(f"{entry.rank:<5} {entry.solver_id:<40} "
              f"{entry.exp_weighted_pct:>10.3f} {entry.mean_adj_percentile:>10.3f} "
              f"{entry.n_rounds:>8} {entry.last_active_year:>6}")
