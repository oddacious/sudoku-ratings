"""Export ratings data for the sudokudos website.

This module provides functions to export rating data in formats consumable
by the sudokudos-github Streamlit app.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from ratings.competitions import get_all_competitions, get_competition_index
from ratings.data_loader import load_normalized_data
from ratings.leaderboard import INACTIVITY_THRESHOLD_YEARS
from ratings.methods.points_based import compute_adjusted_points
from ratings.methods.rating_tracker import RatingTracker


def export_timeseries(
    tracker: RatingTracker,
    normalized: pl.DataFrame,
    adjusted: pl.DataFrame,
) -> pl.DataFrame:
    """Build full rating history for all solvers.

    For each solver-round combination, records the rating after that round,
    plus contextual information like rank and points.

    Args:
        tracker: RatingTracker that has been advanced to the final competition
        normalized: Normalized data (for raw points)
        adjusted: Adjusted data (for adjusted points)

    Returns:
        DataFrame with columns:
        [user_pseudo_id, year, round, competition, comp_idx, rating, n_rounds,
         rank, rank_total, raw_points, adjusted_points]
    """
    all_competitions = get_all_competitions()
    comp_to_idx = get_competition_index()

    # Reset tracker to process from the beginning
    tracker.reset()

    rows = []

    for comp in all_competitions:
        # Advance tracker to include this competition
        tracker.advance_to(comp)

        comp_idx = comp_to_idx[(comp.year, comp.round, comp.event_type)]

        # Get all ratings at this point (including inactive for historical record)
        all_rated = []
        for solver_id, history in tracker.histories.items():
            if len(history) >= tracker.min_history:
                rating = tracker.compute_rating(history)
                all_rated.append((solver_id, rating, len(history)))

        # Sort by rating to get ranks
        all_rated.sort(key=lambda x: x[1], reverse=True)
        solver_to_rank = {s: (i + 1, len(all_rated)) for i, (s, _, _) in enumerate(all_rated)}

        # Get round data for raw and adjusted points
        round_norm = normalized.filter(
            (pl.col('year') == comp.year) &
            (pl.col('round') == comp.round) &
            (pl.col('competition') == comp.event_type)
        )
        round_adj = adjusted.filter(
            (pl.col('year') == comp.year) &
            (pl.col('round') == comp.round) &
            (pl.col('competition') == comp.event_type)
        )

        # Build lookup for points
        raw_points_lookup = {
            row['user_pseudo_id']: row['points']
            for row in round_norm.iter_rows(named=True)
        }
        adj_points_lookup = {
            row['user_pseudo_id']: row['adjusted_points']
            for row in round_adj.iter_rows(named=True)
        }

        # Create row for each participant in this round who has a rating now
        for solver_id in raw_points_lookup:
            if solver_id not in tracker.histories:
                continue
            history = tracker.histories[solver_id]
            if len(history) < tracker.min_history:
                continue

            rating = tracker.compute_rating(history)
            rank, rank_total = solver_to_rank.get(solver_id, (None, None))

            rows.append({
                'user_pseudo_id': solver_id,
                'year': comp.year,
                'round': comp.round,
                'competition': comp.event_type,
                'comp_idx': comp_idx,
                'rating': rating,
                'n_rounds': len(history),
                'rank': rank,
                'rank_total': rank_total,
                'raw_points': raw_points_lookup.get(solver_id),
                'adjusted_points': adj_points_lookup.get(solver_id),
            })

    return pl.DataFrame(rows)


def export_current_leaderboard(
    tracker: RatingTracker,
    normalized: pl.DataFrame,
    as_of_year: int,
) -> pl.DataFrame:
    """Export current active leaderboard.

    Args:
        tracker: RatingTracker advanced to final competition
        normalized: Normalized data for last-round info
        as_of_year: Year to use for activity filtering

    Returns:
        DataFrame with columns:
        [rank, user_pseudo_id, rating, mean_adj_points, n_rounds,
         last_year, last_place, last_round_size]
    """
    min_active_year = as_of_year - INACTIVITY_THRESHOLD_YEARS

    # Get all active solvers with ratings
    active_ratings = []
    for solver_id, history in tracker.histories.items():
        last_year = tracker.last_years.get(solver_id, 0)
        if len(history) >= tracker.min_history and last_year >= min_active_year:
            rating = tracker.compute_rating(history)
            mean_adj = sum(history) / len(history)
            active_ratings.append({
                'user_pseudo_id': solver_id,
                'rating': rating,
                'mean_adj_points': mean_adj,
                'n_rounds': len(history),
                'last_year': last_year,
            })

    # Sort by rating
    active_ratings.sort(key=lambda x: x['rating'], reverse=True)

    # Add ranks and last-round info
    all_competitions = get_all_competitions()

    rows = []
    for rank, entry in enumerate(active_ratings, 1):
        solver_id = entry['user_pseudo_id']

        # Find solver's last round
        last_place = None
        last_round_size = None
        for comp in reversed(all_competitions):
            round_data = normalized.filter(
                (pl.col('year') == comp.year) &
                (pl.col('round') == comp.round) &
                (pl.col('competition') == comp.event_type)
            ).sort('points', descending=True)

            participants = round_data['user_pseudo_id'].to_list()
            if solver_id in participants:
                last_place = participants.index(solver_id) + 1
                last_round_size = len(participants)
                break

        rows.append({
            'rank': rank,
            'user_pseudo_id': solver_id,
            'rating': entry['rating'],
            'mean_adj_points': entry['mean_adj_points'],
            'n_rounds': entry['n_rounds'],
            'last_year': entry['last_year'],
            'last_place': last_place,
            'last_round_size': last_round_size,
        })

    return pl.DataFrame(rows)


def export_alltime_leaderboard(tracker: RatingTracker) -> pl.DataFrame:
    """Export all-time peak ratings for each solver.

    Args:
        tracker: RatingTracker (will be reset and re-run to track peaks)

    Returns:
        DataFrame with columns:
        [rank, user_pseudo_id, peak_rating, peak_year, peak_round,
         peak_competition, n_rounds]
    """
    all_competitions = get_all_competitions()

    # Reset and track peak ratings for each solver
    tracker.reset()

    # Track peak rating for each solver
    peak_ratings: dict[str, dict] = {}

    for comp in all_competitions:
        tracker.advance_to(comp)

        # Check each solver's current rating
        for solver_id, history in tracker.histories.items():
            if len(history) < tracker.min_history:
                continue

            rating = tracker.compute_rating(history)

            if solver_id not in peak_ratings or rating > peak_ratings[solver_id]['peak_rating']:
                peak_ratings[solver_id] = {
                    'user_pseudo_id': solver_id,
                    'peak_rating': rating,
                    'peak_year': comp.year,
                    'peak_round': comp.round,
                    'peak_competition': comp.event_type,
                    'n_rounds': len(history),
                }

    # Sort by peak rating
    sorted_peaks = sorted(peak_ratings.values(), key=lambda x: x['peak_rating'], reverse=True)

    # Add ranks
    rows = []
    for rank, entry in enumerate(sorted_peaks, 1):
        rows.append({
            'rank': rank,
            **entry,
        })

    return pl.DataFrame(rows)


def export_records(
    tracker: RatingTracker,
    normalized: pl.DataFrame,
    adjusted: pl.DataFrame,
) -> pl.DataFrame:
    """Export career records (#1 counts, streaks, wins, totals).

    Args:
        tracker: RatingTracker (will be reset and re-run)
        normalized: Normalized data for round wins
        adjusted: Adjusted data for total points

    Returns:
        DataFrame with columns:
        [user_pseudo_id, ones_count, best_streak, wins_count,
         total_adj_points, total_raw_points, total_rounds]
    """
    all_competitions = get_all_competitions()

    # Reset tracker
    tracker.reset()

    # Track #1 counts and streaks
    ones_count: dict[str, int] = {}
    best_streak: dict[str, int] = {}
    current_streak_solver: Optional[str] = None
    current_streak_len = 0

    for comp in all_competitions:
        tracker.advance_to(comp)
        leaderboard = tracker.get_leaderboard(top_n=1, as_of_year=comp.year)

        if not leaderboard:
            continue

        leader = leaderboard[0].solver_id
        ones_count[leader] = ones_count.get(leader, 0) + 1

        if leader == current_streak_solver:
            current_streak_len += 1
        else:
            # Save previous solver's streak
            if current_streak_solver is not None:
                prev_best = best_streak.get(current_streak_solver, 0)
                best_streak[current_streak_solver] = max(prev_best, current_streak_len)
            current_streak_solver = leader
            current_streak_len = 1

    # Save final streak
    if current_streak_solver is not None:
        prev_best = best_streak.get(current_streak_solver, 0)
        best_streak[current_streak_solver] = max(prev_best, current_streak_len)

    # Compute round wins (1st by raw points)
    wins_count: dict[str, int] = {}
    for comp in all_competitions:
        round_data = normalized.filter(
            (pl.col('year') == comp.year) &
            (pl.col('round') == comp.round) &
            (pl.col('competition') == comp.event_type)
        )
        if len(round_data) == 0:
            continue

        max_points = round_data['points'].max()
        winners = round_data.filter(pl.col('points') == max_points)['user_pseudo_id'].to_list()
        for winner in winners:
            wins_count[winner] = wins_count.get(winner, 0) + 1

    # Compute career totals
    career_totals = (
        adjusted
        .group_by('user_pseudo_id')
        .agg([
            pl.col('adjusted_points').sum().alias('total_adj_points'),
            pl.col('points').sum().alias('total_raw_points'),
            pl.col('points').count().alias('total_rounds'),
        ])
    )

    totals_lookup = {}
    for row in career_totals.iter_rows(named=True):
        totals_lookup[row['user_pseudo_id']] = {
            'total_adj_points': row['total_adj_points'],
            'total_raw_points': row['total_raw_points'],
            'total_rounds': row['total_rounds'],
        }

    # Merge all data
    all_solvers = set(ones_count.keys()) | set(wins_count.keys()) | set(totals_lookup.keys())

    rows = []
    for solver in all_solvers:
        totals = totals_lookup.get(solver, {
            'total_adj_points': 0,
            'total_raw_points': 0,
            'total_rounds': 0,
        })
        # Skip solvers with fewer than 3 rounds
        if totals['total_rounds'] < 3:
            continue

        rows.append({
            'user_pseudo_id': solver,
            'ones_count': ones_count.get(solver, 0),
            'best_streak': best_streak.get(solver, 0),
            'wins_count': wins_count.get(solver, 0),
            'total_adj_points': totals['total_adj_points'],
            'total_raw_points': totals['total_raw_points'],
            'total_rounds': totals['total_rounds'],
        })

    return pl.DataFrame(rows)


def write_metadata(
    output_dir: Path,
    method: str,
    total_solvers: int,
    latest_comp: str,
) -> None:
    """Write export metadata JSON.

    Args:
        output_dir: Directory to write to
        method: Rating method used
        total_solvers: Total number of solvers in export
        latest_comp: String describing latest competition (e.g., "2026 GP R2")
    """
    metadata = {
        'version': '1.0',
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'method': method,
        'data_through': latest_comp,
        'total_solvers': total_solvers,
    }

    output_path = output_dir / 'metadata.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def write_dataframe(df: pl.DataFrame, output_dir: Path, name: str, fmt: str) -> None:
    """Write a DataFrame in the specified format(s).

    Args:
        df: DataFrame to write
        output_dir: Directory to write to
        name: Base filename (without extension)
        fmt: Format - 'parquet', 'csv', or 'both'
    """
    if fmt in ('parquet', 'both'):
        df.write_parquet(output_dir / f'{name}.parquet')
    if fmt in ('csv', 'both'):
        df.write_csv(output_dir / f'{name}.csv')


def run_export(
    output_dir: str = './export/',
    fmt: str = 'both',
    method: str = 'prior',
) -> dict:
    """Run full export pipeline.

    Args:
        output_dir: Output directory path
        fmt: Output format - 'parquet', 'csv', or 'both'
        method: Rating method - 'prior' or 'no-prior'

    Returns:
        Dict with export statistics
    """
    import sys

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prior_k = 0 if method == 'no-prior' else 3

    # Load and process data
    print("Loading data...", file=sys.stderr)
    normalized = load_normalized_data()

    print("Computing difficulty-adjusted points...", file=sys.stderr)
    adjusted = compute_adjusted_points(normalized, n_reference=16, use_gp_baseline=True)

    print("Setting up rating tracker...", file=sys.stderr)
    tracker = RatingTracker(adjusted, min_history=3, decay_rate=0.90, prior_k=prior_k)

    # Find the last competition that actually has data
    all_competitions = get_all_competitions()
    final_comp = None
    for comp in reversed(all_competitions):
        round_data = normalized.filter(
            (pl.col('year') == comp.year) &
            (pl.col('round') == comp.round) &
            (pl.col('competition') == comp.event_type)
        )
        if len(round_data) > 0:
            final_comp = comp
            break

    if final_comp is None:
        raise ValueError("No competition data found")

    # Advance tracker to final competition with actual data
    tracker.advance_to(final_comp)

    latest_comp_str = f"{final_comp.year} {final_comp.event_type} R{final_comp.round}"
    as_of_year = final_comp.year

    # Export timeseries
    print("Exporting timeseries...", file=sys.stderr)
    timeseries_df = export_timeseries(tracker, normalized, adjusted)
    write_dataframe(timeseries_df, output_path, 'ratings_timeseries', fmt)

    # Export current leaderboard
    print("Exporting current leaderboard...", file=sys.stderr)
    # Need to re-advance tracker since timeseries resets it
    tracker.reset()
    tracker.advance_to(final_comp)
    current_lb_df = export_current_leaderboard(tracker, normalized, as_of_year)
    write_dataframe(current_lb_df, output_path, 'leaderboard_current', fmt)

    # Export all-time leaderboard
    print("Exporting all-time leaderboard...", file=sys.stderr)
    alltime_lb_df = export_alltime_leaderboard(tracker)
    write_dataframe(alltime_lb_df, output_path, 'leaderboard_alltime', fmt)

    # Export records
    print("Exporting records...", file=sys.stderr)
    records_df = export_records(tracker, normalized, adjusted)
    write_dataframe(records_df, output_path, 'records', fmt)

    # Get unique solver count from timeseries
    total_solvers = timeseries_df['user_pseudo_id'].n_unique()

    # Write metadata
    print("Writing metadata...", file=sys.stderr)
    write_metadata(output_path, method, total_solvers, latest_comp_str)

    print(f"Export complete: {output_path}", file=sys.stderr)

    return {
        'output_dir': str(output_path),
        'method': method,
        'format': fmt,
        'total_solvers': total_solvers,
        'timeseries_rows': len(timeseries_df),
        'current_leaderboard_rows': len(current_lb_df),
        'alltime_leaderboard_rows': len(alltime_lb_df),
        'records_rows': len(records_df),
        'data_through': latest_comp_str,
    }
