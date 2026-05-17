"""Diff command: compare two ratings export directories."""

import json
from pathlib import Path
from typing import Optional

import polars as pl

_SNAPSHOTS_DIR = Path('./snapshots')
_EXPORT_DIR = Path('./export')

# Thresholds for flagging suspicious changes
_WARN_DELTA_RATING = 50.0
_WARN_DELTA_RANK = 30


def _load_metadata(directory: Path) -> dict:
    meta_path = directory / 'metadata.json'
    if meta_path.exists():
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def _latest_snapshot() -> Optional[Path]:
    if not _SNAPSHOTS_DIR.exists():
        return None
    snapshots = sorted(_SNAPSHOTS_DIR.iterdir())
    return snapshots[-1] if snapshots else None


def _load_parquet(directory: Path, name: str) -> Optional[pl.DataFrame]:
    path = directory / f'{name}.parquet'
    if path.exists():
        return pl.read_parquet(path)
    return None


def cmd_diff(args) -> None:
    """Compare two export directories and print what changed."""
    # Resolve directories
    to_dir = Path(getattr(args, 'to', None) or _EXPORT_DIR)

    from_arg = getattr(args, 'from', None)
    if from_arg:
        from_dir = Path(from_arg)
    else:
        from_dir = _latest_snapshot()
        if from_dir is None:
            print("No snapshots found. Run 'python main.py export' to create one.")
            return

    if not from_dir.exists():
        print(f"Source directory not found: {from_dir}")
        return
    if not to_dir.exists():
        print(f"Target directory not found: {to_dir}")
        return

    from_meta = _load_metadata(from_dir)
    to_meta = _load_metadata(to_dir)

    # --- Summary ---
    print(f"\nComparing: {from_dir}  →  {to_dir}")
    from_through = from_meta.get('data_through', '?')
    to_through = to_meta.get('data_through', '?')
    print(f"Data through:   {from_through}  →  {to_through}")
    from_solvers = from_meta.get('total_solvers', '?')
    to_solvers = to_meta.get('total_solvers', '?')
    if isinstance(from_solvers, int) and isinstance(to_solvers, int):
        delta_solvers = to_solvers - from_solvers
        sign = '+' if delta_solvers >= 0 else ''
        print(f"Rated solvers:  {from_solvers:,}  →  {to_solvers:,}  ({sign}{delta_solvers:,})")
    else:
        print(f"Rated solvers:  {from_solvers}  →  {to_solvers}")

    # --- Leaderboard diff ---
    old_lb = _load_parquet(from_dir, 'leaderboard_current')
    new_lb = _load_parquet(to_dir, 'leaderboard_current')

    if old_lb is None or new_lb is None:
        print("\n(leaderboard_current.parquet missing from one side — skipping leaderboard diff)")
    else:
        _print_leaderboard_diff(old_lb, new_lb)

    # --- Records diff ---
    old_rec = _load_parquet(from_dir, 'records')
    new_rec = _load_parquet(to_dir, 'records')

    if old_rec is not None and new_rec is not None:
        _print_records_diff(old_rec, new_rec)


def _print_leaderboard_diff(
    old_lb: pl.DataFrame,
    new_lb: pl.DataFrame,
) -> None:
    old_sel = old_lb.select(['user_pseudo_id', 'rank', 'rating'])
    new_sel = new_lb.select(['user_pseudo_id', 'rank', 'rating'])

    joined = old_sel.join(new_sel, on='user_pseudo_id', how='full', suffix='_new', coalesce=True)

    # Entrants and exits
    exits = joined.filter(pl.col('rank_new').is_null())['user_pseudo_id'].to_list()
    entrants = joined.filter(pl.col('rank').is_null())['user_pseudo_id'].to_list()

    print(f"\nNew to leaderboard ({len(entrants)}): ", end='')
    print(', '.join(entrants[:10]) + ('...' if len(entrants) > 10 else '') if entrants else 'none')
    print(f"Left leaderboard  ({len(exits)}): ", end='')
    print(', '.join(exits[:10]) + ('...' if len(exits) > 10 else '') if exits else 'none')

    # Movers: only solvers present in both
    both = joined.filter(pl.col('rank').is_not_null() & pl.col('rank_new').is_not_null())
    both = both.with_columns([
        (pl.col('rank_new') - pl.col('rank')).alias('delta_rank'),
        (pl.col('rating_new') - pl.col('rating')).alias('delta_rating'),
    ])
    both = both.with_columns(pl.col('delta_rating').abs().alias('abs_delta_rating'))
    movers = both.sort('abs_delta_rating', descending=True).head(25)

    print("\nTop movers (by |Δrating|):")
    header = f"  {'':1} {'Solver':<42} {'OldRk':>6} {'NewRk':>6} {'ΔRk':>5}  {'OldRating':>9} {'NewRating':>9} {'ΔRating':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in movers.iter_rows(named=True):
        d_rank = row['delta_rank']
        d_rating = row['delta_rating']
        warn = '!' if abs(d_rating) >= _WARN_DELTA_RATING or abs(d_rank) >= _WARN_DELTA_RANK else ' '
        rank_arrow = f"{'-' if d_rank > 0 else '+'}{abs(d_rank)}" if d_rank != 0 else '='
        rating_sign = '+' if d_rating >= 0 else ''
        name = row['user_pseudo_id']
        if len(name) > 42:
            name = name[:39] + '...'
        print(
            f"  {warn} {name:<42} {row['rank']:>6} {row['rank_new']:>6} {rank_arrow:>5}"
            f"  {row['rating']:>9.1f} {row['rating_new']:>9.1f} {rating_sign}{d_rating:>7.1f}"
        )

    unchanged = (both['abs_delta_rating'] == 0).sum()
    if unchanged > 0:
        print(f"  ({unchanged} solvers unchanged)")


def _print_records_diff(old_rec: pl.DataFrame, new_rec: pl.DataFrame) -> None:
    old_sel = old_rec.select(['user_pseudo_id', 'ones_count', 'best_streak', 'wins_count'])
    new_sel = new_rec.select(['user_pseudo_id', 'ones_count', 'best_streak', 'wins_count'])

    joined = old_sel.join(new_sel, on='user_pseudo_id', how='inner', suffix='_new')

    changed = joined.filter(
        (pl.col('ones_count') != pl.col('ones_count_new')) |
        (pl.col('best_streak') != pl.col('best_streak_new')) |
        (pl.col('wins_count') != pl.col('wins_count_new'))
    )

    if len(changed) == 0:
        print("\nRecords: no changes")
        return

    print(f"\nRecords changes ({len(changed)} solvers):")
    header = f"  {'Solver':<42} {'#1s':>5}→{'':>5} {'Streak':>7}→{'':>7} {'Wins':>5}→{'':>5}"
    print(header)
    print("  " + "-" * 70)
    for row in changed.sort('ones_count_new', descending=True).iter_rows(named=True):
        name = row['user_pseudo_id']
        if len(name) > 42:
            name = name[:39] + '...'
        print(
            f"  {name:<42}"
            f" {row['ones_count']:>5}→{row['ones_count_new']:<5}"
            f" {row['best_streak']:>7}→{row['best_streak_new']:<7}"
            f" {row['wins_count']:>5}→{row['wins_count_new']:<5}"
        )
