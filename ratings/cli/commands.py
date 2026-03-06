"""Command implementations for the sudoku ratings CLI.

Each cmd_* function implements one CLI subcommand, receiving
the parsed argparse namespace and handling all business logic.
"""

import sys

import polars as pl

from ratings.data_loader import (
    load_normalized_data,
    load_all_data,
    get_cache_info,
    purge_cache,
)
from ratings.methods.points_based import (
    compute_adjusted_points,
    build_features_and_labels,
    build_features_with_prior,
)
from ratings.methods.percentile_based import compute_percentile_ratings, build_percentile_features
from ratings.methods.rating_tracker import RatingTracker
from ratings.leaderboard import (
    LeaderboardEntry,
    INACTIVITY_THRESHOLD_YEARS,
    generate_leaderboard,
    generate_percentile_leaderboard,
)
from ratings.evaluation import (
    backtest_predictor,
    backtest_glicko,
)
from ratings.backtest_bridge import get_rounds_only
from ratings.competitions import get_all_competitions, get_competition_index
from ratings.competition_difficulty import difficulty_of_all_rounds
from ratings.competition_results import fetch_participant_records
from ratings.export import run_export


def load_features(return_normalized: bool = False, method: str = "prior"):
    """Load data and compute features.

    Args:
        return_normalized: If True, also return the normalized DataFrame.
        method: Rating method to use. "prior" (default) for exp-weighted with
                k=3 pseudo-observations, "no-prior" for standard exp-weighted.

    Returns:
        features DataFrame, or (features, normalized) tuple if return_normalized=True
    """
    print("Loading data...", file=sys.stderr)
    normalized = load_normalized_data()

    print("Computing difficulty-adjusted points...", file=sys.stderr)
    adjusted = compute_adjusted_points(normalized, n_reference=16, use_gp_baseline=True)

    if method == "no-prior":
        print("Building features...", file=sys.stderr)
        features_df, _ = build_features_and_labels(adjusted, min_history=3, decay_rate=0.90)
    else:
        print("Building features with prior (k=3)...", file=sys.stderr)
        features_df, _ = build_features_with_prior(adjusted, min_history=3, decay_rate=0.90, prior_k=3)

    if return_normalized:
        return features_df, normalized
    return features_df


def cmd_leaderboard(args):
    """Generate and print leaderboard."""
    method = getattr(args, 'method', 'prior')
    features_df, normalized = load_features(return_normalized=True, method=method)

    # Determine the year
    if args.year:
        as_of_year = args.year
    else:
        as_of_year = features_df['year'].max()

    if args.all_time:
        # Show each solver at their peak rating
        historical_df = features_df.filter(pl.col('year') <= as_of_year)

        # Get the row with maximum rating for each solver
        peak = (
            historical_df
            .sort(['user_pseudo_id', 'exp_weighted_mean'], descending=[False, True])
            .group_by('user_pseudo_id')
            .first()
        )

        peak = (
            peak
            .sort('exp_weighted_mean', descending=True)
            .head(args.top)
        )

        # Compute place info for the peak round
        place_info = {}
        for row in peak.iter_rows(named=True):
            solver_id = row['user_pseudo_id']
            year = row['year']
            rnd = row['round']
            comp = row['competition']

            round_data = normalized.filter(
                (pl.col('year') == year) &
                (pl.col('round') == rnd) &
                (pl.col('competition') == comp)
            ).sort('points', descending=True)

            if len(round_data) > 0:
                participants = round_data['user_pseudo_id'].to_list()
                total = len(participants)
                if solver_id in participants:
                    place = participants.index(solver_id) + 1
                    label = f"{year} {comp} R{rnd}"
                    place_info[solver_id] = (place, total, label)

        entries = []
        for i, row in enumerate(peak.iter_rows(named=True), 1):
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
    else:
        entries = generate_leaderboard(
            features_df, top_n=args.top, as_of_year=as_of_year, normalized_df=normalized
        )

    # Print header
    label = "Peak Ratings" if args.all_time else f"Active (within {INACTIVITY_THRESHOLD_YEARS} year)"
    round_col = "Peak" if args.all_time else "Latest"
    method_label = " (no prior)" if method == "no-prior" else ""
    print(f"\n{'='*111}")
    print(f"Leaderboard as of {as_of_year} - {label}{method_label}")
    print(f"{'='*111}\n")

    print(f"{'Rank':<5} {'Solver':<40} {'Rating':>10} {'Mean':>10} {'Rounds':>8} {round_col:>12} {'Place':>14}")
    print("-" * 111)
    for entry in entries:
        if entry.last_place is not None and entry.last_round_size is not None:
            pct = 100 * (entry.last_round_size - entry.last_place) / entry.last_round_size
            place_str = f"{entry.last_place}/{entry.last_round_size} ({pct:.0f}%)"
        else:
            place_str = "N/A"
        if entry.last_round_label:
            round_str = entry.last_round_label
        else:
            round_str = str(entry.last_active_year)
        print(f"{entry.rank:<5} {entry.solver_id:<40} "
              f"{entry.exp_weighted_mean:>10.1f} {entry.mean_adj_points:>10.1f} "
              f"{entry.n_rounds:>8} {round_str:>12} {place_str:>14}")


def cmd_progression(args):
    """Show top N leaders after each round."""
    method = getattr(args, 'method', 'prior')
    prior_k = 0 if method == "no-prior" else 3

    print("Loading data...", file=sys.stderr)
    normalized = load_normalized_data()

    print("Computing difficulty-adjusted points...", file=sys.stderr)
    adjusted = compute_adjusted_points(normalized, n_reference=16, use_gp_baseline=True)

    print("Computing ratings...", file=sys.stderr)
    tracker = RatingTracker(adjusted, min_history=3, decay_rate=0.90, prior_k=prior_k)

    all_competitions = get_all_competitions()

    # Filter by year range if specified
    display_from = args.from_year
    display_to = args.to_year

    method_label = " (no prior)" if method == "no-prior" else ""
    print(f"\n{'='*100}")
    print(f"Leadership Progression (Top {args.top}){method_label}")
    if display_from or display_to:
        year_range = f"{display_from or 'start'} - {display_to or 'present'}"
        print(f"Years: {year_range}")
    print(f"{'='*100}\n")

    prev_top = []

    for comp in all_competitions:
        # Always advance tracker to maintain correct state
        tracker.advance_to(comp)

        # Skip display if outside the year filter
        if display_from and comp.year < display_from:
            continue
        if display_to and comp.year > display_to:
            continue

        leaderboard = tracker.get_leaderboard(top_n=args.top, as_of_year=comp.year)

        if not leaderboard:
            continue

        # Check if leadership changed
        current_top = [e.solver_id for e in leaderboard]
        changed = current_top != prev_top

        # Format the round identifier
        round_label = f"{comp.year} {comp.event_type} R{comp.round}"

        # Format top N names (shortened)
        def short_name(solver_id):
            # Extract just the name part before the dash or parenthesis
            name = solver_id.split(' - ')[0].split(' (')[0]
            if len(name) > 20:
                name = name[:17] + "..."
            return name

        # Show ratings too if requested
        if args.ratings:
            top_info = [f"{i+1}. {short_name(e.solver_id)} ({e.rating:.0f})"
                       for i, e in enumerate(leaderboard)]
        else:
            top_info = [f"{i+1}. {short_name(e.solver_id)}" for i, e in enumerate(leaderboard)]

        top_str = " | ".join(top_info)

        # Mark changes with asterisk
        marker = "*" if changed else " "

        print(f"{marker} {round_label:<15} {top_str}")

        prev_top = current_top


def cmd_solver(args):
    """Show rating progression for a specific solver."""
    method = getattr(args, 'method', 'prior')
    prior_k = 0 if method == "no-prior" else 3
    min_history = 3

    print("Loading data...", file=sys.stderr)
    normalized = load_normalized_data()

    print("Computing difficulty-adjusted points...", file=sys.stderr)
    adjusted = compute_adjusted_points(normalized, n_reference=16, use_gp_baseline=True)

    # Find solver by partial match (case-insensitive)
    search_term = args.name.lower()

    # Get unique solver IDs
    all_solvers = normalized['user_pseudo_id'].unique().to_list()

    # Find matches
    matches = [s for s in all_solvers if search_term in s.lower()]

    if not matches:
        print(f"No solver found matching '{args.name}'", file=sys.stderr)
        print("\nDid you mean one of these?", file=sys.stderr)
        suggestions = [s for s in all_solvers if any(word in s.lower() for word in search_term.split())][:10]
        for s in suggestions:
            print(f"  - {s}", file=sys.stderr)
        sys.exit(1)

    if len(matches) > 1:
        print(f"Multiple solvers match '{args.name}':", file=sys.stderr)
        for s in matches[:20]:
            print(f"  - {s}", file=sys.stderr)
        if len(matches) > 20:
            print(f"  ... and {len(matches) - 20} more", file=sys.stderr)
        print("\nPlease be more specific.", file=sys.stderr)
        sys.exit(1)

    solver_id = matches[0]

    print("Computing ratings...", file=sys.stderr)
    tracker = RatingTracker(adjusted, min_history=min_history, decay_rate=0.90, prior_k=prior_k)

    all_competitions = get_all_competitions()
    comp_to_idx = tracker.comp_to_idx

    # Get rounds this solver participated in
    solver_adjusted = adjusted.filter(pl.col('user_pseudo_id') == solver_id)

    if len(solver_adjusted) == 0:
        print(f"No rounds found for '{solver_id}'", file=sys.stderr)
        sys.exit(1)

    # Build set of competition indices for this solver
    solver_comp_indices = set()
    for row in solver_adjusted.iter_rows(named=True):
        idx = comp_to_idx.get((row['year'], row['round'], row['competition']), -1)
        if idx >= 0:
            solver_comp_indices.add(idx)

    # Iterate through all competitions, taking snapshots at solver's rounds
    all_rounds = []

    for comp in all_competitions:
        comp_idx = comp_to_idx[(comp.year, comp.round, comp.event_type)]

        # Advance tracker to include this competition
        tracker.advance_to(comp)

        # Skip if solver didn't participate in this round
        if comp_idx not in solver_comp_indices:
            continue

        # Get solver's rating and rank at this point
        solver_rating = tracker.get_solver_rating(solver_id)
        rank_info = tracker.get_solver_rank(solver_id, as_of_year=comp.year)

        # Get place info from normalized data
        round_data = normalized.filter(
            (pl.col('year') == comp.year) &
            (pl.col('round') == comp.round) &
            (pl.col('competition') == comp.event_type)
        ).sort('points', descending=True)

        place, round_size = None, None
        if len(round_data) > 0:
            participants = round_data['user_pseudo_id'].to_list()
            round_size = len(participants)
            if solver_id in participants:
                place = participants.index(solver_id) + 1

        # Build row data
        rating = solver_rating.rating if solver_rating else None
        n_rounds = solver_rating.n_rounds if solver_rating else len(tracker.histories.get(solver_id, []))

        rank, rank_total, rank_pct = None, None, None
        if rank_info:
            rank, rank_total = rank_info
            rank_pct = 100 * (rank_total - rank) / rank_total if rank_total > 0 else 0

        all_rounds.append({
            'year': comp.year,
            'round': comp.round,
            'competition': comp.event_type,
            'comp_idx': comp_idx,
            'place': place,
            'round_size': round_size,
            'rating': rating,
            'n_rounds': n_rounds,
            'rank': rank,
            'rank_total': rank_total,
            'rank_pct': rank_pct,
        })

    # Print header
    total_rounds = len(all_rounds)
    method_label = " (no prior)" if method == "no-prior" else ""
    print(f"\n{'='*93}")
    print(f"Rating Progression: {solver_id}{method_label}")
    print(f"{'='*93}\n")

    rows = all_rounds

    # Limit if requested
    if args.top and args.top < len(rows):
        rows = rows[-args.top:]
        print(f"(Showing last {args.top} of {total_rounds} rounds)\n")

    def print_header():
        print("-" * 93)
        print(f"{'Competition':<16} {'Place':>14} {'Rounds':>8} {'Rating':>12} {'Rank':>18}")
        print("-" * 93)

    prev_rating = None
    prev_year = None
    for row in rows:
        # Print header when year changes
        if row['year'] != prev_year:
            print_header()
            prev_year = row['year']

        comp_label = f"{row['year']} {row['competition']} R{row['round']}"

        # Place string
        if row['place'] is not None and row['round_size'] is not None:
            place_pct = 100 * (row['round_size'] - row['place']) / row['round_size']
            place_str = f"{row['place']}/{row['round_size']} ({place_pct:.0f}%)"
        else:
            place_str = "N/A"

        # Rounds string
        rounds_str = str(row['n_rounds']) if row['n_rounds'] is not None else "-"

        # Rating string with change indicator
        if row['rating'] is not None:
            change = ""
            if prev_rating is not None:
                if row['rating'] > prev_rating + 5:
                    change = "^"
                elif row['rating'] < prev_rating - 5:
                    change = "v"
            rating_str = f"{row['rating']:>.1f}{change}"
            prev_rating = row['rating']
        else:
            rating_str = "N/A"

        # Rank string
        if row['rank'] is not None:
            rank_str = f"{row['rank']}/{row['rank_total']} ({row['rank_pct']:.0f}%)"
        else:
            rank_str = "N/A"

        print(f"{comp_label:<16} {place_str:>14} {rounds_str:>8} {rating_str:>12} {rank_str:>18}")

    # Print summary
    final = rows[-1]
    print("-" * 93)
    if final['rating'] is not None:
        print(f"\nCurrent Rating: {final['rating']:.1f}")
    if final['rank']:
        print(f"Current Rank: {final['rank']} of {final['rank_total']} ({final['rank_pct']:.0f}%)")
    print(f"Total Rounds: {total_rounds}")
    print(f"Last Active: {final['year']} {final['competition']} R{final['round']}", end="")
    if final['place'] and final['round_size']:
        final_place_pct = 100 * (final['round_size'] - final['place']) / final['round_size']
        print(f" - Place: {final['place']}/{final['round_size']} ({final_place_pct:.0f}%)")
    else:
        print()


def cmd_compare(args):
    """Compare different rating methods."""
    print("Loading data...", file=sys.stderr)
    normalized = load_normalized_data()

    if args.leaderboard:
        # Compare leaderboards across methods
        _compare_leaderboards(normalized, args.top, args.year)
    else:
        # Compare accuracy metrics
        _compare_accuracy(normalized, args.burn_in, args.horizon)


def compute_horizon_accuracy(features_df: pl.DataFrame, predictor: str, horizon: int) -> tuple:
    """Compute pairwise accuracy over next N rounds.

    For each pair of solvers at time T, check if the higher-rated solver
    has better average performance over rounds T+1 through T+horizon.

    Returns:
        Tuple of (accuracy, total_pairs)
    """
    all_competitions = get_all_competitions()
    comp_to_idx = get_competition_index()

    # Add competition index
    features_df = features_df.with_columns(
        pl.struct(["year", "round", "competition"]).map_elements(
            lambda x: comp_to_idx.get((x["year"], x["round"], x["competition"]), -1),
            return_dtype=pl.Int64
        ).alias("comp_idx")
    )

    # Group by solver to get their trajectory
    solver_data = {}
    for row in features_df.iter_rows(named=True):
        solver = row['user_pseudo_id']
        if solver not in solver_data:
            solver_data[solver] = []
        solver_data[solver].append({
            'comp_idx': row['comp_idx'],
            'rating': row[predictor],
            'label': row['label'],
        })

    # Sort each solver's data by comp_idx
    for solver in solver_data:
        solver_data[solver].sort(key=lambda x: x['comp_idx'])

    # For each round, get ratings and compute future performance
    max_idx = max(row['comp_idx'] for rows in solver_data.values() for row in rows)

    total_correct = 0.0
    total_pairs = 0

    # Iterate through rounds where we can look ahead by 'horizon'
    for eval_idx in range(max_idx - horizon + 1):
        # Get solvers who have ratings at this point and data for next 'horizon' rounds
        ratings_at_eval = {}
        future_perf = {}

        for solver, data in solver_data.items():
            # Find rating at or just before eval_idx
            rating_entry = None
            for entry in data:
                if entry['comp_idx'] <= eval_idx:
                    rating_entry = entry
                else:
                    break

            if rating_entry is None:
                continue

            # Get performance in next 'horizon' rounds
            future_labels = []
            for entry in data:
                if eval_idx < entry['comp_idx'] <= eval_idx + horizon:
                    future_labels.append(entry['label'])

            if future_labels:
                ratings_at_eval[solver] = rating_entry['rating']
                future_perf[solver] = sum(future_labels) / len(future_labels)

        # Compute pairwise accuracy for solvers with both rating and future data
        solvers = list(ratings_at_eval.keys())
        for i in range(len(solvers)):
            for j in range(i + 1, len(solvers)):
                s1, s2 = solvers[i], solvers[j]
                r1, r2 = ratings_at_eval[s1], ratings_at_eval[s2]
                f1, f2 = future_perf[s1], future_perf[s2]

                if r1 == r2:
                    total_correct += 0.5
                elif (r1 > r2) == (f1 > f2):
                    total_correct += 1.0

                total_pairs += 1

    accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0
    return accuracy, total_pairs


def _compare_accuracy(normalized: pl.DataFrame, burn_in: int, horizon: int):
    """Compare pairwise prediction accuracy across methods."""
    print("Computing points-based features...", file=sys.stderr)
    adjusted = compute_adjusted_points(normalized, n_reference=16, use_gp_baseline=True)
    points_features, _ = build_features_and_labels(adjusted, min_history=3, decay_rate=0.90)

    print("Computing points-based features with prior (k=3)...", file=sys.stderr)
    prior_features, _ = build_features_with_prior(adjusted, min_history=3, decay_rate=0.90, prior_k=3)

    print("Computing percentile-based features...", file=sys.stderr)
    _, pct_adjusted = compute_percentile_ratings(normalized, n_iterations=3, min_history=3)
    pct_features, _ = build_percentile_features(pct_adjusted, min_history=3, decay_rate=0.90)

    print("Running backtests (next-round)...", file=sys.stderr)

    # Points-based predictors (burn_in=0 means evaluate from start)
    points_exp_90 = backtest_predictor(points_features, burn_in, predictor="exp_weighted_mean")
    points_exp_95 = backtest_predictor(points_features, burn_in, predictor="exp_weighted_mean_095")
    points_mean = backtest_predictor(points_features, burn_in, predictor="mean_adj_points")

    # Points with prior
    prior_exp = backtest_predictor(prior_features, burn_in, predictor="exp_weighted_mean")

    # Percentile-based predictors
    pct_exp = backtest_predictor(pct_features, burn_in, predictor="exp_weighted_pct")

    # Glicko (needs wide-format data, and needs burn-in for ratings to stabilize)
    print("Running Glicko backtest...", file=sys.stderr)
    wide_data = load_all_data()
    rounds = get_rounds_only(wide_data)
    glicko_burn_in = max(burn_in, 10)
    glicko_result = backtest_glicko(rounds, glicko_burn_in)

    # Horizon-based accuracy (next N rounds)
    print(f"Running backtests (next-{horizon} rounds)...", file=sys.stderr)
    points_exp_90_hz, points_exp_hz_pairs = compute_horizon_accuracy(
        points_features, "exp_weighted_mean", horizon)
    points_exp_95_hz, _ = compute_horizon_accuracy(
        points_features, "exp_weighted_mean_095", horizon)
    points_mean_hz, _ = compute_horizon_accuracy(
        points_features, "mean_adj_points", horizon)
    prior_exp_hz, _ = compute_horizon_accuracy(
        prior_features, "exp_weighted_mean", horizon)
    pct_exp_hz, _ = compute_horizon_accuracy(
        pct_features, "exp_weighted_pct", horizon)

    # Print results
    print(f"\n{'='*85}")
    print("Rating Method Comparison - Pairwise Prediction Accuracy")
    print(f"{'='*85}")
    if burn_in > 0:
        print(f"Burn-in rounds skipped: {burn_in}")
    print(f"{'='*85}\n")

    print(f"{'Method':<30} {'Next Round':>12} {'Next ' + str(horizon) + ' Rounds':>14}")
    print("-" * 85)

    results = [
        ("Points: Exp-Wtd (0.95)", points_exp_95.pairwise_accuracy, points_exp_95_hz),
        ("Points: Exp-Wtd (0.90)", points_exp_90.pairwise_accuracy, points_exp_90_hz),
        ("Points: With Prior (k=3)", prior_exp.pairwise_accuracy, prior_exp_hz),
        ("Points: Simple Mean", points_mean.pairwise_accuracy, points_mean_hz),
        ("Percentile: Exp-Weighted", pct_exp.pairwise_accuracy, pct_exp_hz),
        ("Glicko-1", glicko_result.accuracy, None),
    ]

    # Sort by next-round accuracy
    results.sort(key=lambda x: x[1], reverse=True)

    for method, acc1, acc_hz in results:
        hz_str = f"{acc_hz*100:>13.2f}%" if acc_hz is not None else "           N/A"
        print(f"{method:<30} {acc1*100:>11.2f}% {hz_str}")

    print("-" * 85)
    print(f"Next-round pairs: {points_exp_95.total_pairs:,}")
    print(f"Next-{horizon} pairs: {points_exp_hz_pairs:,}")
    print("\nHigher accuracy = better at predicting pairwise outcomes")


def _compare_leaderboards(normalized: pl.DataFrame, top_n: int, as_of_year):
    """Show leaderboards side-by-side for different methods."""
    print("Computing points-based ratings...", file=sys.stderr)
    adjusted = compute_adjusted_points(normalized, n_reference=16, use_gp_baseline=True)
    points_features, _ = build_features_and_labels(adjusted, min_history=3, decay_rate=0.90)

    print("Computing percentile-based ratings...", file=sys.stderr)
    _, pct_adjusted = compute_percentile_ratings(normalized, n_iterations=3, min_history=3)
    pct_features, _ = build_percentile_features(pct_adjusted, min_history=3, decay_rate=0.90)

    # Generate leaderboards
    points_lb = generate_leaderboard(points_features, top_n=top_n, as_of_year=as_of_year)
    pct_lb = generate_percentile_leaderboard(pct_features, top_n=top_n, as_of_year=as_of_year)

    year_label = as_of_year or points_features['year'].max()

    print(f"\n{'='*100}")
    print(f"Leaderboard Comparison as of {year_label}")
    print(f"{'='*100}\n")

    # Create lookup for percentile rankings
    pct_ranks = {e.solver_id: (e.rank, e.exp_weighted_pct) for e in pct_lb}

    # Also get all percentile-ranked solvers to find ranks beyond top_n
    pct_all = generate_percentile_leaderboard(pct_features, top_n=1000, as_of_year=as_of_year)
    pct_all_ranks = {e.solver_id: e.rank for e in pct_all}

    print(f"{'Pts':>4} {'Solver':<35} {'Points Rating':>14} {'Pct':>4} {'Pct Rating':>12} {'Diff':>6}")
    print("-" * 100)

    for entry in points_lb:
        solver = entry.solver_id
        # Shorten name for display
        short = solver.split(' - ')[0].split(' (')[0]
        if len(short) > 32:
            short = short[:29] + "..."

        pts_rating = entry.exp_weighted_mean

        if solver in pct_ranks:
            pct_rank, pct_rating = pct_ranks[solver]
            diff = entry.rank - pct_rank
            diff_str = f"{diff:+d}" if diff != 0 else "="
        elif solver in pct_all_ranks:
            pct_rank = pct_all_ranks[solver]
            # Need to get the actual rating
            pct_entry = next((e for e in pct_all if e.solver_id == solver), None)
            pct_rating = pct_entry.exp_weighted_pct if pct_entry else 0
            diff = entry.rank - pct_rank
            diff_str = f"{diff:+d}" if diff != 0 else "="
        else:
            pct_rank = "N/A"
            pct_rating = 0
            diff_str = ""

        pct_rank_str = str(pct_rank) if isinstance(pct_rank, int) else pct_rank

        print(f"{entry.rank:>4} {short:<35} {pts_rating:>14.1f} {pct_rank_str:>4} {pct_rating:>12.3f} {diff_str:>6}")

    print("\n" + "-" * 100)
    print("Pts = Points-based rank, Pct = Percentile-based rank")
    print("Diff = Points rank minus Percentile rank (negative = ranked higher by percentile)")


def cmd_cache(args):
    """Show cache information or purge the cache."""
    if args.purge:
        if purge_cache():
            print("Cache purged.")
        else:
            print("No cache to purge.")
        return

    info = get_cache_info()

    if not info.exists:
        print("No cache exists. Run any command to populate it.")
        return

    print(f"\nCache directory: {info.normalized_path.parent}")
    print("-" * 60)

    def format_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    if info.normalized_modified:
        print(f"Normalized data: {info.normalized_path.name}")
        print(f"  Modified: {info.normalized_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Size: {format_size(info.normalized_size)}")

    if info.merged_modified:
        print(f"Merged data: {info.merged_path.name}")
        print(f"  Modified: {info.merged_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Size: {format_size(info.merged_size)}")

    if info.difficulty_modified:
        print(f"Difficulty data: {info.difficulty_path.name}")
        print(f"  Modified: {info.difficulty_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Size: {format_size(info.difficulty_size)}")

    total_size = info.normalized_size + info.merged_size + info.difficulty_size
    print("-" * 60)
    print(f"Total cache size: {format_size(total_size)}")
    print("\nUse --purge to clear the cache.")


def cmd_records(args):
    """Show career records: #1 counts, streaks, round wins, totals."""
    method = getattr(args, 'method', 'prior')
    prior_k = 0 if method == "no-prior" else 3
    sort_by = getattr(args, 'sort', 'ones')
    top_n = getattr(args, 'top', 20)

    # Phase 1: Load data
    print("Loading data...", file=sys.stderr)
    normalized = load_normalized_data()

    print("Computing difficulty-adjusted points...", file=sys.stderr)
    adjusted = compute_adjusted_points(normalized, n_reference=16, use_gp_baseline=True)

    # Phase 2: Track #1 counts and streaks (method-dependent)
    print("Computing ratings...", file=sys.stderr)
    tracker = RatingTracker(adjusted, min_history=3, decay_rate=0.90, prior_k=prior_k)

    all_competitions = get_all_competitions()

    ones_count = {}
    best_streak = {}
    current_streak_solver = None
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

    # Phase 3: Compute round wins (method-independent)
    wins_count = {}
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

    # Phase 4: Compute career totals (method-independent)
    career_totals = (
        adjusted
        .group_by('user_pseudo_id')
        .agg([
            pl.col('adjusted_points').sum().alias('total_adj_pts'),
            pl.col('points').sum().alias('total_raw_pts'),
            pl.col('points').count().alias('total_rounds'),
        ])
    )

    totals_lookup = {}
    for row in career_totals.iter_rows(named=True):
        totals_lookup[row['user_pseudo_id']] = {
            'adj_pts': row['total_adj_pts'],
            'raw_pts': row['total_raw_pts'],
            'rounds': row['total_rounds'],
        }

    # Phase 5: Merge, sort, display
    all_solvers = set(ones_count.keys()) | set(wins_count.keys()) | set(totals_lookup.keys())

    records = []
    for solver in all_solvers:
        totals = totals_lookup.get(solver, {'adj_pts': 0, 'raw_pts': 0, 'rounds': 0})
        if totals['rounds'] < 3:
            continue

        records.append({
            'solver': solver,
            'ones': ones_count.get(solver, 0),
            'streak': best_streak.get(solver, 0),
            'wins': wins_count.get(solver, 0),
            'adj_pts': totals['adj_pts'],
            'raw_pts': totals['raw_pts'],
            'rounds': totals['rounds'],
        })

    # Sort
    sort_keys = {
        'ones': lambda r: (-r['ones'], -r['adj_pts'], r['solver']),
        'streak': lambda r: (-r['streak'], -r['adj_pts'], r['solver']),
        'wins': lambda r: (-r['wins'], -r['adj_pts'], r['solver']),
        'adj-points': lambda r: (-r['adj_pts'], r['solver']),
        'points': lambda r: (-r['raw_pts'], r['solver']),
        'rounds': lambda r: (-r['rounds'], -r['adj_pts'], r['solver']),
    }
    records.sort(key=sort_keys[sort_by])
    records = records[:top_n]

    # Display
    method_label = " (no prior)" if method == "no-prior" else ""
    print(f"\n{'='*95}")
    print(f"Career Records{method_label}")
    print(f"{'='*95}\n")

    print(f"{'Rank':<6}{'Solver':<30}{'#1s':>6}{'Streak':>8}{'Wins':>7}"
          f"{'Adj Pts':>10}{'Raw Pts':>10}{'Rounds':>8}")
    print("-" * 95)

    for i, rec in enumerate(records, 1):
        name = rec['solver']
        if len(name) > 27:
            name = name[:24] + "..."
        print(f"{i:<6}{name:<30}{rec['ones']:>6}{rec['streak']:>8}{rec['wins']:>7}"
              f"{rec['adj_pts']:>10.0f}{rec['raw_pts']:>10.0f}{rec['rounds']:>8}")

    print("-" * 95)
    print(f"\n#1s and Streak depend on rating method{method_label or ' (prior)'}.")
    print("Wins = 1st place by raw points (ties counted for each winner).")


def cmd_competitions(args):
    """Show competition statistics with difficulty relative to 2025 GP."""
    print("Loading data...", file=sys.stderr)
    normalized = load_normalized_data()

    print("Computing difficulties...", file=sys.stderr)
    difficulties = difficulty_of_all_rounds(normalized, n_reference=16, use_gp_baseline=True)

    # Build lookup for difficulties
    diff_lookup = {d.competition: d.outcome_weighted for d in difficulties}

    # Calculate 2025 GP baseline (mean difficulty of 2025 GP rounds only)
    comps_2025_gp = [d for d in difficulties
                    if d.competition.year == 2025 and d.competition.event_type == "GP"]
    if comps_2025_gp:
        valid_2025 = [d.outcome_weighted for d in comps_2025_gp
                      if d.outcome_weighted == d.outcome_weighted]
        baseline_2025_gp = sum(valid_2025) / len(valid_2025) if valid_2025 else 1.0
    else:
        baseline_2025_gp = 1.0

    # Get all competitions
    all_competitions = get_all_competitions()

    # Parse year filter
    years_filter = None
    if args.year:
        if '-' in args.year:
            start, end = args.year.split('-')
            years_filter = range(int(start), int(end) + 1)
        else:
            years_filter = [int(args.year)]

    # Apply filters
    filtered_comps = all_competitions
    if years_filter:
        filtered_comps = [c for c in filtered_comps if c.year in years_filter]
    if args.event:
        filtered_comps = [c for c in filtered_comps if c.event_type == args.event.upper()]

    if not filtered_comps:
        print("No competitions match the specified filters.", file=sys.stderr)
        sys.exit(1)

    # Compute stats for each competition
    stats = []
    for comp in filtered_comps:
        records = fetch_participant_records([comp], normalized)
        if len(records) == 0:
            continue

        points = records['points'].to_list()
        n_participants = len(points)
        max_pts = max(points)
        mean_pts = sum(points) / len(points)
        sorted_pts = sorted(points)
        median_pts = sorted_pts[len(sorted_pts) // 2]

        raw_diff = diff_lookup.get(comp, float('nan'))
        # Convert to percentage relative to 2025 GP baseline
        # Invert so higher = harder: (baseline / raw_diff - 1) * 100
        # If raw_diff < baseline, round was harder -> positive %
        # If raw_diff > baseline, round was easier -> negative %
        if raw_diff == raw_diff and raw_diff > 0:  # not NaN and positive
            rel_diff_pct = (baseline_2025_gp / raw_diff - 1) * 100
        else:
            rel_diff_pct = float('nan')

        stats.append({
            'comp': comp,
            'n': n_participants,
            'max': max_pts,
            'mean': mean_pts,
            'median': median_pts,
            'rel_diff_pct': rel_diff_pct,
        })

    # Print header
    year_label = args.year if args.year else "All Years"
    event_label = args.event.upper() if args.event else "GP + WSC"
    print(f"\nCompetition Statistics - {event_label} - {year_label}")
    print("Difficulty relative to 2025 GP (positive = harder, negative = easier)")

    def print_table_header():
        print("-" * 78)
        print(f"{'Competition':<18} {'N':>6} {'Max':>8} {'Mean':>8} {'Median':>8} {'Difficulty':>12}")
        print("-" * 78)

    current_year = None
    for s in stats:
        # Print header when year changes
        if s['comp'].year != current_year:
            current_year = s['comp'].year
            print_table_header()

        comp_label = f"{s['comp'].year} {s['comp'].event_type} R{s['comp'].round}"
        if s['rel_diff_pct'] == s['rel_diff_pct']:  # not NaN
            sign = "+" if s['rel_diff_pct'] >= 0 else ""
            diff_str = f"{sign}{s['rel_diff_pct']:.0f}%"
        else:
            diff_str = "N/A"
        print(f"{comp_label:<18} {s['n']:>6} {s['max']:>8.0f} {s['mean']:>8.1f} "
              f"{s['median']:>8.0f} {diff_str:>12}")

    # Print summary
    print("-" * 78)
    print(f"Total: {len(stats)} competitions", end="")
    valid_diffs = [s['rel_diff_pct'] for s in stats if s['rel_diff_pct'] == s['rel_diff_pct']]
    if valid_diffs:
        print(f", difficulty range: {min(valid_diffs):+.0f}% to {max(valid_diffs):+.0f}%")
    else:
        print()


def cmd_export(args):
    """Export rating data for the sudokudos website."""
    output_dir = getattr(args, 'output_dir', './export/')
    fmt = getattr(args, 'format', 'both')
    method = getattr(args, 'method', 'prior')

    stats = run_export(output_dir=output_dir, fmt=fmt, method=method)

    # Print summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"Output directory: {stats['output_dir']}")
    print(f"Method: {stats['method']}")
    print(f"Format: {stats['format']}")
    print(f"Data through: {stats['data_through']}")
    print(f"\nFiles exported:")
    print(f"  ratings_timeseries:   {stats['timeseries_rows']:>8,} rows")
    print(f"  leaderboard_current:  {stats['current_leaderboard_rows']:>8,} rows")
    print(f"  leaderboard_alltime:  {stats['alltime_leaderboard_rows']:>8,} rows")
    print(f"  records:              {stats['records_rows']:>8,} rows")
    print(f"  metadata.json")
    print(f"\nTotal unique solvers: {stats['total_solvers']:,}")
