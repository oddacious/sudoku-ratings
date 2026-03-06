"""Percentile-based rating system.

An alternative rating approach that uses percentile finishes rather than
raw points, adjusted for field strength. Achieves 83.88% pairwise accuracy.

The approach:
1. Convert raw points to percentile finish within each round
2. Estimate field strength from participant ratings
3. Adjust percentiles by field strength (stronger field = higher adjusted percentile)
4. Compute exponentially-weighted mean of adjusted percentiles
"""

import polars as pl

from ratings.competitions import CompetitionIdentifier, get_all_competitions
from ratings.methods.utils import sample_std, compute_exp_weighted_mean


# =============================================================================
# Percentile-based rating system parameters
# =============================================================================

# Default rating for solvers without history (50th percentile equivalent)
DEFAULT_PERCENTILE_RATING = 0.5

# Adjustment coefficient for field strength
FIELD_STRENGTH_K = 0.3

# Reference field strength (GP average as baseline)
REFERENCE_STRENGTH = 0.5

# Number of iterations for convergence
N_ITERATIONS = 3

# Bootstrap field strengths for first rounds
BOOTSTRAP_FIELD_STRENGTH = {
    "GP": 0.5,
    "WSC": 0.75,
}


def compute_percentiles(normalized_data: pl.DataFrame) -> pl.DataFrame:
    """Add rank, field_size, and raw_percentile columns to normalized data.

    Handles ties via average rank (two tied for 3rd both get 3.5).
    DNF (zero points) included in ranking with low percentile.

    Args:
        normalized_data: Long-format DataFrame with columns:
            [user_pseudo_id, year, round, competition, points]

    Returns:
        DataFrame with additional columns: rank, field_size, raw_percentile
    """
    # Group by competition round and compute rank within each round
    # Using average rank for ties
    result = normalized_data.with_columns([
        # Rank within each round (higher points = lower rank number = better)
        pl.col("points").rank(method="average", descending=True).over(
            ["year", "round", "competition"]
        ).alias("rank"),
        # Field size for each round
        pl.len().over(["year", "round", "competition"]).alias("field_size"),
    ])

    # Compute raw percentile: 1 - (rank - 1) / (field_size - 1)
    # Winner (rank=1) gets percentile=1.0, last place gets percentile=0.0
    # For single-participant rounds, assign percentile=1.0
    result = result.with_columns(
        pl.when(pl.col("field_size") == 1)
        .then(pl.lit(1.0))
        .otherwise(
            1.0 - (pl.col("rank") - 1.0) / (pl.col("field_size") - 1.0)
        )
        .alias("raw_percentile")
    )

    return result


def estimate_field_strength(
    percentile_data: pl.DataFrame,
    ratings: dict[str, float],
    default_rating: float = DEFAULT_PERCENTILE_RATING
) -> dict[tuple[int, int, str], float]:
    """Estimate round strength from participant ratings.

    Field strength is the average rating of participants in a round.

    Args:
        percentile_data: DataFrame with percentile columns
        ratings: Dict mapping user_pseudo_id -> rating
        default_rating: Rating to use for unknown solvers

    Returns:
        Dict mapping (year, round, competition) -> field_strength
    """
    field_strengths = {}

    # Get unique rounds
    rounds = percentile_data.select(
        ["year", "round", "competition"]
    ).unique()

    for row in rounds.iter_rows(named=True):
        year, round_num, competition = row["year"], row["round"], row["competition"]

        # Get participants in this round
        participants = percentile_data.filter(
            (pl.col("year") == year) &
            (pl.col("round") == round_num) &
            (pl.col("competition") == competition)
        )["user_pseudo_id"].to_list()

        if not participants:
            # Use bootstrap value for empty rounds
            field_strengths[(year, round_num, competition)] = BOOTSTRAP_FIELD_STRENGTH.get(
                competition, DEFAULT_PERCENTILE_RATING
            )
            continue

        # Calculate mean rating of participants
        participant_ratings = [
            ratings.get(p, default_rating) for p in participants
        ]
        field_strengths[(year, round_num, competition)] = sum(participant_ratings) / len(participant_ratings)

    return field_strengths


def compute_adjusted_percentiles(
    percentile_data: pl.DataFrame,
    field_strengths: dict[tuple[int, int, str], float],
    reference_strength: float = REFERENCE_STRENGTH,
    k: float = FIELD_STRENGTH_K
) -> pl.DataFrame:
    """Adjust percentiles by field strength.

    Formula: adj_pct = raw_pct + k * (field_strength - reference)
    Capped to [0, 1] range.

    Args:
        percentile_data: DataFrame with raw_percentile column
        field_strengths: Dict mapping (year, round, competition) -> field_strength
        reference_strength: Baseline field strength (typically GP average)
        k: Adjustment coefficient

    Returns:
        DataFrame with additional columns: field_strength, adjusted_percentile
    """
    # Build field strength column
    field_strength_col = []
    for row in percentile_data.iter_rows(named=True):
        key = (row["year"], row["round"], row["competition"])
        fs = field_strengths.get(key, reference_strength)
        field_strength_col.append(fs)

    result = percentile_data.with_columns(
        pl.Series("field_strength", field_strength_col)
    )

    # Compute adjusted percentile with capping
    result = result.with_columns(
        (pl.col("raw_percentile") + k * (pl.col("field_strength") - reference_strength))
        .clip(0.0, 1.0)
        .alias("adjusted_percentile")
    )

    return result


def compute_percentile_ratings(
    normalized_data: pl.DataFrame,
    n_iterations: int = N_ITERATIONS,
    min_history: int = 3
) -> tuple[dict[str, float], pl.DataFrame]:
    """Iteratively compute ratings with field strength adjustment.

    Pass 0: Raw percentiles -> initial ratings (mean raw percentile per solver)
    Pass 1-N: Field strength -> adjusted percentiles -> updated ratings

    Args:
        normalized_data: Long-format DataFrame from load_normalized_data()
        n_iterations: Number of refinement iterations (3 typically converges)
        min_history: Minimum rounds for a solver to have a stable rating

    Returns:
        Tuple of (ratings dict, final adjusted DataFrame)
    """
    # Phase 1: Compute raw percentiles
    data = compute_percentiles(normalized_data)

    # Initial ratings: mean raw percentile per solver
    solver_means = (
        data
        .group_by("user_pseudo_id")
        .agg(pl.col("raw_percentile").mean().alias("mean_pct"))
    )
    ratings = {
        row["user_pseudo_id"]: row["mean_pct"]
        for row in solver_means.iter_rows(named=True)
    }

    # Iterative refinement
    for iteration in range(n_iterations):
        # Estimate field strength from current ratings
        field_strengths = estimate_field_strength(data, ratings)

        # Adjust percentiles
        data = compute_adjusted_percentiles(data, field_strengths)

        # Update ratings based on adjusted percentiles
        solver_means = (
            data
            .group_by("user_pseudo_id")
            .agg([
                pl.col("adjusted_percentile").mean().alias("mean_pct"),
                pl.len().alias("n_rounds")
            ])
        )
        ratings = {
            row["user_pseudo_id"]: row["mean_pct"]
            for row in solver_means.iter_rows(named=True)
            if row["n_rounds"] >= min_history
        }

    return ratings, data


def build_percentile_features(
    adjusted_data: pl.DataFrame,
    min_history: int = 3,
    decay_rate: float = 0.90
) -> tuple[pl.DataFrame, list[CompetitionIdentifier]]:
    """Build feature matrix and labels for percentile-based prediction.

    For each solver-round combination (with sufficient history), creates:
    - Features based on historical adjusted percentile performance
    - Label: raw_percentile for this round (used for pairwise accuracy evaluation)

    Args:
        adjusted_data: DataFrame with adjusted_percentile column
        min_history: Minimum past rounds required to include a sample
        decay_rate: Decay rate for exponentially weighted mean

    Returns:
        Tuple of (features_df, competition_order) where features_df has columns:
        [user_pseudo_id, year, round, competition, label,
         mean_adj_percentile, std_adj_percentile, n_rounds, best_adj_percentile,
         prev_adj_percentile, recent_3_pct, recent_5_pct, exp_weighted_pct,
         avg_field_strength, pct_in_wsc, pct_vs_overall]
    """
    all_competitions = get_all_competitions()

    # Track history per solver
    solver_pct_history: dict[str, list[float]] = {}
    solver_field_strengths: dict[str, list[float]] = {}
    solver_wsc_count: dict[str, int] = {}
    solver_total_count: dict[str, int] = {}

    samples = []

    for comp in all_competitions:
        # Get participants in this round
        round_data = adjusted_data.filter(
            (pl.col("year") == comp.year) &
            (pl.col("round") == comp.round) &
            (pl.col("competition") == comp.event_type)
        )

        for row in round_data.iter_rows(named=True):
            solver_id = row["user_pseudo_id"]
            adj_pct = row.get("adjusted_percentile")
            raw_pct = row.get("raw_percentile")
            field_strength = row.get("field_strength", REFERENCE_STRENGTH)

            if adj_pct is None or raw_pct is None:
                continue

            # Get solver's history (before this round)
            pct_history = solver_pct_history.get(solver_id, [])
            fs_history = solver_field_strengths.get(solver_id, [])
            wsc_count = solver_wsc_count.get(solver_id, 0)
            total_count = solver_total_count.get(solver_id, 0)

            # Only include if enough history
            if len(pct_history) >= min_history:
                recent_3 = pct_history[-3:] if len(pct_history) >= 3 else pct_history
                recent_5 = pct_history[-5:] if len(pct_history) >= 5 else pct_history

                # Exponentially weighted mean (more recent = higher weight)
                exp_weighted = compute_exp_weighted_mean(pct_history, decay_rate)

                samples.append({
                    "user_pseudo_id": solver_id,
                    "year": comp.year,
                    "round": comp.round,
                    "competition": comp.event_type,
                    "label": raw_pct,  # Use raw percentile as label for evaluation
                    # Overall stats
                    "mean_adj_percentile": sum(pct_history) / len(pct_history),
                    "std_adj_percentile": sample_std(pct_history) if len(pct_history) > 1 else 0.0,
                    "n_rounds": len(pct_history),
                    "best_adj_percentile": max(pct_history),
                    # Recent performance
                    "prev_adj_percentile": pct_history[-1],
                    "recent_3_pct": sum(recent_3) / len(recent_3),
                    "recent_5_pct": sum(recent_5) / len(recent_5),
                    "exp_weighted_pct": exp_weighted,
                    # Field strength history
                    "avg_field_strength": sum(fs_history) / len(fs_history) if fs_history else REFERENCE_STRENGTH,
                    # Competition mix
                    "pct_in_wsc": wsc_count / total_count if total_count > 0 else 0.0,
                    # Trend
                    "pct_vs_overall": (sum(recent_5) / len(recent_5)) - (sum(pct_history) / len(pct_history)),
                })

            # Update history for next rounds
            if solver_id not in solver_pct_history:
                solver_pct_history[solver_id] = []
                solver_field_strengths[solver_id] = []
                solver_wsc_count[solver_id] = 0
                solver_total_count[solver_id] = 0

            solver_pct_history[solver_id].append(adj_pct)
            solver_field_strengths[solver_id].append(field_strength)
            solver_total_count[solver_id] += 1
            if comp.event_type == "WSC":
                solver_wsc_count[solver_id] += 1

    return pl.DataFrame(samples), all_competitions
