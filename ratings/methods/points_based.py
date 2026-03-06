"""Points-based rating system using difficulty-adjusted exponentially-weighted mean.

This is the leading rating method, achieving 84.34% pairwise prediction accuracy.

The approach:
1. Adjust raw points by round difficulty (harder rounds = higher adjusted points)
2. Compute exponentially-weighted mean of adjusted points (recent rounds weighted higher)
3. Use this as the solver's rating for prediction and leaderboards
"""

from dataclasses import dataclass
from typing import Optional

import polars as pl

from ratings.competitions import CompetitionIdentifier, get_all_competitions
from ratings.competition_difficulty import difficulty_of_all_rounds
from ratings.methods.utils import sample_std, compute_exp_weighted_mean


# Minimum difficulty floor to prevent extreme inflation from low-difficulty rounds.
# Set to 0.1 to limit inflation to 10x compared to a "normal" round with difficulty 1.0.
DIFFICULTY_FLOOR = 0.1

# Default features for fitted regression (used in evaluation.py)
DEFAULT_FEATURE_COLS = [
    'mean_adj_points',
    'std_adj_points',
    'n_rounds',
    'best_adj_points',
    'prev_adj_points',
    'recent_3_mean',
    'recent_5_mean',
    'recent_10_mean',
    'exp_weighted_mean',
    'recent_vs_overall',
]


@dataclass
class SolverHistory:
    """Historical performance data for a solver."""
    adjusted_points: list[float]  # Past difficulty-adjusted points
    rounds_participated: int
    days_since_last: Optional[int]  # None if this is their first round


def compute_adjusted_points(
    normalized_data: pl.DataFrame,
    n_reference: int = 16,
    use_gp_baseline: bool = False
) -> pl.DataFrame:
    """Compute difficulty-adjusted points for all solver-round combinations.

    Args:
        normalized_data: Long-format DataFrame from load_normalized_data()
        n_reference: Number of reference rounds for difficulty calculation
        use_gp_baseline: If True, use GP rounds as baseline for all competitions.
            This normalizes WSC scores to the GP scale, fixing systematic bias
            caused by different participant pools between GP and WSC.

    Returns:
        DataFrame with columns:
        [user_pseudo_id, year, round, competition, points, difficulty, adjusted_points]
    """
    # Get difficulty for all rounds
    difficulties = difficulty_of_all_rounds(normalized_data, n_reference=n_reference,
                                            use_gp_baseline=use_gp_baseline)

    # Create a lookup dict: (year, round, competition) -> difficulty
    difficulty_lookup = {
        (d.competition.year, d.competition.round, d.competition.event_type): d.outcome_weighted
        for d in difficulties
    }

    # Add difficulty and adjusted_points columns
    def get_difficulty(year, round_num, competition):
        return difficulty_lookup.get((year, round_num, competition), 1.0)

    # Build lists for new columns
    difficulties_col = []
    adjusted_col = []

    for row in normalized_data.iter_rows(named=True):
        diff = get_difficulty(row["year"], row["round"], row["competition"])
        difficulties_col.append(diff)  # Store original difficulty for transparency
        if diff and diff > 0:
            # Apply floor to prevent extreme inflation from anomalous rounds
            capped_diff = max(diff, DIFFICULTY_FLOOR)
            adjusted_col.append(row["points"] / capped_diff)
        else:
            adjusted_col.append(None)

    result = normalized_data.with_columns([
        pl.Series("difficulty", difficulties_col),
        pl.Series("adjusted_points", adjusted_col)
    ])

    return result


def build_features_and_labels(
    adjusted_data: pl.DataFrame,
    min_history: int = 3,
    decay_rate: float = 0.90
) -> tuple[pl.DataFrame, list[CompetitionIdentifier]]:
    """Build feature matrix and labels for prediction/evaluation.

    For each solver-round combination (with sufficient history), creates:
    - Features based on historical adjusted performance
    - Label: adjusted_points for this round

    Args:
        adjusted_data: DataFrame with adjusted_points column
        min_history: Minimum past rounds required to include a sample
        decay_rate: Decay rate for exponentially weighted mean (0.90 = 10% decay per round)

    Returns:
        Tuple of (features_df, competition_order) where features_df has columns:
        [user_pseudo_id, year, round, competition, label,
         mean_adj_points, std_adj_points, n_rounds, best_adj_points, prev_adj_points,
         recent_3_mean, recent_5_mean, recent_10_mean, exp_weighted_mean, recent_vs_overall]
    """
    all_competitions = get_all_competitions()

    # Track history per solver
    solver_history: dict[str, list[float]] = {}

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
            adj_points = row["adjusted_points"]

            if adj_points is None:
                continue

            # Get solver's history (before this round)
            history = solver_history.get(solver_id, [])

            # Only include if enough history
            if len(history) >= min_history:
                recent_3 = history[-3:] if len(history) >= 3 else history
                recent_5 = history[-5:] if len(history) >= 5 else history
                recent_10 = history[-10:] if len(history) >= 10 else history

                # Exponentially weighted mean (more recent = higher weight)
                exp_weighted = compute_exp_weighted_mean(history, decay_rate)

                # Also compute with 0.95 decay (less aggressive recency weighting, for comparison)
                exp_weighted_095 = compute_exp_weighted_mean(history, 0.95)

                samples.append({
                    "user_pseudo_id": solver_id,
                    "year": comp.year,
                    "round": comp.round,
                    "competition": comp.event_type,
                    "label": adj_points,
                    # Overall stats
                    "mean_adj_points": sum(history) / len(history),
                    "std_adj_points": sample_std(history) if len(history) > 1 else 0.0,
                    "n_rounds": len(history),
                    "best_adj_points": max(history),
                    # Exp weighted with different decay rates
                    "exp_weighted_mean_095": exp_weighted_095,
                    # Recent performance
                    "prev_adj_points": history[-1],  # Immediately prior round
                    "recent_3_mean": sum(recent_3) / len(recent_3),
                    "recent_5_mean": sum(recent_5) / len(recent_5),
                    "recent_10_mean": sum(recent_10) / len(recent_10),
                    "exp_weighted_mean": exp_weighted,
                    # Trend (recent vs overall)
                    "recent_vs_overall": (sum(recent_5) / len(recent_5)) - (sum(history) / len(history)),
                })

            # Update history for next rounds
            if solver_id not in solver_history:
                solver_history[solver_id] = []
            solver_history[solver_id].append(adj_points)

    return pl.DataFrame(samples), all_competitions


def build_features_with_prior(
    adjusted_data: pl.DataFrame,
    min_history: int = 3,
    decay_rate: float = 0.90,
    prior_k: int = 1
) -> tuple[pl.DataFrame, list[CompetitionIdentifier]]:
    """Build features with pseudo-observations for regularization.

    Similar to build_features_and_labels, but prepends k pseudo-observations
    at the prior mean (population mean of all prior adjusted points) to each
    solver's history. This regularizes ratings toward the population mean,
    with stronger effect for solvers with fewer rounds.

    Args:
        adjusted_data: DataFrame with adjusted_points column
        min_history: Minimum past rounds required to include a sample
        decay_rate: Decay rate for exponentially weighted mean
        prior_k: Number of pseudo-observations at prior mean to prepend

    Returns:
        Tuple of (features_df, competition_order)
    """
    all_competitions = get_all_competitions()

    # Track history per solver
    solver_history: dict[str, list[float]] = {}

    # Track all adjusted points seen so far for computing rolling prior mean
    all_prior_points: list[float] = []

    samples = []

    for comp in all_competitions:
        # Compute prior mean from all points seen before this round
        if all_prior_points:
            prior_mean = sum(all_prior_points) / len(all_prior_points)
        else:
            prior_mean = 500.0  # Default for very first round

        # Get participants in this round
        round_data = adjusted_data.filter(
            (pl.col("year") == comp.year) &
            (pl.col("round") == comp.round) &
            (pl.col("competition") == comp.event_type)
        )

        round_points = []  # Collect points from this round to add to prior after

        for row in round_data.iter_rows(named=True):
            solver_id = row["user_pseudo_id"]
            adj_points = row["adjusted_points"]

            if adj_points is None:
                continue

            round_points.append(adj_points)

            # Get solver's history (before this round)
            history = solver_history.get(solver_id, [])

            # Only include if enough history
            if len(history) >= min_history:
                # Prepend pseudo-observations at prior mean
                regularized_history = [prior_mean] * prior_k + history

                recent_3 = history[-3:] if len(history) >= 3 else history
                recent_5 = history[-5:] if len(history) >= 5 else history

                # Exponentially weighted mean on regularized history
                exp_weighted_prior = compute_exp_weighted_mean(regularized_history, decay_rate)

                # Also compute without prior for comparison
                exp_weighted_no_prior = compute_exp_weighted_mean(history, decay_rate)

                samples.append({
                    "user_pseudo_id": solver_id,
                    "year": comp.year,
                    "round": comp.round,
                    "competition": comp.event_type,
                    "label": adj_points,
                    "prior_mean": prior_mean,
                    "mean_adj_points": sum(history) / len(history),
                    "std_adj_points": sample_std(history) if len(history) > 1 else 0.0,
                    "n_rounds": len(history),
                    "best_adj_points": max(history),
                    "prev_adj_points": history[-1],
                    "recent_3_mean": sum(recent_3) / len(recent_3),
                    "recent_5_mean": sum(recent_5) / len(recent_5),
                    # Main feature: exp weighted mean with prior
                    "exp_weighted_mean": exp_weighted_prior,
                    # For comparison: without prior
                    "exp_weighted_no_prior": exp_weighted_no_prior,
                    "recent_vs_overall": (sum(recent_5) / len(recent_5)) - (sum(history) / len(history)),
                })

            # Update history for next rounds
            if solver_id not in solver_history:
                solver_history[solver_id] = []
            solver_history[solver_id].append(adj_points)

        # Add this round's points to the prior pool
        all_prior_points.extend(round_points)

    return pl.DataFrame(samples), all_competitions
