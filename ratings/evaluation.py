"""Evaluation framework for rating systems.

Provides tools for:
- Extracting pairwise comparisons from competition results
- Measuring prediction accuracy
- Backtesting rating systems on historical data
- Backtesting feature-based predictors
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ratings.competitions import get_all_competitions, get_competition_index
from ratings.methods.glicko import GlickoRating, expected_score, process_round


@dataclass
class PairwiseResult:
    """A single pairwise comparison from a competition."""
    winner_id: str
    loser_id: str
    is_tie: bool = False


@dataclass
class PairwisePrediction:
    """A prediction for a pairwise matchup."""
    player_a_id: str
    player_b_id: str
    prob_a_wins: float  # Probability player A beats player B


def extract_pairwise(results: list[tuple[str, float]]) -> list[PairwiseResult]:
    """Extract all pairwise comparisons from a round's results.

    Args:
        results: List of (player_id, points) tuples

    Returns:
        List of PairwiseResult for each pair where first player beat or tied second
    """
    pairs = []

    for i, (player_a, points_a) in enumerate(results):
        for j, (player_b, points_b) in enumerate(results):
            if i >= j:
                continue

            if points_a > points_b:
                pairs.append(PairwiseResult(
                    winner_id=player_a,
                    loser_id=player_b,
                    is_tie=False
                ))
            elif points_b > points_a:
                pairs.append(PairwiseResult(
                    winner_id=player_b,
                    loser_id=player_a,
                    is_tie=False
                ))
            else:
                # Tie - record once with arbitrary ordering
                pairs.append(PairwiseResult(
                    winner_id=player_a,
                    loser_id=player_b,
                    is_tie=True
                ))

    return pairs


def predict_pairwise(
    pairs: list[PairwiseResult],
    ratings: dict[str, GlickoRating]
) -> list[PairwisePrediction]:
    """Generate predictions for pairwise matchups based on current ratings.

    Args:
        pairs: List of pairwise results (we use this to know which pairs to predict)
        ratings: Current ratings for players

    Returns:
        List of predictions with probability estimates
    """
    predictions = []
    default_rating = GlickoRating()

    for pair in pairs:
        rating_a = ratings.get(pair.winner_id, default_rating)
        rating_b = ratings.get(pair.loser_id, default_rating)

        prob_a_wins = expected_score(rating_a, rating_b)

        predictions.append(PairwisePrediction(
            player_a_id=pair.winner_id,
            player_b_id=pair.loser_id,
            prob_a_wins=prob_a_wins
        ))

    return predictions


def pairwise_accuracy(
    predictions: list[PairwisePrediction],
    actuals: list[PairwiseResult]
) -> float:
    """Calculate accuracy of pairwise predictions.

    A prediction is correct if:
    - prob_a_wins > 0.5 and A actually won
    - prob_a_wins < 0.5 and B actually won
    - prob_a_wins == 0.5 counts as 0.5 correct (tie prediction)
    - Actual ties count as 0.5 correct for any prediction

    Args:
        predictions: List of predictions (must align with actuals)
        actuals: List of actual results

    Returns:
        Accuracy as fraction between 0 and 1
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    if not predictions:
        return 0.0

    correct = 0.0

    for pred, actual in zip(predictions, actuals):
        # Verify alignment
        if pred.player_a_id != actual.winner_id or pred.player_b_id != actual.loser_id:
            raise ValueError("Prediction and actual pairs don't match")

        if actual.is_tie:
            # Ties count as half correct regardless of prediction
            correct += 0.5
        elif pred.prob_a_wins > 0.5:
            # Predicted A wins, A is in winner position
            correct += 1.0
        elif pred.prob_a_wins < 0.5:
            # Predicted B wins, but A is in winner position = wrong
            correct += 0.0
        else:
            # Predicted tie (0.5), actual was decisive = half credit
            correct += 0.5

    return correct / len(predictions)


@dataclass
class GlickoBacktestResult:
    """Results from backtesting a Glicko rating system."""
    accuracy: float
    total_pairs: int
    rounds_evaluated: int
    final_ratings: dict[str, GlickoRating]


# Legacy alias
BacktestResult = GlickoBacktestResult


def backtest_glicko(
    rounds: list[list[tuple[str, float]]],
    train_rounds: int,
    c: float = 34.6
) -> GlickoBacktestResult:
    """Backtest Glicko rating system on historical data.

    Args:
        rounds: List of rounds, each round is list of (player_id, points) tuples
        train_rounds: Number of initial rounds to train on (no evaluation)
        c: RD decay constant

    Returns:
        GlickoBacktestResult with accuracy and other metrics
    """
    if train_rounds >= len(rounds):
        raise ValueError("train_rounds must be less than total rounds")

    ratings: dict[str, GlickoRating] = {}

    # Training phase - just update ratings, no evaluation
    for round_idx in range(train_rounds):
        ratings = process_round(rounds[round_idx], ratings, c=c)

    # Evaluation phase - predict then update
    total_correct = 0.0
    total_pairs = 0
    rounds_evaluated = 0

    for round_idx in range(train_rounds, len(rounds)):
        round_results = rounds[round_idx]

        # Extract pairs and predict before seeing results
        pairs = extract_pairwise(round_results)
        if not pairs:
            continue

        predictions = predict_pairwise(pairs, ratings)

        # Calculate accuracy for this round
        round_correct = 0.0
        for pred, actual in zip(predictions, pairs):
            if actual.is_tie:
                round_correct += 0.5
            elif pred.prob_a_wins > 0.5:
                round_correct += 1.0
            elif pred.prob_a_wins == 0.5:
                round_correct += 0.5

        total_correct += round_correct
        total_pairs += len(pairs)
        rounds_evaluated += 1

        # Now update ratings with actual results
        ratings = process_round(round_results, ratings, c=c)

    accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0

    return GlickoBacktestResult(
        accuracy=accuracy,
        total_pairs=total_pairs,
        rounds_evaluated=rounds_evaluated,
        final_ratings=ratings
    )


def grid_search_glicko(
    rounds: list[list[tuple[str, float]]],
    train_rounds: int,
    c_values: list[float],
) -> tuple[float, float]:
    """Find optimal Glicko parameter via grid search.

    Args:
        rounds: Historical round data
        train_rounds: Number of training rounds
        c_values: List of c values to try

    Returns:
        Tuple of (best_c, best_accuracy)
    """
    best_c = c_values[0]
    best_accuracy = 0.0

    for c in c_values:
        result = backtest_glicko(rounds, train_rounds, c=c)
        if result.accuracy > best_accuracy:
            best_accuracy = result.accuracy
            best_c = c

    return best_c, best_accuracy


# =============================================================================
# Feature-based backtest functions (moved from regression.py)
# =============================================================================


@dataclass
class FeatureBacktestResult:
    """Results from feature predictor backtest."""
    pairwise_accuracy: float
    total_pairs: int
    rounds_evaluated: int
    mean_absolute_error: float
    predictions: pl.DataFrame  # For analysis


# Legacy aliases for backward compatibility
PercentileBacktestResult = FeatureBacktestResult


def backtest_predictor(
    features_df: pl.DataFrame,
    train_rounds: int,
    predictor: Optional[str] = None
) -> FeatureBacktestResult:
    """Backtest a single feature as a predictor.

    Uses the specified feature directly as the prediction (no fitting).
    For multi-feature regression, use backtest_fitted_regression instead.

    Works with both points-based features (from build_features_and_labels)
    and percentile-based features (from build_percentile_features).

    Args:
        features_df: DataFrame from build_features_and_labels() or build_percentile_features()
        train_rounds: Number of initial rounds to skip (burn-in period)
        predictor: Feature column to use as predictor (default: "mean_adj_points")

    Returns:
        FeatureBacktestResult with accuracy metrics
    """
    if predictor is None:
        predictor = "mean_adj_points"

    predictor_col = predictor
    comp_to_idx = get_competition_index()

    features_df = features_df.with_columns(
        pl.struct(["year", "round", "competition"]).map_elements(
            lambda x: comp_to_idx.get((x["year"], x["round"], x["competition"]), -1),
            return_dtype=pl.Int64
        ).alias("round_idx")
    )

    # Evaluation: for rounds after train_rounds
    eval_df = features_df.filter(pl.col("round_idx") >= train_rounds)

    if len(eval_df) == 0:
        return FeatureBacktestResult(
            pairwise_accuracy=0.0,
            total_pairs=0,
            rounds_evaluated=0,
            mean_absolute_error=0.0,
            predictions=pl.DataFrame()
        )

    # Add predictions using specified feature
    eval_df = eval_df.with_columns(
        pl.col(predictor_col).alias("prediction")
    )

    # Calculate MAE
    eval_df = eval_df.with_columns(
        (pl.col("label") - pl.col("prediction")).abs().alias("abs_error")
    )
    mae = eval_df["abs_error"].mean()

    # Calculate pairwise accuracy per round
    total_correct = 0.0
    total_pairs = 0
    rounds_evaluated = 0

    eval_rounds = eval_df.select(["year", "round", "competition"]).unique()

    for row in eval_rounds.iter_rows(named=True):
        round_df = eval_df.filter(
            (pl.col("year") == row["year"]) &
            (pl.col("round") == row["round"]) &
            (pl.col("competition") == row["competition"])
        )

        if len(round_df) < 2:
            continue

        rounds_evaluated += 1

        # Extract predictions and actuals
        predictions = round_df["prediction"].to_list()
        actuals = round_df["label"].to_list()

        # Count pairwise accuracy
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i_better = predictions[i] > predictions[j]
                actual_i_better = actuals[i] > actuals[j]

                if predictions[i] == predictions[j]:
                    total_correct += 0.5
                elif pred_i_better == actual_i_better:
                    total_correct += 1.0

                total_pairs += 1

    pairwise_accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0

    return FeatureBacktestResult(
        pairwise_accuracy=pairwise_accuracy,
        total_pairs=total_pairs,
        rounds_evaluated=rounds_evaluated,
        mean_absolute_error=mae,
        predictions=eval_df
    )


# Legacy alias for percentile backtest
def backtest_percentile_predictor(
    features_df: pl.DataFrame,
    train_rounds: int,
    predictor: str = "exp_weighted_pct"
) -> FeatureBacktestResult:
    """Backtest a percentile-based predictor using pairwise accuracy.

    This is a convenience wrapper around backtest_predictor.
    """
    return backtest_predictor(features_df, train_rounds, predictor)


@dataclass
class FittedRegressionResult:
    """Results from fitted regression backtest."""
    pairwise_accuracy: float
    total_pairs: int
    rounds_evaluated: int
    mean_absolute_error: float
    model: Ridge
    scaler: StandardScaler
    feature_cols: list[str]
    coefficients: dict[str, float]


# Default features for fitted regression
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


def _calc_pairwise_accuracy(
    eval_rows: list[dict],
    y_eval: np.ndarray,
    predictions: np.ndarray
) -> tuple[float, int, int]:
    """Calculate pairwise accuracy from predictions.

    Args:
        eval_rows: List of row dicts with year, round, competition keys
        y_eval: Actual labels
        predictions: Model predictions

    Returns:
        Tuple of (accuracy, total_pairs, rounds_evaluated)
    """
    # Build round groups
    rounds = {}
    for i, row in enumerate(eval_rows):
        key = (row['year'], row['round'], row['competition'])
        if key not in rounds:
            rounds[key] = []
        rounds[key].append(i)

    total_correct = 0.0
    total_pairs = 0
    rounds_evaluated = 0

    for key, indices in rounds.items():
        if len(indices) < 2:
            continue
        rounds_evaluated += 1
        preds = [predictions[i] for i in indices]
        actuals = [y_eval[i] for i in indices]

        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if preds[i] == preds[j]:
                    total_correct += 0.5
                elif (preds[i] > preds[j]) == (actuals[i] > actuals[j]):
                    total_correct += 1.0
                total_pairs += 1

    accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0
    return accuracy, total_pairs, rounds_evaluated


def backtest_fitted_regression(
    features_df: pl.DataFrame,
    train_rounds: int,
    feature_cols: Optional[list[str]] = None,
    alpha: float = 10.0
) -> FittedRegressionResult:
    """Backtest a fitted Ridge regression model.

    Args:
        features_df: DataFrame from build_features_and_labels()
        train_rounds: Number of initial rounds for training
        feature_cols: Features to use (default: DEFAULT_FEATURE_COLS)
        alpha: Ridge regularization strength

    Returns:
        FittedRegressionResult with accuracy metrics and model
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    comp_to_idx = get_competition_index()

    features_df = features_df.with_columns([
        features_df.select(['year', 'round', 'competition']).map_rows(
            lambda x: comp_to_idx.get((x[0], x[1], x[2]), -1)
        ).to_series().alias('round_idx')
    ])

    # Split train/eval
    train_df = features_df.filter(pl.col('round_idx') < train_rounds)
    eval_df = features_df.filter(pl.col('round_idx') >= train_rounds)

    # Prepare arrays
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df['label'].to_numpy()
    X_eval = eval_df.select(feature_cols).to_numpy()
    y_eval = eval_df['label'].to_numpy()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Fit model
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    # Predict
    predictions = model.predict(X_eval_scaled)

    # Calculate metrics
    eval_rows = list(eval_df.iter_rows(named=True))
    accuracy, total_pairs, rounds_evaluated = _calc_pairwise_accuracy(
        eval_rows, y_eval, predictions
    )

    mae = float(np.mean(np.abs(y_eval - predictions)))

    # Build coefficients dict
    coefficients = {feat: coef for feat, coef in zip(feature_cols, model.coef_)}
    coefficients['intercept'] = model.intercept_

    return FittedRegressionResult(
        pairwise_accuracy=accuracy,
        total_pairs=total_pairs,
        rounds_evaluated=rounds_evaluated,
        mean_absolute_error=mae,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        coefficients=coefficients
    )


def compare_models(
    features_df: pl.DataFrame,
    train_rounds: int,
    alphas: Optional[list[float]] = None
) -> dict[str, float]:
    """Compare different regression configurations.

    Args:
        features_df: DataFrame from build_features_and_labels()
        train_rounds: Number of initial rounds for training
        alphas: List of Ridge alpha values to try

    Returns:
        Dict mapping model name to pairwise accuracy
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]

    results = {}

    for alpha in alphas:
        result = backtest_fitted_regression(
            features_df, train_rounds, alpha=alpha
        )
        results[f'Ridge(alpha={alpha})'] = result.pairwise_accuracy

    return results
