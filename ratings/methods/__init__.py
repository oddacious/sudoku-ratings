"""Rating methods for sudoku competitions.

This package contains different approaches to computing solver ratings:
- points_based: Difficulty-adjusted exponentially-weighted mean (the leading method)
- percentile_based: Field-strength-adjusted percentile ratings
- glicko: Glicko-1 rating system (experimental, has issues with mass-start events)
"""

from ratings.methods.utils import sample_std

from ratings.methods.points_based import (
    DIFFICULTY_FLOOR,
    DEFAULT_FEATURE_COLS,
    SolverHistory,
    compute_adjusted_points,
    build_features_and_labels,
    build_features_with_prior,
)

from ratings.methods.percentile_based import (
    DEFAULT_PERCENTILE_RATING,
    FIELD_STRENGTH_K,
    REFERENCE_STRENGTH,
    N_ITERATIONS,
    BOOTSTRAP_FIELD_STRENGTH,
    compute_percentiles,
    estimate_field_strength,
    compute_adjusted_percentiles,
    compute_percentile_ratings,
    build_percentile_features,
)

from ratings.methods.glicko import (
    GlickoRating,
    expected_score,
    update_rating,
    decay_rd,
    process_round,
)

__all__ = [
    # Shared utilities
    'sample_std',
    # Points-based
    'DIFFICULTY_FLOOR',
    'DEFAULT_FEATURE_COLS',
    'SolverHistory',
    'compute_adjusted_points',
    'build_features_and_labels',
    'build_features_with_prior',
    # Percentile-based
    'DEFAULT_PERCENTILE_RATING',
    'FIELD_STRENGTH_K',
    'REFERENCE_STRENGTH',
    'N_ITERATIONS',
    'BOOTSTRAP_FIELD_STRENGTH',
    'compute_percentiles',
    'estimate_field_strength',
    'compute_adjusted_percentiles',
    'compute_percentile_ratings',
    'build_percentile_features',
    # Glicko
    'GlickoRating',
    'expected_score',
    'update_rating',
    'decay_rd',
    'process_round',
]
