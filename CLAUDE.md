# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python data analysis project that calculates ratings for sudoku players, updated with each additional sudoku event. As part of that, it calculates relative difficulty ratings for Sudoku competitions (Grand Prix and World Sudoku Championship) by analyzing participant performance data across multiple years.

## Rating Philosophy

**Evidence of effective ratings**

We don't have any oracle ground truth of what constitutes good ratings. But we could have some evidence to check:

- People who have never won a round should never be the ratings leader.
- Anybody who wins both GP and WSC in the same year should definitely be the ratings leader at the end of that year.
- I would be skeptical of anyone being the ratings leader without having won at least one year of GP or WSC. I could see it happening if they are consistently near the top, such as second, across events while others are less consistent.
- Tantan Dai: Tantan has won 6 of the 9 full competitions that I have on record from 2021-2025. She should be the ratings leader through most of that stretch and beyond.
- Rating stability: Established world-class solvers shouldn't see their ratings change dramatically over short periods. Kota was already best in the world by 2012—his rating shouldn't quadruple between 2016 and 2017. Moderate growth is reasonable; 4x swings indicate a calibration problem, not genuine skill change.
- Solvers can miss the occasional round without it affecting their rating. For example, in GP the scoring takes the top 6 of 8 rounds.

**Ratings are predictions, not historical summaries.**

The goal of a rating is to summarize a solver's current strength, where "current strength" means the ability to predict upcoming performance. A good rating system is one where ratings are highly correlated with performance in subsequent events.

This framing has important implications:
- **Empirically testable**: Rating systems can be compared by measuring prediction accuracy via backtesting
- **Resolves design choices**: Questions like "how much should recent results matter?" are answered by whatever predicts best
- **Handles difficulty naturally**: If difficulty adjustment improves prediction, it shows up in validation metrics

The approach:
1. Define a prediction target (points, percentile, rank in next competition)
2. Split data temporally (train on past, predict future)
3. Try candidate models (simple averages, weighted recency, regression, etc.)
4. Measure prediction error (RMSE, correlation)
5. The model that predicts best *is* the rating system

## Commands

**Run all tests:**
```bash
python -m unittest discover tests/
```

**Run a single test file:**
```bash
python -m unittest tests/test_competitions.py
```

**Run a specific test class or method:**
```bash
python -m unittest tests.test_competition_difficulty.TestRelativeDifficultySolverWeighted.test_single_solver
```

**Lint with pylint:**
```bash
pylint ratings/
```

## Development Environment

- Python virtual environment located at `./venv`
- Activate with: `source venv/bin/activate`
- Dependencies should be installed in this venv

## Architecture

### Core Data Structures

- `CompetitionIdentifier(year, round, event_type)` - NamedTuple identifying a specific competition round
- `CompetitionDifficulty(competition, outcome_weighted, n_reference, truncate_enabled)` - NamedTuple holding difficulty calculation results

### Module Responsibilities

**`ratings/data_loader.py`** - Data loading and caching
- Imports loaders from sibling `sudokudos-github` project
- `load_all_data()` returns merged GP+WSC DataFrame with unified `user_pseudo_id`
- `load_normalized_data()` returns long-format DataFrame (one row per solver-round)
- `load_gp_wsc_separate()` returns GP and WSC as separate DataFrames
- Caching functions: `get_cache_info()`, `purge_cache()`, `load_difficulty_cache()`, `save_difficulty_cache()`
- Handles name mapping between GP and WSC datasets

**`ratings/competitions.py`** - Competition metadata and scheduling
- Maintains hardcoded competition schedules for GP (2014-2026) and WSC (2010-2025)
- `get_all_competitions()` returns chronologically ordered list of all competition rounds
- `identify_n_prior_competitions()` finds N competitions immediately preceding a given competition
- `get_prior_gp_rounds()` returns all GP rounds before a given competition (for GP-baseline difficulty)

**`ratings/competition_results.py`** - Data transformation using Polars
- `normalize_all_tables(gp_df, wsc_df)` converts wide-format data to long format (one row per solver-year-competition-round)
- `normalize_gp_scoring_scale()` normalizes 2014-2015 GP scores to match 2016+ scale
- Output schema: `[user_pseudo_id, year, round, points, competition]`
- `fetch_participant_records()` filters normalized data by competition identifiers

**`ratings/competition_difficulty.py`** - Difficulty calculation algorithms
- Two weighting strategies:
  - `relative_difficulty_solver_weighted()` - equal weight per competitor
  - `relative_difficulty_outcome_weighted()` - weight by number of historical outcomes
- `difficulty_of_rounds()` - calculates difficulty for a list of competitions
- `difficulty_of_all_rounds()` - calculates difficulty for all rounds with caching support
- Higher ratio = easier competition (participants scored better than their historical average)
- First competition anchored to 1.0; insufficient history yields 1.0 (truncate mode) or NaN
- **GP-baseline mode** (`use_gp_baseline=True`): Uses only GP history as baseline for all competitions. This fixes the systematic bias where WSC scores were undervalued due to different participant pools (see DATA_ERRATA.md)

**`ratings/methods/glicko.py`** - Glicko-1 rating system implementation
- `GlickoRating` dataclass with rating and RD (rating deviation)
- `expected_score()`, `update_rating()`, `decay_rd()`, `process_round()`
- See "Design Learnings" below for known issues with this approach

**`ratings/methods/points_based.py`** - Points-based rating system (best performer at 84.34% accuracy)
- `compute_adjusted_points()` - applies difficulty adjustment to all solver-round records
- `build_features_and_labels()` - computes historical features (mean, std, recent, exp-weighted)
- `DIFFICULTY_FLOOR` constant prevents extreme inflation from anomalous rounds

**`ratings/methods/percentile_based.py`** - Percentile-based rating system (83.88% accuracy)
- `compute_percentiles()` - converts points to percentile ranks within each round
- `estimate_field_strength()` - estimates round strength from participant ratings
- `compute_adjusted_percentiles()` - adjusts percentiles by field strength
- `compute_percentile_ratings()` - iteratively computes ratings with field strength adjustment
- `build_percentile_features()` - builds features from adjusted percentile history

**`ratings/methods/rating_tracker.py`** - Shared rating computation for consistent results
- `RatingTracker` class - maintains solver histories and computes ratings incrementally
- Used by `cmd_progression`, `cmd_solver`, and `cmd_records` to ensure consistent ranking logic
- Uses shared utilities from `ratings/methods/utils.py`

**`ratings/methods/utils.py`** - Shared utilities for rating methods
- `compute_exp_weighted_mean()` - exponentially weighted mean (single source of truth)
- `sample_std()` - sample standard deviation calculation

**`ratings/competitions.py`** - Competition metadata and scheduling
- `get_competition_index()` - returns dict mapping (year, round, event_type) to chronological index

**`ratings/leaderboard.py`** - Leaderboard generation and display
- `generate_leaderboard()` - produces rankings from feature snapshots
- `print_leaderboard()` - formatted table output
- `generate_leaderboard_after_round()` - produces rankings after a specific round
- `generate_percentile_leaderboard()` - percentile-based rankings
- `print_percentile_leaderboard()` - formatted percentile leaderboard output
- `INACTIVITY_THRESHOLD_YEARS` constant (solvers drop after 1 year of inactivity)

**`ratings/evaluation.py`** - Backtesting and evaluation framework
- `extract_pairwise()` - convert round results to winner/loser pairs
- `pairwise_accuracy()` - measure prediction accuracy
- `backtest_glicko()` - Glicko backtest pipeline
- `grid_search_glicko()` - Glicko parameter tuning
- `backtest_predictor()` - evaluates single-feature predictors via pairwise accuracy
- `backtest_percentile_predictor()` - convenience wrapper for percentile predictors
- `backtest_fitted_regression()` - evaluates fitted Ridge regression models
- `compare_models()` - compares different regression configurations

**`ratings/backtest_bridge.py`** - Data format conversion
- `extract_rounds_for_backtest()` - converts loaded DataFrame to chronological round results
- `get_rounds_only()` - simplified version for backtest_glicko()
- `summarize_rounds()` - prints summary statistics of extracted rounds

**`ratings/methods/utils.py`** - Shared utilities for rating methods
- `sample_std()` - sample standard deviation calculation (used by points_based and percentile_based)

**`ratings/cli/commands.py`** - CLI command implementations
- `load_features()` - loads data and computes features for CLI commands
- `cmd_leaderboard()` - generates and prints leaderboard
- `cmd_progression()` - shows leadership progression over time
- `cmd_solver()` - shows rating progression for a specific solver
- `cmd_compare()` - compares rating methods (accuracy or leaderboards)
- `cmd_cache()` - shows cache info or purges cache
- `cmd_competitions()` - shows competition statistics
- `cmd_records()` - shows career records (#1 counts, longest #1 streak, round wins, total adjusted/raw points, total rounds); supports `--sort`, `--method`, `--top`
- `compute_horizon_accuracy()` - computes pairwise accuracy over N future rounds

**`main.py`** - CLI entry point
- Argument parsing with argparse
- Command routing to `ratings.cli.commands` functions
- No business logic - purely parsing and dispatch

### Data Flow

**Difficulty calculation:**
```
Raw CSV data (sudokudos-github/data/)
    → load_all_data() → merged solver-year DataFrame with unified user_pseudo_id
    → normalize_all_tables() → long-format DataFrame (one row per solver-round)
    → fetch_participant_records() → filtered by competition
    → relative_difficulty_*_weighted() → difficulty ratio
    → difficulty_of_rounds() → list of CompetitionDifficulty
```

**Rating computation:**
```
load_normalized_data() → long-format DataFrame
    → compute_adjusted_points() → adds difficulty and adjusted_points columns
    → build_features_and_labels() → features per solver-round (exp_weighted_mean, etc.)
    → generate_leaderboard() → ranked list of solvers
```

## Key Details

- Uses Polars for DataFrame operations (recently converted from pandas)
- External data loaded from sibling `sudokudos-github` directory
- Competition schedules are hardcoded and need manual updates when new years are added
- WSC has irregular round numbering (gaps between rounds 7-10 in some years)
- No package configuration (pyproject.toml, setup.py) - run directly with Python path including project root

## Data Quality

See **[DATA_ERRATA.md](DATA_ERRATA.md)** for detailed documentation of:
- GP vs WSC scoring incompatibility (difficulty adjustment doesn't fully normalize)
- Anomalous WSC rounds (~10% have unusual formats like all-or-nothing scoring)
- WSC score distribution variance (max scores vary 5x across rounds)
- Recommendations for handling problematic data

## Design Learnings

### Glicko Does Not Work for Mass-Start Competitions

**Problem:** Standard Glicko produces wildly unstable ratings (swings of 50,000+ points, negative ratings) when applied to sudoku competition data.

**Root cause:** Glicko was designed for 1v1 matchups (chess), not mass-start events. When treating a round with 500+ participants as simultaneous pairwise "games":

1. A player with low rating who finishes mid-pack "beats" 300+ opponents
2. Each opponent contributes to the rating update
3. The cumulative update explodes because expected score vs each was near 0%

**Example from data:**
```
Anna Szymańska:
- 2024 GP R1: Finished 438th of 1025, rating went 12,535 → -60,749
- 2024 GP R6: Finished 137th of 633, rating went -60,749 → 41,802
```

**Possible solutions (not yet implemented):**
1. **Cap opponents per round** - only consider N random or closest-rated opponents
2. **Scale by field size** - divide update contribution by number of opponents
3. **Use percentile outcomes** - model finish position / field size instead of pairwise wins
4. **Regression approach** - learn a rating function from features rather than using Glicko's formula

**Current status:** Glicko implementation exists in `ratings/methods/glicko.py` and achieves 81.2% pairwise accuracy despite the instability, but the rating values themselves are not meaningful. A different approach is needed for interpretable ratings.

### Current Best Method: Exponentially Weighted Difficulty-Adjusted Points

**Implementation:** `ratings/methods/points_based.py` (features), `ratings/leaderboard.py` (rankings), `ratings/evaluation.py` (backtesting)

**Performance:** 84.34% pairwise accuracy (best among tested approaches)

**How it works:**

1. **Difficulty adjustment**: Each round's raw points are divided by a difficulty ratio calculated from how participants performed relative to their historical averages in prior rounds. Uses GP-baseline mode (`use_gp_baseline=True`) to normalize WSC scores to the GP scale.

2. **Feature computation**: For each solver with ≥3 rounds of history, compute statistics from their difficulty-adjusted points history.

3. **Exponentially weighted mean**: The rating is the exponentially weighted mean of past adjusted points, with decay rate 0.9 (most recent rounds weighted highest).

**Key functions:**
- `compute_adjusted_points()` (points_based.py) - applies difficulty adjustment to all rounds
- `build_features_and_labels()` (points_based.py) - computes historical features per solver-round
- `backtest_predictor()` (evaluation.py) - evaluates prediction accuracy
- `generate_leaderboard()` (leaderboard.py) - produces year-end rankings
- `generate_leaderboard_after_round()` (leaderboard.py) - produces rankings after a specific round
- `print_leaderboard_progression()` (leaderboard.py) - shows top N after each round with change markers

**Parameters:**
- `n_reference=16` - number of prior rounds for difficulty calculation
- `min_history=3` - minimum rounds before a solver gets a rating
- `decay_rate=0.90` - exponential decay (10% per round)
- `DIFFICULTY_FLOOR=0.1` (points_based.py) - prevents extreme inflation from anomalous rounds
- `INACTIVITY_THRESHOLD_YEARS=1` (leaderboard.py) - solvers drop from leaderboard after 1 full year of inactivity

**Historical leaders (year-end, with 1-year inactivity threshold):**
- 2010: Jan Mrozowski (volatile early data)
- 2011-2012: Thomas Snyder
- 2013: No competitions held (Snyder would drop due to inactivity by 2014)
- 2014: Tiit Vunk
- 2015-2017: Tiit Vunk
- 2018-2020: Kota Morinishi
- 2021: Tiit Vunk (very close race)
- 2022: Kota Morinishi
- 2023-2025: Tantan Dai

**Comparison with other approaches:**

| Method | Pairwise Accuracy |
|--------|-------------------|
| Exp-weighted difficulty-adjusted points | **84.34%** |
| Mean difficulty-adjusted points | 83.81% |
| Exp-weighted percentile (field-strength adjusted) | 83.88% |
| Glicko-1 | 81.2% |

**Limitations:**
- Ratings are in arbitrary "adjusted points" units (typically 600-1000 for top solvers)
- New solvers need 3 rounds before appearing in rankings
- Hot start issue: solvers with only 3 exceptional rounds can briefly top rankings before regressing
- Solvers drop from leaderboard after 1 full calendar year of inactivity
