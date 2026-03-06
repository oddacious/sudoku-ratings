"""Utilities for evaluating the relative difficulty of competitions.
"""
import typing

import numpy as np
import polars as pl

from ratings.competitions import (
    CompetitionIdentifier, get_all_competitions, identify_n_prior_competitions, get_prior_gp_rounds)
from ratings.competition_results import fetch_participant_records
from ratings.data_loader import load_difficulty_cache, save_difficulty_cache

class CompetitionDifficulty(typing.NamedTuple):
    """Structure to represent the difficulty of a competition."""
    competition: CompetitionIdentifier
    outcome_weighted: float
    n_reference: int
    truncate_enabled: bool

def relative_difficulty_solver_weighted(
        competition: CompetitionIdentifier,
        anchors: list[CompetitionIdentifier],
        outcomes: pl.DataFrame) -> float:
    """Returns how hard a competition is comapred to comparison ("anchor") competitions.
    
    Higher numbers mean that competitors scored more points than in their anchors, implying that
    the competition was easier.

    This is evenly weighted across competitors.
    """

    competition_data = fetch_participant_records([competition], outcomes)
    anchor_data = fetch_participant_records(anchors, outcomes)

    means = []
    for matching_user in competition_data["user_pseudo_id"].unique():
        user_criteria = pl.col('user_pseudo_id') == matching_user
        competition_filtered = competition_data.filter(user_criteria)
        competition_perf = competition_filtered.get_column("points").to_list()
        if len(competition_perf) > 0:
            anchor_perf = anchor_data.filter(user_criteria).get_column("points").mean()
            means.append(competition_perf[0] / anchor_perf)

    mean_of_means = sum(means) / len(means)

    return mean_of_means

def relative_difficulty_outcome_weighted(
        competition: CompetitionIdentifier,
        anchors: list[CompetitionIdentifier],
        outcomes: pl.DataFrame) -> float:
    """Returns how hard a competition is compared to comparison ("anchor") competitions.

    Higher numbers mean that competitors scored more points than in their anchors, implying that
    the competition was easier.

    This is evenly weighted by outcome, meaning that solvers who participated in more events carry
    more weight.

    Only participants who have historical data in the anchor rounds are included in the
    calculation. This ensures the numerator and denominator use matching populations,
    avoiding bias from participants without anchor history.
    """
    competition_data = fetch_participant_records([competition], outcomes)

    # No data for this competition
    if len(competition_data) == 0:
        return np.nan

    anchor_data = fetch_participant_records(anchors, outcomes)

    # Collect current and historical performance only for participants with anchor history
    current_outcomes = []
    past_outcomes = []
    for matching_user in competition_data["user_pseudo_id"].unique():
        user_record_index = pl.col("user_pseudo_id") == matching_user
        prior_outcomes = anchor_data.filter(user_record_index).get_column("points").to_list()

        # Only include participants who have anchor history
        if prior_outcomes:
            current_score = competition_data.filter(user_record_index).get_column("points").to_list()
            if current_score:
                current_outcomes.append(current_score[0])
            for outcome in prior_outcomes:
                past_outcomes.append(outcome)

    # No historical data for any participants
    if len(past_outcomes) == 0 or len(current_outcomes) == 0:
        return np.nan

    competition_performance = sum(current_outcomes) / len(current_outcomes)
    mean_of_means = sum(past_outcomes) / len(past_outcomes)

    return competition_performance / mean_of_means

def difficulty_of_all_rounds(
        outcomes: pl.DataFrame,
        n_reference: int = 16,
        truncate: bool = True,
        use_gp_baseline: bool = False,
        use_cache: bool = True
        ) -> list[CompetitionDifficulty]:
    """Generate the difficulty for every round ever.

    Args:
        outcomes: Long-format DataFrame with competition results
        n_reference: Number of reference rounds for mixed baseline (ignored if use_gp_baseline=True)
        truncate: Whether to allow fewer than n_reference anchors
        use_gp_baseline: If True, use GP rounds as baseline for all competitions.
            This normalizes WSC scores to the GP scale, fixing systematic bias
            caused by different participant pools between GP and WSC.
        use_cache: If True and use_gp_baseline=True, use cached difficulty data if available
    """
    # Only cache the GP baseline case (the common/default case)
    if use_cache and use_gp_baseline and truncate:
        cached = load_difficulty_cache()
        if cached is not None:
            # Convert cached DataFrame back to list of CompetitionDifficulty
            results = []
            for row in cached.iter_rows(named=True):
                comp = CompetitionIdentifier(row['year'], row['round'], row['event_type'])
                results.append(CompetitionDifficulty(
                    comp, row['outcome_weighted'], row['n_reference'], row['truncate_enabled']
                ))
            return results

    results = difficulty_of_rounds(get_all_competitions(), outcomes, n_reference, truncate,
                                   use_gp_baseline)

    # Cache the results for GP baseline case
    if use_gp_baseline and truncate:
        # Convert to DataFrame for caching
        cache_data = {
            'year': [d.competition.year for d in results],
            'round': [d.competition.round for d in results],
            'event_type': [d.competition.event_type for d in results],
            'outcome_weighted': [d.outcome_weighted for d in results],
            'n_reference': [d.n_reference for d in results],
            'truncate_enabled': [d.truncate_enabled for d in results],
        }
        save_difficulty_cache(pl.DataFrame(cache_data))

    return results

def difficulty_of_rounds(
        competitions: list[CompetitionIdentifier],
        outcomes: pl.DataFrame,
        n_reference: int = 16,
        truncate: bool = True,
        use_gp_baseline: bool = False
        ) -> list[CompetitionDifficulty]:
    """Generate the difficulty for each of a number of rounds.

    Args:
        competitions: List of competitions to calculate difficulty for
        outcomes: Long-format DataFrame with competition results
        n_reference: Number of reference rounds for mixed baseline (ignored if use_gp_baseline=True)
        truncate: Whether to allow fewer than n_reference anchors
        use_gp_baseline: If True, use GP rounds as baseline for all competitions.
            This normalizes WSC scores to the GP scale, fixing systematic bias
            caused by different participant pools between GP and WSC.
    """
    # First year of GP data - can't calibrate WSC before this
    GP_START_YEAR = 2014

    results = []

    for competition in competitions:
        # For pre-GP era WSC, we can't reliably calibrate against GP data.
        # Using 2014 GP as retroactive anchor produces unreliable difficulty ratios
        # because score scales differ and participant overlap is limited.
        # Use difficulty 0.5 to provide a reasonable bridge between early WSC and GP era,
        # avoiding a large rating jump when GP data begins in 2014.
        if (use_gp_baseline and
            competition.event_type == "WSC" and
            competition.year < GP_START_YEAR):
            results.append(CompetitionDifficulty(competition, 0.5, n_reference, truncate))
            continue

        if use_gp_baseline:
            prior_events = get_prior_gp_rounds(competition)
        else:
            prior_events = identify_n_prior_competitions(competition, n_reference)

        # Might not have enough events
        if not use_gp_baseline and len(prior_events) < n_reference and not truncate:
            results.append(CompetitionDifficulty(competition, np.nan, n_reference, truncate))
        # If no prior events, set difficulty to 1.0
        elif len(prior_events) == 0:
            results.append(CompetitionDifficulty(competition, 1.0, n_reference, truncate))
        else:
            difficulty = relative_difficulty_outcome_weighted(competition, prior_events, outcomes)
            results.append(CompetitionDifficulty(competition, difficulty, n_reference, truncate))

    return results
