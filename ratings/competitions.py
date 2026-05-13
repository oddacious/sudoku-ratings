"""Structures for representing Sudoku competitions."""

import typing
from datetime import date
from typing import Optional

MAXIMUM_GP_YEAR = 2026
MAX_ROUNDS = 16

class CompetitionIdentifier(typing.NamedTuple):
    """Structure to represent a specific competition."""
    year: int
    round: int
    event_type: str

# Known dates for specific competitions. Sparse — only fill in where ordering matters.
# ESC and WSC are single multi-day events; use the event start date for all rounds.
# Add GP round dates as needed to resolve ambiguities flagged by find_ambiguous_orderings().
COMPETITION_DATES: dict[tuple[int, int, str], date] = {
    (2026, 1, "ESC"): date(2026, 5, 12),
    (2026, 2, "ESC"): date(2026, 5, 12),
    (2026, 3, "ESC"): date(2026, 5, 12),
    (2026, 4, "ESC"): date(2026, 5, 12),
    (2026, 5, "ESC"): date(2026, 5, 12),
    (2026, 6, "ESC"): date(2026, 5, 12),
    (2026, 7, "ESC"): date(2026, 5, 12),
}

def get_competition_date(c: CompetitionIdentifier) -> Optional[date]:
    """Return the known date for a competition, or None if unknown."""
    return COMPETITION_DATES.get((c.year, c.round, c.event_type))


# Typical month each event type falls in, used as a sentinel for undated rounds.
# This gives a reasonable cross-type ordering when dates are missing.
# GP spans the year; March places undated rounds before mid-year championships.
_SENTINEL_MONTH = {"GP": 3, "ESC": 5, "WSC": 10}


def _sort_key(c: CompetitionIdentifier) -> tuple[int, int]:
    """Sort key for ordering competitions chronologically within a year.

    Competitions with known dates sort by date. Undated competitions use a
    typical-month sentinel (GP≈March, ESC≈May, WSC≈October) so they cluster
    near when those events usually occur. The round number breaks ties within
    the same event type.
    """
    d = get_competition_date(c)
    if d is not None:
        return (d.toordinal(), c.round)
    month = _SENTINEL_MONTH.get(c.event_type, 6)
    return (date(c.year, month, 1).toordinal(), c.round)


def gp_rounds_by_year() -> dict[int, list[int]]:
    """Return the list of rounds (as integers) for each GP year."""
    gp_rounds = {
        2014: [1, 2, 3, 4, 5, 6, 7]
    }
    for i in range(2015, MAXIMUM_GP_YEAR + 1):
        gp_rounds[i] = range(1, 9)

    return gp_rounds

def esc_rounds_by_year() -> dict[int, list[int]]:
    """Return the list of rounds (as integers) for each ESC year."""
    return {
        2026: range(1, 8),
    }

def wsc_rounds_by_year():
    """Return the list of rounds (as integers) for each WSC year.

    Note that the WSC often has a gap in between rounds, with a first set of rounds from 1 to N and
    then another set from 10 or 11 onwards.
    """
    return {
        2025: [1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
        2024: [1, 2, 3, 4, 5, 6, 7, 10, 11],
        2023: range(1, 11),
        2022: [1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
        # No event in 2021 or 2020, although I may later on support the 2021 World Sudoku
        # "Competition"
        2019: [1, 2, 3, 4, 5, 6, 7, 11, 12, 13],
        2018: range(1, 11),
        2017: [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16],
        2016: [1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
        2015: [1, 2, 3, 4, 5, 6, 8, 9],
        2014: [1, 2, 3, 4, 5, 6, 9, 10],
        # No data currently for the 2013 WSC
        2012: range(1, 8),
        2011: range(1, 11),
        2010: range(1, 11), # These did not originally have numeric names
        # Missing data for 2006-2009 competitions.
    }

def get_all_years(list_of_year_maps: list[dict[int, list[int]]]) -> list[int]:
    """Return all years included in the provided competition year maps."""
    return sorted({year for competition in list_of_year_maps for year in competition.keys()})

def get_all_competitions() -> list[CompetitionIdentifier]:
    """Construct a chronologically ordered list of all competitions.

    Competitions with known dates in COMPETITION_DATES sort by date. Undated
    competitions fall back to stable event-type order (GP < ESC < WSC) within
    each year.
    """
    gp_rounds = gp_rounds_by_year()
    esc_rounds = esc_rounds_by_year()
    wsc_rounds = wsc_rounds_by_year()

    years = get_all_years([gp_rounds, esc_rounds, wsc_rounds])

    competitions = []

    for year in years:
        year_comps = []
        if year in gp_rounds:
            for event_round in gp_rounds[year]:
                year_comps.append(CompetitionIdentifier(year, event_round, "GP"))
        if year in esc_rounds:
            for event_round in esc_rounds[year]:
                year_comps.append(CompetitionIdentifier(year, event_round, "ESC"))
        if year in wsc_rounds:
            for event_round in wsc_rounds[year]:
                year_comps.append(CompetitionIdentifier(year, event_round, "WSC"))
        competitions.extend(sorted(year_comps, key=_sort_key))

    return competitions

def all_gp_round_names() -> list:
    """Return the column names of all GP rounds."""
    gp_rounds = []
    for gp_round in range(1, 9):
        gp_rounds.append(f"GP_t{gp_round} points")

    return gp_rounds

def all_esc_round_names() -> list:
    """Return the column names of all ESC rounds (up to MAX_ROUNDS)."""
    return [f"ESC_t{r} points" for r in range(1, MAX_ROUNDS + 1)]

def all_wsc_round_names() -> list:
    """Return the column names of all WSC rounds."""
    wsc_rounds = []
    for wsc_round in range(1, MAX_ROUNDS + 1):
        wsc_rounds.append(f"WSC_t{wsc_round} points")

    return wsc_rounds

def identify_n_prior_competitions(
        competition: CompetitionIdentifier, num_competition: int) -> list[CompetitionIdentifier]:
    """Given a competition, find the n competitions that immediately preceded it."""
    all_competitions = get_all_competitions()
    location_of_competition = all_competitions.index(competition)
    n_index_before = max(location_of_competition - num_competition, 0)

    return list(reversed(all_competitions[n_index_before:location_of_competition]))

def get_competition_index() -> dict[tuple[int, int, str], int]:
    """Get a mapping from competition tuple to chronological index.

    Returns:
        Dict mapping (year, round, event_type) -> chronological index
    """
    all_competitions = get_all_competitions()
    return {
        (c.year, c.round, c.event_type): i
        for i, c in enumerate(all_competitions)
    }


def get_prior_gp_rounds(competition: CompetitionIdentifier) -> list[CompetitionIdentifier]:
    """Get all GP rounds that occurred before (or during the same year as) a competition.

    For GP rounds: returns all prior GP rounds (same year earlier rounds + all previous years)
    For WSC and ESC rounds: returns all GP rounds up to and including the competition's year

    For pre-2014 competitions (before GP existed), returns 2014 GP rounds as the anchor.
    This allows early WSC rounds to be calibrated against the first available GP data.

    This is used for GP-baseline difficulty calculation, which uses GP performance
    as a stable reference point for measuring difficulty.
    """
    gp_rounds = gp_rounds_by_year()
    prior_gp = []

    for year in sorted(gp_rounds.keys()):
        if year > competition.year:
            break

        for rnd in gp_rounds[year]:
            # For GP competitions, exclude same round and later in same year
            if competition.event_type == "GP":
                if year == competition.year and rnd >= competition.round:
                    continue
            # For WSC and ESC, include all GP rounds up to and including this year
            prior_gp.append(CompetitionIdentifier(year, rnd, "GP"))

    # For pre-2014 WSC/ESC competitions, use 2014 GP as anchor (first available GP data)
    if len(prior_gp) == 0 and competition.event_type in ("WSC", "ESC") and 2014 in gp_rounds:
        for rnd in gp_rounds[2014]:
            prior_gp.append(CompetitionIdentifier(2014, rnd, "GP"))

    return prior_gp


def find_ambiguous_orderings() -> list[str]:
    """Return human-readable warnings for competitions with uncertain timeline position.

    A competition C is ambiguous when:
    - C has no known date, AND
    - In the same year, at least one competition from a different event type has a known date.

    In that case, C's position relative to the dated event is uncertain. Add dates for the
    relevant rounds to COMPETITION_DATES to resolve the warning.
    """
    gp_rounds = gp_rounds_by_year()
    esc_rounds = esc_rounds_by_year()
    wsc_rounds = wsc_rounds_by_year()

    years = get_all_years([gp_rounds, esc_rounds, wsc_rounds])
    warnings = []

    for year in years:
        year_comps: list[CompetitionIdentifier] = []
        for rnd in gp_rounds.get(year, []):
            year_comps.append(CompetitionIdentifier(year, rnd, "GP"))
        for rnd in esc_rounds.get(year, []):
            year_comps.append(CompetitionIdentifier(year, rnd, "ESC"))
        for rnd in wsc_rounds.get(year, []):
            year_comps.append(CompetitionIdentifier(year, rnd, "WSC"))

        dated_types = {c.event_type for c in year_comps if get_competition_date(c) is not None}

        for event_type in sorted(
            {c.event_type for c in year_comps if get_competition_date(c) is None}
        ):
            other_dated = dated_types - {event_type}
            if not other_dated:
                continue
            rounds_without_dates = sorted(
                c.round for c in year_comps
                if c.event_type == event_type and get_competition_date(c) is None
            )
            dated_summary = ", ".join(sorted(other_dated))
            warnings.append(
                f"{year} {event_type} R{rounds_without_dates[0]}–R{rounds_without_dates[-1]} "
                f"has no dates but shares the year with dated {dated_summary}. "
                f"Add {event_type} {year} round dates to COMPETITION_DATES to resolve."
            )

    return warnings
