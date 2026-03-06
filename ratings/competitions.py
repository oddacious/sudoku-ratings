"""Structures for representing Sudoku competitions."""

import typing

MAXIMUM_GP_YEAR = 2026
MAX_ROUNDS = 16

class CompetitionIdentifier(typing.NamedTuple):
    """Structure to represent a specific competition."""
    year: int
    round: int
    event_type: str

def gp_rounds_by_year() -> dict[int, list[int]]:
    """Return the list of rounds (as integers) for each GP year."""
    gp_rounds = {
        2014: [1, 2, 3, 4, 5, 6, 7]
    }
    for i in range(2015, MAXIMUM_GP_YEAR + 1):
        gp_rounds[i] = range(1, 9)

    return gp_rounds

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
    """Construct an ordered list of all competitions."""
    gp_rounds = gp_rounds_by_year()
    wsc_rounds = wsc_rounds_by_year()

    years = get_all_years([gp_rounds, wsc_rounds])

    competitions = []

    for year in years:
        # Always record the GP as occurring first. This is typically, but not always, the case.
        if year in gp_rounds:
            for event_round in gp_rounds[year]:
                competitions.append(CompetitionIdentifier(year, event_round, "GP"))
        if year in wsc_rounds:
            for event_round in wsc_rounds[year]:
                competitions.append(CompetitionIdentifier(year, event_round, "WSC"))

    return competitions

def all_gp_round_names() -> list:
    """Return the column names of all GP rounds."""
    gp_rounds = []
    for gp_round in range(1, 9):
        gp_rounds.append(f"GP_t{gp_round} points")

    return gp_rounds

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
    For WSC rounds: returns all GP rounds up to and including the competition's year

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
            # For WSC, include all GP rounds up to and including this year
            prior_gp.append(CompetitionIdentifier(year, rnd, "GP"))

    # For pre-2014 WSC competitions, use 2014 GP as anchor (first available GP data)
    # This allows early WSC rounds to be calibrated against GP performance.
    # Don't apply this to GP rounds (2014 GP R1 should have no prior anchor).
    if len(prior_gp) == 0 and competition.event_type == "WSC" and 2014 in gp_rounds:
        for rnd in gp_rounds[2014]:
            prior_gp.append(CompetitionIdentifier(2014, rnd, "GP"))

    return prior_gp
