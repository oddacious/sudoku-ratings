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

# Known END dates for specific competitions. Sparse — only fill in where ordering matters.
# All dates are the END date of the event (last day of competition), not the start date.
# For multi-round events like GP, each round has its own end date.
# For multi-day events like ESC and WSC, all rounds share the event's final day.
# Add GP round dates as needed to resolve ambiguities flagged by find_ambiguous_orderings().
COMPETITION_DATES: dict[tuple[int, int, str], date] = {
    # ESC 2026: all rounds held at the same event, ending 2026-05-12
    (2026, 1, "ESC"): date(2026, 5, 12),
    (2026, 2, "ESC"): date(2026, 5, 12),
    (2026, 3, "ESC"): date(2026, 5, 12),
    (2026, 4, "ESC"): date(2026, 5, 12),
    (2026, 5, "ESC"): date(2026, 5, 12),
    (2026, 6, "ESC"): date(2026, 5, 12),
    (2026, 7, "ESC"): date(2026, 5, 12),
    # GP 2026: rounds 4 and 5 added to bracket ESC 2026 (May 12)
    (2026, 4, "GP"): date(2026, 4, 15),
    (2026, 5, "GP"): date(2026, 5, 20),
}

def get_competition_date(c: CompetitionIdentifier) -> Optional[date]:
    """Return the known end date for a competition, or None if unknown."""
    return COMPETITION_DATES.get((c.year, c.round, c.event_type))


# Default interval between rounds when extrapolating beyond known anchor dates.
_DEFAULT_ROUND_INTERVAL_DAYS = 35

# Typical month each event type falls in, used when NO dates at all are known
# for that event type in a year.
_SENTINEL_MONTH = {"GP": 3, "ESC": 5, "WSC": 10}


def _interpolate_dates(
    year_comps: list[CompetitionIdentifier],
) -> dict[CompetitionIdentifier, int]:
    """Compute ordinal sort keys for all competitions in a single year.

    Competitions with known end dates use those dates directly. Undated rounds
    within the same event type are assigned virtual dates by interpolating
    (or extrapolating) from the known anchor dates in that event type, using
    round number as the ordering constraint.

    For event types with no known dates at all, a typical-month sentinel is
    used so they cluster near when those events usually occur.
    """
    result = {}

    # Group by event type
    by_type: dict[str, list[CompetitionIdentifier]] = {}
    for c in year_comps:
        by_type.setdefault(c.event_type, []).append(c)

    for event_type, comps in by_type.items():
        comps = sorted(comps, key=lambda c: c.round)

        # Collect known (round, ordinal) anchors for this event type
        anchors = {
            c.round: get_competition_date(c).toordinal()
            for c in comps
            if get_competition_date(c) is not None
        }

        if not anchors:
            # No dates known for this event type: use typical-month sentinel
            month = _SENTINEL_MONTH.get(event_type, 6)
            sentinel = date(comps[0].year, month, 1).toordinal()
            for c in comps:
                result[c] = sentinel
            continue

        # Estimate the per-round interval from consecutive known anchors
        sorted_anchor_rounds = sorted(anchors)
        if len(sorted_anchor_rounds) >= 2:
            intervals = [
                (anchors[sorted_anchor_rounds[i + 1]] - anchors[sorted_anchor_rounds[i]])
                / (sorted_anchor_rounds[i + 1] - sorted_anchor_rounds[i])
                for i in range(len(sorted_anchor_rounds) - 1)
            ]
            interval = sum(intervals) / len(intervals)
        else:
            interval = _DEFAULT_ROUND_INTERVAL_DAYS

        # Assign ordinals to all rounds by interpolating/extrapolating from anchors
        ref_round = sorted_anchor_rounds[0]
        ref_ordinal = anchors[ref_round]
        for c in comps:
            if c.round in anchors:
                result[c] = anchors[c.round]
            else:
                result[c] = round(ref_ordinal + (c.round - ref_round) * interval)

    return result


def _sort_key_for_year(
    year_comps: list[CompetitionIdentifier],
) -> list[CompetitionIdentifier]:
    """Return year_comps sorted chronologically using date interpolation."""
    ordinals = _interpolate_dates(year_comps)
    return sorted(year_comps, key=lambda c: (ordinals[c], c.round))


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

    Within each year, competitions are ordered using known end dates from
    COMPETITION_DATES. Undated rounds within the same event type are assigned
    virtual dates by interpolating from the nearest known anchor dates in that
    type (with round number determining direction). Event types with no known
    dates at all fall back to a typical-month sentinel (GP≈March, ESC≈May,
    WSC≈October).
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
        competitions.extend(_sort_key_for_year(year_comps))

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

    A competition C is considered ambiguous when its position relative to a dated
    event from a different event type cannot be resolved even through interpolation.
    Specifically, C is ambiguous when:
    - Its event type has NO known end dates at all in that year (so interpolation
      cannot anchor its position), AND
    - In the same year, at least one competition from a different event type has a
      known date.

    When at least one round of an event type is dated, the others can be ordered
    by interpolation from those anchors, so no warning is issued.
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

        # Types with at least one known date can interpolate; types with none cannot.
        anchored_types = {
            c.event_type for c in year_comps if get_competition_date(c) is not None
        }
        unanchored_types = {
            c.event_type for c in year_comps if c.event_type not in anchored_types
        }

        for event_type in sorted(unanchored_types):
            other_anchored = anchored_types - {event_type}
            if not other_anchored:
                continue
            rounds = sorted(c.round for c in year_comps if c.event_type == event_type)
            dated_summary = ", ".join(sorted(other_anchored))
            warnings.append(
                f"{year} {event_type} R{rounds[0]}–R{rounds[-1]} has no dates "
                f"(cannot interpolate position) but shares the year with dated {dated_summary}. "
                f"Add at least one {event_type} {year} round date to COMPETITION_DATES to resolve."
            )

    return warnings
