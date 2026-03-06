"""Bridge between loaded data and backtesting framework.

Converts the wide-format solver-year DataFrame into the chronological
round-by-round format needed for backtesting.
"""

import polars as pl

from ratings.competitions import get_all_competitions, CompetitionIdentifier


def extract_rounds_for_backtest(
    df: pl.DataFrame
) -> list[tuple[CompetitionIdentifier, list[tuple[str, float]]]]:
    """Convert loaded DataFrame to chronological round results.

    Args:
        df: DataFrame from load_all_data() with columns like 'GP_t1 points',
            'WSC_t1 points', 'user_pseudo_id', 'year'

    Returns:
        List of (competition_id, results) tuples in chronological order,
        where results is list of (player_id, points) tuples
    """
    all_competitions = get_all_competitions()
    rounds_data = []

    for comp in all_competitions:
        # Determine column name for this round's points
        if comp.event_type == "GP":
            points_col = f"GP_t{comp.round} points"
        else:
            points_col = f"WSC_t{comp.round} points"

        # Skip if column doesn't exist
        if points_col not in df.columns:
            continue

        # Filter to this year and get non-null results
        year_df = df.filter(pl.col("year") == comp.year)
        round_df = year_df.filter(pl.col(points_col).is_not_null())

        if len(round_df) == 0:
            continue

        # Extract (player_id, points) tuples
        results = []
        for row in round_df.iter_rows(named=True):
            player_id = row["user_pseudo_id"]
            points = row[points_col]
            if player_id is not None and points is not None:
                results.append((player_id, float(points)))

        if results:
            # Sort by points descending (winner first)
            results.sort(key=lambda x: x[1], reverse=True)
            rounds_data.append((comp, results))

    return rounds_data


def get_rounds_only(
    df: pl.DataFrame
) -> list[list[tuple[str, float]]]:
    """Get just the round results without competition identifiers.

    Convenience function for backtest_glicko which only needs the results.

    Args:
        df: DataFrame from load_all_data()

    Returns:
        List of rounds, each round is list of (player_id, points) tuples
    """
    rounds_with_ids = extract_rounds_for_backtest(df)
    return [results for _, results in rounds_with_ids]


def summarize_rounds(df: pl.DataFrame) -> None:
    """Print summary of rounds extracted from data."""
    rounds_data = extract_rounds_for_backtest(df)

    print(f"Total rounds: {len(rounds_data)}")

    gp_rounds = [r for r in rounds_data if r[0].event_type == "GP"]
    wsc_rounds = [r for r in rounds_data if r[0].event_type == "WSC"]

    print(f"  GP rounds: {len(gp_rounds)}")
    print(f"  WSC rounds: {len(wsc_rounds)}")

    if rounds_data:
        first = rounds_data[0][0]
        last = rounds_data[-1][0]
        print(f"Date range: {first.year} {first.event_type} R{first.round} to {last.year} {last.event_type} R{last.round}")

    # Participant counts
    participant_counts = [len(results) for _, results in rounds_data]
    print(f"Participants per round: min={min(participant_counts)}, max={max(participant_counts)}, avg={sum(participant_counts)/len(participant_counts):.0f}")
