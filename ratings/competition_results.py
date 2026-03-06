"""Interact with dataframes of competition results."""

import polars as pl

from ratings.competitions import (
    CompetitionIdentifier, all_gp_round_names, all_wsc_round_names)

# GP scoring scale changed between 2015 and 2016 from 0-100 to 0-1000.
# This factor normalizes 2014-2015 scores to the 2016+ scale.
# Calculated from mean scores of participants who competed in both periods.
GP_2014_2015_SCALE_FACTOR = 8.5

def melt_by_columns(table: pl.DataFrame, event_type: str) -> pl.DataFrame:
    """Normalize a table to one row per solver-year-competition-round."""
    if event_type == "GP":
        list_of_rounds = all_gp_round_names()
    else:
        list_of_rounds = all_wsc_round_names()

    long_df = table.melt(
        id_vars=["user_pseudo_id", "year"],
        value_vars=list_of_rounds
    )

    long_df = long_df.rename({"variable": "round", "value": "points"})
    long_df = long_df.with_columns(
        pl.col("round").str.extract(r"(\d+) points$").cast(pl.Int64).alias("round")
    )
    long_df = long_df.filter(pl.col("points").is_not_null())
    if long_df["points"].dtype == "object":
        long_df = long_df.with_column(
            pl.col("points").str.replace(',', '').cast(pl.Float64).alias("points"))
    long_df = long_df.with_columns(pl.lit(event_type).alias("competition"))

    return long_df

def normalize_table_gp(table: pl.DataFrame) -> pl.DataFrame:
    """Normalize a GP table to one row per solver-year-competition-round."""
    return melt_by_columns(table, "GP")

def normalize_table_wsc(table: pl.DataFrame) -> pl.DataFrame:
    """Normalize a WSC table to one row per solver-year-competition-round."""
    return melt_by_columns(table, "WSC")

def normalize_gp_scoring_scale(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize GP scores from 2014-2015 to match 2016+ scale.

    GP changed its scoring system between 2015 and 2016 from a 0-100 scale
    to a 0-1000 scale. This multiplies 2014-2015 GP scores by the scale factor
    so they're comparable to later years.
    """
    return df.with_columns(
        pl.when((pl.col("competition") == "GP") & (pl.col("year").is_in([2014, 2015])))
        .then(pl.col("points") * GP_2014_2015_SCALE_FACTOR)
        .otherwise(pl.col("points"))
        .alias("points")
    )

def normalize_all_tables(gp: pl.DataFrame, wsc: pl.DataFrame) -> pl.DataFrame:
    """Normalize GP and WSC tables together to one row per solver-year-competition-round."""
    gp = normalize_table_gp(gp)
    wsc = normalize_table_wsc(wsc)

    combined = pl.concat([gp, wsc])

    # Apply GP scoring scale normalization for 2014-2015
    combined = normalize_gp_scoring_scale(combined)

    return combined

def fetch_participant_records(
        events: list[CompetitionIdentifier], result_table: pl.DataFrame) -> pl.DataFrame:
    """Return all applicable records for the chosen events."""

    all_records = None
    for competition in events:
        subset = result_table.filter(
            (pl.col("year") == competition.year) &
            (pl.col("round") == competition.round) &
            (pl.col("competition") == competition.event_type)
        )
        if all_records is None:
            all_records = subset
        else:
            all_records = pl.concat([all_records, subset])

    return all_records
