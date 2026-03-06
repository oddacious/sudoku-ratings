"""Load competition data by reusing loaders from sudokudos-github."""

import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import polars as pl

# Add the sibling project to the path so we can import its loaders
_SUDOKUDOS_PATH = Path(__file__).parent.parent.parent / "sudokudos-github"
if str(_SUDOKUDOS_PATH) not in sys.path:
    sys.path.append(str(_SUDOKUDOS_PATH))

# pylint: disable=wrong-import-position
# Loading the uncached versions, to avoid importing streamlit
from shared.data.loaders.gp import load_gp  # noqa: E402
from shared.data.loaders.wsc import load_wsc  # noqa: E402
from shared.data.manipulation import attempted_mapping, merge_unflat_datasets  # noqa: E402

from ratings.competition_results import normalize_all_tables  # noqa: E402

# Cache directory and files
_CACHE_DIR = Path(__file__).parent.parent / ".cache"
_NORMALIZED_CACHE = _CACHE_DIR / "normalized_data.parquet"
_MERGED_CACHE = _CACHE_DIR / "merged_data.parquet"
_DIFFICULTY_CACHE = _CACHE_DIR / "difficulty_gp_baseline.parquet"


class CacheInfo(NamedTuple):
    """Information about the data cache."""
    exists: bool
    normalized_path: Path
    merged_path: Path
    difficulty_path: Path
    normalized_modified: Optional[datetime]
    merged_modified: Optional[datetime]
    difficulty_modified: Optional[datetime]
    normalized_size: int
    merged_size: int
    difficulty_size: int


def get_cache_info() -> CacheInfo:
    """Get information about the current cache state."""
    norm_exists = _NORMALIZED_CACHE.exists()
    merged_exists = _MERGED_CACHE.exists()
    diff_exists = _DIFFICULTY_CACHE.exists()

    return CacheInfo(
        exists=norm_exists or merged_exists or diff_exists,
        normalized_path=_NORMALIZED_CACHE,
        merged_path=_MERGED_CACHE,
        difficulty_path=_DIFFICULTY_CACHE,
        normalized_modified=datetime.fromtimestamp(_NORMALIZED_CACHE.stat().st_mtime)
            if norm_exists else None,
        merged_modified=datetime.fromtimestamp(_MERGED_CACHE.stat().st_mtime)
            if merged_exists else None,
        difficulty_modified=datetime.fromtimestamp(_DIFFICULTY_CACHE.stat().st_mtime)
            if diff_exists else None,
        normalized_size=_NORMALIZED_CACHE.stat().st_size if norm_exists else 0,
        merged_size=_MERGED_CACHE.stat().st_size if merged_exists else 0,
        difficulty_size=_DIFFICULTY_CACHE.stat().st_size if diff_exists else 0,
    )


def purge_cache() -> bool:
    """Delete all cached data files. Returns True if any files were deleted."""
    deleted = False
    for cache_file in [_NORMALIZED_CACHE, _MERGED_CACHE, _DIFFICULTY_CACHE]:
        if cache_file.exists():
            cache_file.unlink()
            deleted = True
    if _CACHE_DIR.exists() and not any(_CACHE_DIR.iterdir()):
        _CACHE_DIR.rmdir()
    return deleted


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(exist_ok=True)


def get_difficulty_cache_path() -> Path:
    """Get the path to the difficulty cache file."""
    return _DIFFICULTY_CACHE


def save_difficulty_cache(df: pl.DataFrame) -> None:
    """Save difficulty data to cache."""
    _ensure_cache_dir()
    df.write_parquet(_DIFFICULTY_CACHE)


def load_difficulty_cache() -> Optional[pl.DataFrame]:
    """Load difficulty data from cache if it exists."""
    if _DIFFICULTY_CACHE.exists():
        return pl.read_parquet(_DIFFICULTY_CACHE)
    return None


def load_gp_wsc_separate(
    gp_directory: str = "data/processed/gp",
    wsc_directory: str = "data/raw/wsc/"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load GP and WSC data as separate DataFrames.

    Returns DataFrames with user_pseudo_id mapped (WSC names matched to GP IDs).

    Args:
        gp_directory: Path to GP CSV files
        wsc_directory: Path to WSC CSV files

    Returns:
        Tuple of (gp_dataframe, wsc_dataframe) with user_pseudo_id column
    """
    if not Path(gp_directory).is_absolute():
        gp_directory = str(_SUDOKUDOS_PATH / gp_directory)
    if not Path(wsc_directory).is_absolute():
        wsc_directory = str(_SUDOKUDOS_PATH / wsc_directory)

    gp_df = load_gp(csv_directory=gp_directory)
    wsc_df = load_wsc(csv_directory=wsc_directory)

    # Map WSC names to GP user_pseudo_id
    wsc_mapped = attempted_mapping(wsc_df, gp_df)

    return gp_df, wsc_mapped


def load_normalized_data(
    gp_directory: str = "data/processed/gp",
    wsc_directory: str = "data/raw/wsc/",
    use_cache: bool = True
) -> pl.DataFrame:
    """Load and normalize data to long format (one row per solver-round).

    Args:
        gp_directory: Path to GP CSV files
        wsc_directory: Path to WSC CSV files
        use_cache: If True, use cached data if available

    Returns:
        Long-format DataFrame with columns:
        [user_pseudo_id, year, round, points, competition]
    """
    if use_cache and _NORMALIZED_CACHE.exists():
        return pl.read_parquet(_NORMALIZED_CACHE)

    gp_df, wsc_df = load_gp_wsc_separate(gp_directory, wsc_directory)
    result = normalize_all_tables(gp_df, wsc_df)

    # Cache the result
    _ensure_cache_dir()
    result.write_parquet(_NORMALIZED_CACHE)

    return result


def load_all_data(
    gp_directory: str = "data/processed/gp",
    wsc_directory: str = "data/raw/wsc/",
    use_cache: bool = True
) -> pl.DataFrame:
    """Load and merge all GP and WSC competition data.

    Uses the loaders from sudokudos-github which handle:
    - Year-specific CSV format differences
    - Name/nick normalization and adjustments
    - User identifier creation (user_pseudo_id)
    - Mapping WSC names to GP identifiers

    Args:
        gp_directory: Path to GP CSV files (relative to sudokudos-github or absolute)
        wsc_directory: Path to WSC CSV files (relative to sudokudos-github or absolute)
        use_cache: If True, use cached data if available

    Returns:
        Combined Polars DataFrame with one row per solver-year, containing
        both GP and WSC results with unified user_pseudo_id.
    """
    if use_cache and _MERGED_CACHE.exists():
        return pl.read_parquet(_MERGED_CACHE)

    # If paths are relative, make them relative to sudokudos-github
    if not Path(gp_directory).is_absolute():
        gp_directory = str(_SUDOKUDOS_PATH / gp_directory)
    if not Path(wsc_directory).is_absolute():
        wsc_directory = str(_SUDOKUDOS_PATH / wsc_directory)

    gp_df = load_gp(csv_directory=gp_directory)
    wsc_df = load_wsc(csv_directory=wsc_directory)

    # Map WSC names to GP user_pseudo_id where possible
    wsc_mapped = attempted_mapping(wsc_df, gp_df)

    # Merge into a single solver-year level dataset
    combined = merge_unflat_datasets(gp_df, wsc_mapped)

    # Cache the result
    _ensure_cache_dir()
    combined.write_parquet(_MERGED_CACHE)

    return combined
