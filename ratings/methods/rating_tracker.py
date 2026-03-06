"""Shared rating computation logic for consistent results across commands.

This module provides a RatingTracker class that maintains solver histories
and computes ratings incrementally as competitions are processed. Both the
progression and solver commands use this to ensure consistent ranking logic.
"""

from dataclasses import dataclass
from typing import Optional

import polars as pl

from ratings.competitions import CompetitionIdentifier, get_all_competitions, get_competition_index
from ratings.leaderboard import INACTIVITY_THRESHOLD_YEARS
from ratings.methods.utils import compute_exp_weighted_mean


@dataclass
class SolverRating:
    """Rating information for a solver at a point in time."""
    solver_id: str
    rating: float
    n_rounds: int
    last_year: int


class RatingTracker:
    """Tracks solver ratings incrementally across competitions.

    This class maintains the state needed to compute ratings as we
    iterate through competitions in chronological order. It provides
    consistent rating computation for both progression and solver commands.

    Usage:
        tracker = RatingTracker(adjusted_data, min_history=3, decay_rate=0.90, prior_k=3)

        for comp in competitions:
            tracker.advance_to(comp)
            leaderboard = tracker.get_leaderboard(top_n=10, as_of_year=comp.year)
    """

    def __init__(
        self,
        adjusted_data: pl.DataFrame,
        min_history: int = 3,
        decay_rate: float = 0.90,
        prior_k: int = 0
    ):
        """Initialize the tracker.

        Args:
            adjusted_data: DataFrame with adjusted_points column
            min_history: Minimum rounds before a solver gets a rating
            decay_rate: Exponential decay rate for weighting
            prior_k: Number of pseudo-observations at prior mean (0 = no prior)
        """
        self.adjusted = adjusted_data
        self.min_history = min_history
        self.decay_rate = decay_rate
        self.prior_k = prior_k

        # Build competition index for chronological ordering
        self.all_competitions = get_all_competitions()
        self.comp_to_idx = get_competition_index()

        # State tracking
        self.histories: dict[str, list[float]] = {}
        self.last_years: dict[str, int] = {}
        self.prior_sum = 0.0
        self.prior_count = 0
        self.current_comp_idx = -1

    def _get_comp_idx(self, comp: CompetitionIdentifier) -> int:
        """Get the chronological index of a competition."""
        return self.comp_to_idx.get((comp.year, comp.round, comp.event_type), -1)

    def advance_to(self, competition: CompetitionIdentifier) -> None:
        """Advance state to include this competition and all prior ones.

        This processes all competitions from the current position up to
        and including the target competition.

        Args:
            competition: The competition to advance to
        """
        target_idx = self._get_comp_idx(competition)
        if target_idx < 0:
            return

        # Process all competitions from current position to target
        for idx in range(self.current_comp_idx + 1, target_idx + 1):
            comp = self.all_competitions[idx]
            self._process_competition(comp)

        self.current_comp_idx = target_idx

    def _process_competition(self, comp: CompetitionIdentifier) -> None:
        """Process a single competition, updating all solver histories."""
        round_data = self.adjusted.filter(
            (pl.col("year") == comp.year) &
            (pl.col("round") == comp.round) &
            (pl.col("competition") == comp.event_type)
        )

        for row in round_data.iter_rows(named=True):
            solver_id = row["user_pseudo_id"]
            adj_points = row["adjusted_points"]

            if adj_points is None:
                continue

            # Update solver's history
            if solver_id not in self.histories:
                self.histories[solver_id] = []
            self.histories[solver_id].append(adj_points)
            self.last_years[solver_id] = comp.year

            # Update prior statistics
            self.prior_sum += adj_points
            self.prior_count += 1

    def get_prior_mean(self) -> float:
        """Get the current prior mean (average of all points seen so far)."""
        return self.prior_sum / self.prior_count if self.prior_count > 0 else 500.0

    def compute_rating(self, history: list[float]) -> float:
        """Compute rating from a solver's history.

        Args:
            history: List of adjusted points (oldest first)

        Returns:
            The solver's rating
        """
        if self.prior_k > 0:
            prior_mean = self.get_prior_mean()
            regularized = [prior_mean] * self.prior_k + history
        else:
            regularized = history
        return compute_exp_weighted_mean(regularized, self.decay_rate)

    def get_solver_rating(self, solver_id: str) -> Optional[SolverRating]:
        """Get the current rating for a specific solver.

        Args:
            solver_id: The solver's ID

        Returns:
            SolverRating if solver has enough history, None otherwise
        """
        history = self.histories.get(solver_id)
        if history is None or len(history) < self.min_history:
            return None

        return SolverRating(
            solver_id=solver_id,
            rating=self.compute_rating(history),
            n_rounds=len(history),
            last_year=self.last_years[solver_id]
        )

    def get_all_ratings(self, as_of_year: int) -> list[SolverRating]:
        """Get ratings for all active solvers with enough history.

        Args:
            as_of_year: The reference year for activity filtering

        Returns:
            List of SolverRating for all qualifying solvers
        """
        min_active_year = as_of_year - INACTIVITY_THRESHOLD_YEARS

        ratings = []
        for solver_id, history in self.histories.items():
            if len(history) >= self.min_history and self.last_years[solver_id] >= min_active_year:
                ratings.append(SolverRating(
                    solver_id=solver_id,
                    rating=self.compute_rating(history),
                    n_rounds=len(history),
                    last_year=self.last_years[solver_id]
                ))

        return ratings

    def get_leaderboard(self, top_n: int, as_of_year: int) -> list[SolverRating]:
        """Get the top N solvers by rating.

        Args:
            top_n: Number of top solvers to return
            as_of_year: The reference year for activity filtering

        Returns:
            List of SolverRating sorted by rating descending
        """
        ratings = self.get_all_ratings(as_of_year)
        ratings.sort(key=lambda x: x.rating, reverse=True)
        return ratings[:top_n]

    def get_solver_rank(self, solver_id: str, as_of_year: int) -> Optional[tuple[int, int]]:
        """Get a solver's rank among all active solvers.

        Args:
            solver_id: The solver's ID
            as_of_year: The reference year for activity filtering

        Returns:
            Tuple of (rank, total_ranked) or None if solver not ranked
        """
        ratings = self.get_all_ratings(as_of_year)
        ratings.sort(key=lambda x: x.rating, reverse=True)

        for i, r in enumerate(ratings):
            if r.solver_id == solver_id:
                return (i + 1, len(ratings))

        return None

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.histories.clear()
        self.last_years.clear()
        self.prior_sum = 0.0
        self.prior_count = 0
        self.current_comp_idx = -1
