"""Glicko-1 rating system implementation.

Based on Mark Glickman's original Glicko system:
http://www.glicko.net/glicko/glicko.pdf

WARNING: Standard Glicko produces unstable ratings for mass-start events.
When treating a round with 500+ participants as simultaneous pairwise "games",
rating swings of 50,000+ points can occur. See CLAUDE.md for details.

This implementation achieves 81.2% pairwise accuracy but produces
non-interpretable rating values. Consider using points_based or
percentile_based methods instead.
"""

import math
from dataclasses import dataclass

# Glicko constants
Q = math.log(10) / 400  # ~0.00575646
PI_SQUARED = math.pi ** 2

# Default parameters (tunable)
DEFAULT_RATING = 1500
DEFAULT_RD = 350
MIN_RD = 30
MAX_RD = 350


@dataclass
class GlickoRating:
    """A player's Glicko rating with rating deviation."""
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD

    def copy(self) -> "GlickoRating":
        """Return a copy of this rating."""
        return GlickoRating(rating=self.rating, rd=self.rd)


def g(rd: float) -> float:
    """The g function - reduces impact of opponents with high RD."""
    return 1 / math.sqrt(1 + 3 * Q**2 * rd**2 / PI_SQUARED)


def expected_score(player: GlickoRating, opponent: GlickoRating) -> float:
    """Calculate expected score (probability of winning) for player against opponent."""
    g_rd = g(opponent.rd)
    exponent = -g_rd * (player.rating - opponent.rating) / 400
    return 1 / (1 + 10**exponent)


def update_rating(
    player: GlickoRating,
    opponents: list[GlickoRating],
    scores: list[float]
) -> GlickoRating:
    """Update a player's rating based on results against multiple opponents.

    Args:
        player: Current rating of the player being updated
        opponents: List of opponent ratings
        scores: List of scores (1.0 = win, 0.5 = draw, 0.0 = loss) against each opponent

    Returns:
        New GlickoRating with updated rating and RD
    """
    if not opponents:
        return player.copy()

    if len(opponents) != len(scores):
        raise ValueError("opponents and scores must have same length")

    # Calculate d^2 (the sum of variance contributions from each game)
    d_squared_inv = 0.0
    for opp in opponents:
        g_rd = g(opp.rd)
        e = expected_score(player, opp)
        d_squared_inv += g_rd**2 * e * (1 - e)
    d_squared_inv *= Q**2

    # Avoid division by zero
    if d_squared_inv == 0:
        return player.copy()

    d_squared = 1 / d_squared_inv

    # Calculate rating change
    rating_change_sum = 0.0
    for opp, score in zip(opponents, scores):
        g_rd = g(opp.rd)
        e = expected_score(player, opp)
        rating_change_sum += g_rd * (score - e)

    rd_squared = player.rd ** 2
    new_rating = player.rating + Q / (1/rd_squared + 1/d_squared) * rating_change_sum

    # Calculate new RD
    new_rd = math.sqrt(1 / (1/rd_squared + 1/d_squared))
    new_rd = max(MIN_RD, min(MAX_RD, new_rd))

    return GlickoRating(rating=new_rating, rd=new_rd)


def decay_rd(rating: GlickoRating, periods_inactive: int, c: float = 34.6) -> GlickoRating:
    """Increase RD for a player who hasn't competed recently.

    The RD grows over time to reflect increasing uncertainty about
    a player's current skill level.

    Args:
        rating: Current rating
        periods_inactive: Number of rating periods since last competition
        c: RD growth constant per period (tunable parameter)

    Returns:
        New GlickoRating with increased RD (rating unchanged)
    """
    if periods_inactive <= 0:
        return rating.copy()

    new_rd_squared = rating.rd**2 + c**2 * periods_inactive
    new_rd = min(math.sqrt(new_rd_squared), MAX_RD)

    return GlickoRating(rating=rating.rating, rd=new_rd)


def process_round(
    results: list[tuple[str, float]],
    ratings: dict[str, GlickoRating],
    c: float = 34.6
) -> dict[str, GlickoRating]:
    """Process a single competition round and update all ratings.

    Args:
        results: List of (player_id, points) tuples, ordered by finish
        ratings: Current ratings for all known players
        c: RD decay constant for non-participants

    Returns:
        Updated ratings dictionary
    """
    # Track who participated this round
    participants = {player_id for player_id, _ in results}

    # Initialize new players
    for player_id, _ in results:
        if player_id not in ratings:
            ratings[player_id] = GlickoRating()

    # Build pairwise results for each participant
    # For each player, collect all opponents and scores
    new_ratings = {}

    for i, (player_id, player_points) in enumerate(results):
        player = ratings[player_id]
        opponents = []
        scores = []

        for j, (opp_id, opp_points) in enumerate(results):
            if i == j:
                continue

            opponents.append(ratings[opp_id])

            if player_points > opp_points:
                scores.append(1.0)
            elif player_points < opp_points:
                scores.append(0.0)
            else:
                scores.append(0.5)

        new_ratings[player_id] = update_rating(player, opponents, scores)

    # Decay RD for non-participants
    for player_id, rating in ratings.items():
        if player_id not in participants:
            new_ratings[player_id] = decay_rd(rating, periods_inactive=1, c=c)

    return new_ratings
