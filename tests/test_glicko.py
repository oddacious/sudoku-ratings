"""Tests for the Glicko rating system."""

import unittest
import math

from ratings.methods.glicko import (
    GlickoRating,
    g,
    expected_score,
    update_rating,
    decay_rd,
    process_round,
    DEFAULT_RATING,
    DEFAULT_RD,
    MIN_RD,
    MAX_RD,
)


class TestGFunction(unittest.TestCase):
    """Test the g(RD) function."""

    def test_low_rd_near_one(self):
        """g(RD) should approach 1 as RD approaches 0."""
        self.assertAlmostEqual(g(0), 1.0, places=5)
        self.assertGreater(g(30), 0.95)

    def test_high_rd_lower(self):
        """g(RD) should be lower for higher RD."""
        self.assertLess(g(350), g(100))
        self.assertLess(g(100), g(30))

    def test_default_rd(self):
        """g at default RD should be reasonable."""
        g_default = g(DEFAULT_RD)
        self.assertGreater(g_default, 0.5)
        self.assertLess(g_default, 1.0)


class TestExpectedScore(unittest.TestCase):
    """Test expected score calculation."""

    def test_equal_ratings(self):
        """Equal ratings should give 0.5 expected score."""
        a = GlickoRating(1500, 100)
        b = GlickoRating(1500, 100)
        self.assertAlmostEqual(expected_score(a, b), 0.5, places=5)

    def test_higher_rating_favored(self):
        """Higher rated player should have expected score > 0.5."""
        a = GlickoRating(1600, 100)
        b = GlickoRating(1400, 100)
        self.assertGreater(expected_score(a, b), 0.5)
        self.assertLess(expected_score(b, a), 0.5)

    def test_expected_scores_complement_equal_rd(self):
        """Expected scores should sum to 1 when RDs are equal."""
        a = GlickoRating(1600, 100)
        b = GlickoRating(1400, 100)
        self.assertAlmostEqual(
            expected_score(a, b) + expected_score(b, a), 1.0, places=5
        )

    def test_large_rating_difference(self):
        """Large rating difference should give near-certain outcome."""
        a = GlickoRating(2000, 100)
        b = GlickoRating(1000, 100)
        self.assertGreater(expected_score(a, b), 0.99)


class TestUpdateRating(unittest.TestCase):
    """Test rating updates after games."""

    def test_no_opponents(self):
        """No games should leave rating unchanged."""
        player = GlickoRating(1500, 200)
        updated = update_rating(player, [], [])
        self.assertEqual(updated.rating, player.rating)
        self.assertEqual(updated.rd, player.rd)

    def test_win_increases_rating(self):
        """Winning against equal opponent should increase rating."""
        player = GlickoRating(1500, 200)
        opponent = GlickoRating(1500, 200)
        updated = update_rating(player, [opponent], [1.0])
        self.assertGreater(updated.rating, player.rating)

    def test_loss_decreases_rating(self):
        """Losing against equal opponent should decrease rating."""
        player = GlickoRating(1500, 200)
        opponent = GlickoRating(1500, 200)
        updated = update_rating(player, [opponent], [0.0])
        self.assertLess(updated.rating, player.rating)

    def test_draw_no_change_equal_ratings(self):
        """Draw against equal opponent should barely change rating."""
        player = GlickoRating(1500, 200)
        opponent = GlickoRating(1500, 200)
        updated = update_rating(player, [opponent], [0.5])
        self.assertAlmostEqual(updated.rating, player.rating, places=1)

    def test_rd_decreases_after_games(self):
        """RD should decrease after playing games."""
        player = GlickoRating(1500, 200)
        opponent = GlickoRating(1500, 100)
        updated = update_rating(player, [opponent], [1.0])
        self.assertLess(updated.rd, player.rd)

    def test_multiple_opponents(self):
        """Should handle multiple opponents correctly."""
        player = GlickoRating(1500, 200)
        opponents = [
            GlickoRating(1400, 100),
            GlickoRating(1500, 100),
            GlickoRating(1600, 100),
        ]
        # Win against lower, draw with equal, lose to higher
        scores = [1.0, 0.5, 0.0]
        updated = update_rating(player, opponents, scores)
        # Should be close to original since results match expectations
        self.assertAlmostEqual(updated.rating, player.rating, delta=50)


class TestDecayRD(unittest.TestCase):
    """Test RD decay for inactive players."""

    def test_no_decay_zero_periods(self):
        """Zero inactive periods should not change RD."""
        rating = GlickoRating(1500, 100)
        decayed = decay_rd(rating, 0)
        self.assertEqual(decayed.rd, rating.rd)

    def test_rd_increases_with_inactivity(self):
        """RD should increase with inactive periods."""
        rating = GlickoRating(1500, 100)
        decayed = decay_rd(rating, 5)
        self.assertGreater(decayed.rd, rating.rd)

    def test_rd_capped_at_max(self):
        """RD should not exceed MAX_RD."""
        rating = GlickoRating(1500, 300)
        decayed = decay_rd(rating, 100)
        self.assertLessEqual(decayed.rd, MAX_RD)

    def test_rating_unchanged(self):
        """Rating itself should not change from decay."""
        rating = GlickoRating(1600, 100)
        decayed = decay_rd(rating, 5)
        self.assertEqual(decayed.rating, rating.rating)


class TestProcessRound(unittest.TestCase):
    """Test processing a full competition round."""

    def test_new_players_initialized(self):
        """New players should get default ratings."""
        results = [("alice", 100), ("bob", 90)]
        ratings = process_round(results, {})
        self.assertIn("alice", ratings)
        self.assertIn("bob", ratings)

    def test_winner_rating_increases(self):
        """Round winner's rating should increase."""
        ratings = {
            "alice": GlickoRating(1500, 100),
            "bob": GlickoRating(1500, 100),
        }
        results = [("alice", 100), ("bob", 90)]  # alice wins
        new_ratings = process_round(results, ratings.copy())
        self.assertGreater(new_ratings["alice"].rating, 1500)
        self.assertLess(new_ratings["bob"].rating, 1500)

    def test_non_participants_rd_decays(self):
        """Non-participants should have RD increase."""
        ratings = {
            "alice": GlickoRating(1500, 100),
            "bob": GlickoRating(1500, 100),
            "charlie": GlickoRating(1500, 100),
        }
        results = [("alice", 100), ("bob", 90)]  # charlie doesn't play
        new_ratings = process_round(results, ratings.copy())
        self.assertGreater(new_ratings["charlie"].rd, 100)

    def test_ties_handled(self):
        """Tied scores should result in minimal rating change."""
        ratings = {
            "alice": GlickoRating(1500, 100),
            "bob": GlickoRating(1500, 100),
        }
        results = [("alice", 100), ("bob", 100)]  # tie
        new_ratings = process_round(results, ratings.copy())
        # Both should stay close to 1500
        self.assertAlmostEqual(new_ratings["alice"].rating, 1500, delta=5)
        self.assertAlmostEqual(new_ratings["bob"].rating, 1500, delta=5)

    def test_multiple_players_ordering(self):
        """Should correctly handle finish order with multiple players."""
        ratings = {}
        results = [
            ("first", 100),
            ("second", 80),
            ("third", 60),
            ("fourth", 40),
        ]
        new_ratings = process_round(results, ratings)
        # Ratings should be ordered by finish
        self.assertGreater(new_ratings["first"].rating, new_ratings["second"].rating)
        self.assertGreater(new_ratings["second"].rating, new_ratings["third"].rating)
        self.assertGreater(new_ratings["third"].rating, new_ratings["fourth"].rating)


if __name__ == "__main__":
    unittest.main()
