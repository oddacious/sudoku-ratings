"""Tests for the competition_difficulty module."""

import unittest

import numpy as np
import polars as pl

from ratings.competitions import CompetitionIdentifier

from ratings.competition_difficulty import (
    CompetitionDifficulty,
    relative_difficulty_solver_weighted,
    relative_difficulty_outcome_weighted,
    difficulty_of_rounds)

class TestRelativeDifficultySolverWeighted(unittest.TestCase):
    """Test the relative_difficulty_solver_weighted function."""
    def test_single_solver(self):
        """Test with only one solver."""
        target = CompetitionIdentifier(year=2022, round=6, event_type="GP")
        anchors = [
            CompetitionIdentifier(year=2022, round=4, event_type="GP"),
            CompetitionIdentifier(year=2022, round=3, event_type="GP")]

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2022, "round": 6, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 1, "year": 2022, "round": 4, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 3, "competition": "GP", "points": 10},
        ])

        outcome = relative_difficulty_solver_weighted(target, anchors, test_data)
        expected = 1.0

        self.assertEqual(outcome, expected)

    def test_multiple_solvers(self):
        """Test with multiple solvers."""
        target = CompetitionIdentifier(year=2022, round=6, event_type="GP")
        anchors = [
            CompetitionIdentifier(year=2022, round=4, event_type="GP"),
            CompetitionIdentifier(year=2022, round=3, event_type="WSC")]

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2022, "round": 6, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 1, "year": 2022, "round": 4, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 3, "competition": "GP", "points": 10},
            {"user_pseudo_id": 2, "year": 2022, "round": 6, "competition": "GP", "points": 20},
            {"user_pseudo_id": 2, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 2, "year": 2022, "round": 4, "competition": "GP", "points": 20},
        ])

        outcome = relative_difficulty_solver_weighted(target, anchors, test_data)
        expected = 1.0

        self.assertEqual(outcome, expected)

    def test_varying_solvers(self):
        """Test with multiple solvers and different relative performances."""
        target = CompetitionIdentifier(year=2022, round=6, event_type="GP")
        anchors = [
            CompetitionIdentifier(year=2022, round=4, event_type="GP"),
            CompetitionIdentifier(year=2022, round=3, event_type="WSC")]

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2022, "round": 6, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 1, "year": 2022, "round": 4, "competition": "GP", "points": 20},
            {"user_pseudo_id": 1, "year": 2022, "round": 3, "competition": "GP", "points": 20},
            {"user_pseudo_id": 2, "year": 2022, "round": 6, "competition": "GP", "points": 20},
            {"user_pseudo_id": 2, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 2, "year": 2022, "round": 4, "competition": "GP", "points": 5},
        ])

        outcome = relative_difficulty_solver_weighted(target, anchors, test_data)
        expected = (0.5 + 4.0) / 2

        self.assertEqual(outcome, expected)

class TestRelativeDifficultyOutcomeWeighted(unittest.TestCase):
    """Test the relative_difficulty_outcome_weighted function."""
    def test_single_solver(self):
        """Test with only one solver."""
        target = CompetitionIdentifier(year=2022, round=6, event_type="GP")
        anchors = [
            CompetitionIdentifier(year=2022, round=4, event_type="GP"),
            CompetitionIdentifier(year=2022, round=3, event_type="GP")]

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2022, "round": 6, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 1, "year": 2022, "round": 4, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 3, "competition": "GP", "points": 10},
        ])

        outcome = relative_difficulty_outcome_weighted(target, anchors, test_data)
        expected = 1.0

        self.assertEqual(outcome, expected)

    def test_multiple_solvers(self):
        """Test with multiple solvers."""
        target = CompetitionIdentifier(year=2022, round=6, event_type="GP")
        anchors = [
            CompetitionIdentifier(year=2022, round=4, event_type="GP"),
            CompetitionIdentifier(year=2022, round=3, event_type="WSC")]

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2022, "round": 6, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 1, "year": 2022, "round": 4, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 3, "competition": "WSC", "points": 10},
            {"user_pseudo_id": 2, "year": 2022, "round": 6, "competition": "GP", "points": 20},
            {"user_pseudo_id": 2, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 2, "year": 2022, "round": 4, "competition": "GP", "points": 20},
        ])

        outcome = relative_difficulty_outcome_weighted(target, anchors, test_data)
        expected = 15.0 / (40.0 / 3)

        self.assertEqual(outcome, expected)

    def test_varying_solvers(self):
        """Test with multiple solvers and different relative performances."""
        target = CompetitionIdentifier(year=2022, round=6, event_type="GP")
        anchors = [
            CompetitionIdentifier(year=2022, round=4, event_type="GP"),
            CompetitionIdentifier(year=2022, round=3, event_type="WSC")]

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2022, "round": 6, "competition": "GP", "points": 10},
            {"user_pseudo_id": 1, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 1, "year": 2022, "round": 4, "competition": "GP", "points": 20},
            {"user_pseudo_id": 1, "year": 2022, "round": 3, "competition": "WSC", "points": 20},
            {"user_pseudo_id": 2, "year": 2022, "round": 6, "competition": "GP", "points": 40},
            {"user_pseudo_id": 2, "year": 2022, "round": 5, "competition": "GP", "points": 0},
            {"user_pseudo_id": 2, "year": 2022, "round": 4, "competition": "GP", "points": 5},
        ])

        outcome = relative_difficulty_outcome_weighted(target, anchors, test_data)
        expected = (50.0 / 2) / (45.0 / 3)

        self.assertEqual(outcome, expected)

class TestDifficultyOfRounds(unittest.TestCase):
    """Test the difficulty_of_rounds function."""
    def test_non_truncating(self):
        """Confirm that the truncate setting is respected."""
        # First round in our data
        target = CompetitionIdentifier(year=2010, round=1, event_type="WSC")

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2010, "round": 1, "competition": "WSC", "points": 10},
        ])

        outcome = difficulty_of_rounds([target], test_data, 16, False)
        expected = [
            CompetitionDifficulty(
                competition=CompetitionIdentifier(year=2010, round=1, event_type="WSC"),
                outcome_weighted=np.nan,
                n_reference=16,
                truncate_enabled=False)]

        self.assertEqual(outcome, expected)

    def test_oldest_round_with_truncate(self):
        """Confirm that the oldest round would be anchored to 1.0."""
        target = CompetitionIdentifier(year=2010, round=1, event_type="WSC")

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2010, "round": 1, "competition": "WSC", "points": 10},
        ])

        outcome = difficulty_of_rounds([target], test_data, 16, True)
        expected = [
            CompetitionDifficulty(
                competition=CompetitionIdentifier(year=2010, round=1, event_type="WSC"),
                outcome_weighted=1.0,
                n_reference=16,
                truncate_enabled=True)]

        self.assertEqual(outcome, expected)

    def test_multiple_rounds(self):
        """Test that difficulty is correctly calculated and returned for multiple rounds."""
        competitions = [
            CompetitionIdentifier(year=2010, round=3, event_type="WSC"),
            CompetitionIdentifier(year=2010, round=2, event_type="WSC"),
            CompetitionIdentifier(year=2010, round=1, event_type="WSC"),
        ]

        test_data = pl.DataFrame([
            {"user_pseudo_id": 1, "year": 2010, "round": 1, "competition": "WSC", "points": 10},
            {"user_pseudo_id": 1, "year": 2010, "round": 2, "competition": "WSC", "points": 20},
            {"user_pseudo_id": 1, "year": 2010, "round": 3, "competition": "WSC", "points": 60},
        ])

        outcome = difficulty_of_rounds(competitions, test_data, 16, True)
        expected = [
            CompetitionDifficulty(
                competition=CompetitionIdentifier(year=2010, round=3, event_type="WSC"),
                outcome_weighted=4.0,
                n_reference=16,
                truncate_enabled=True),
            CompetitionDifficulty(
                competition=CompetitionIdentifier(year=2010, round=2, event_type="WSC"),
                outcome_weighted=2.0,
                n_reference=16,
                truncate_enabled=True),
            CompetitionDifficulty(
                competition=CompetitionIdentifier(year=2010, round=1, event_type="WSC"),
                outcome_weighted=1.0,
                n_reference=16,
                truncate_enabled=True)]

        self.assertEqual(outcome, expected)

if __name__ == "__main__":
    unittest.main()
