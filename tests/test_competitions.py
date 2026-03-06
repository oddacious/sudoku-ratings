"""Tests for the competitions module."""

import unittest

from ratings.competitions import (
    CompetitionIdentifier,
    identify_n_prior_competitions,
    get_prior_gp_rounds,
    gp_rounds_by_year,
    wsc_rounds_by_year,
    get_all_years)

class TestIdentifyNPriorEvents(unittest.TestCase):
    """Test the identify_n_prior_competitions function."""
    def test_gp_in_year(self):
        """Test on GP-only for in-year prior events."""
        event = CompetitionIdentifier(year=2022, round=6, event_type="GP")
        expected = [
            CompetitionIdentifier(year=2022, round=5, event_type="GP"),
            CompetitionIdentifier(year=2022, round=4, event_type="GP"),
            CompetitionIdentifier(year=2022, round=3, event_type="GP")
        ]
        self.assertEqual(identify_n_prior_competitions(event, 3), expected)

    def test_wsc_through_gp(self):
        """Test on WSC-only for in-year prior events."""
        event = CompetitionIdentifier(year=2022, round=3, event_type="WSC")
        expected = [
            CompetitionIdentifier(year=2022, round=2, event_type="WSC"),
            CompetitionIdentifier(year=2022, round=1, event_type="WSC"),
            CompetitionIdentifier(year=2022, round=8, event_type="GP"),
            CompetitionIdentifier(year=2022, round=7, event_type="GP")
        ]
        self.assertEqual(identify_n_prior_competitions(event, 4), expected)

    def test_multi_year_typical(self):
        """Test across years for GP and WSC."""
        event = CompetitionIdentifier(year=2024, round=3, event_type="GP")
        expected = [
            CompetitionIdentifier(year=2024, round=2, event_type="GP"),
            CompetitionIdentifier(year=2024, round=1, event_type="GP"),
            CompetitionIdentifier(year=2023, round=10, event_type="WSC"),
            CompetitionIdentifier(year=2023, round=9, event_type="WSC")
        ]
        self.assertEqual(identify_n_prior_competitions(event, 4), expected)

    def test_multi_year_no_wsc(self):
        """Test across years, covering years without a WSC."""
        event = CompetitionIdentifier(year=2022, round=3, event_type="GP")
        expected = [
            CompetitionIdentifier(year=2022, round=2, event_type="GP"),
            CompetitionIdentifier(year=2022, round=1, event_type="GP"),
            CompetitionIdentifier(year=2021, round=8, event_type="GP"),
            CompetitionIdentifier(year=2021, round=7, event_type="GP")
        ]
        self.assertEqual(identify_n_prior_competitions(event, 4), expected)

class TestGetAllYears(unittest.TestCase):
    """Test the get_all_years function."""
    def test_gp_only(self):
        """Test the GP years."""
        gp_years = gp_rounds_by_year()
        expected = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
        self.assertEqual(get_all_years([gp_years]), expected)

    def test_wsc_only(self):
        """Test the WSC years."""
        wsc_years = wsc_rounds_by_year()
        expected = [2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025]
        self.assertEqual(get_all_years([wsc_years]), expected)

    def test_all_competitions(self):
        """Test both GP and WSC together."""
        gp_years = gp_rounds_by_year()
        wsc_years = wsc_rounds_by_year()
        expected = [2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
                    2024, 2025, 2026]
        self.assertEqual(get_all_years([gp_years, wsc_years]), expected)

class TestGetPriorGpRounds(unittest.TestCase):
    """Test the get_prior_gp_rounds function for GP-baseline difficulty."""

    def test_gp_round_excludes_same_and_later(self):
        """For a GP round, should return all prior GP rounds only."""
        event = CompetitionIdentifier(year=2022, round=3, event_type="GP")
        result = get_prior_gp_rounds(event)

        # Should include 2022 R1, R2 and all prior years' GP rounds
        self.assertIn(CompetitionIdentifier(2022, 1, "GP"), result)
        self.assertIn(CompetitionIdentifier(2022, 2, "GP"), result)
        self.assertIn(CompetitionIdentifier(2021, 8, "GP"), result)

        # Should NOT include R3 or later
        self.assertNotIn(CompetitionIdentifier(2022, 3, "GP"), result)
        self.assertNotIn(CompetitionIdentifier(2022, 4, "GP"), result)

        # Should NOT include any WSC
        for comp in result:
            self.assertEqual(comp.event_type, "GP")

    def test_wsc_round_includes_same_year_gp(self):
        """For a WSC round, should include all GP rounds up to that year."""
        event = CompetitionIdentifier(year=2022, round=5, event_type="WSC")
        result = get_prior_gp_rounds(event)

        # Should include all 2022 GP rounds (WSC happens after GP)
        for rnd in range(1, 9):
            self.assertIn(CompetitionIdentifier(2022, rnd, "GP"), result)

        # Should include prior years too
        self.assertIn(CompetitionIdentifier(2021, 8, "GP"), result)

        # Should NOT include any WSC
        for comp in result:
            self.assertEqual(comp.event_type, "GP")

    def test_first_gp_round_returns_empty(self):
        """First GP round ever should return empty list."""
        event = CompetitionIdentifier(year=2014, round=1, event_type="GP")
        result = get_prior_gp_rounds(event)
        self.assertEqual(result, [])

    def test_early_wsc_returns_2014_gp_as_anchor(self):
        """WSC in 2010 (before GP existed) should return 2014 GP as anchor."""
        event = CompetitionIdentifier(year=2010, round=1, event_type="WSC")
        result = get_prior_gp_rounds(event)
        # Pre-2014 WSC uses 2014 GP as anchor for calibration
        self.assertEqual(len(result), 7)  # 2014 had 7 GP rounds
        self.assertEqual(result[0], CompetitionIdentifier(2014, 1, "GP"))

if __name__ == "__main__":
    unittest.main()
