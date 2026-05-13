"""Tests for the competitions module."""

import unittest
from datetime import date
from unittest.mock import patch

from ratings.competitions import (
    COMPETITION_DATES,
    CompetitionIdentifier,
    identify_n_prior_competitions,
    get_prior_gp_rounds,
    get_competition_date,
    get_all_competitions,
    find_ambiguous_orderings,
    esc_rounds_by_year,
    all_esc_round_names,
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

class TestESCRoundsByYear(unittest.TestCase):
    """Test esc_rounds_by_year."""

    def test_2026_has_seven_rounds(self):
        rounds = esc_rounds_by_year()
        self.assertIn(2026, rounds)
        self.assertEqual(list(rounds[2026]), list(range(1, 8)))

    def test_unknown_year_absent(self):
        rounds = esc_rounds_by_year()
        self.assertNotIn(2025, rounds)
        self.assertNotIn(2020, rounds)


class TestAllESCRoundNames(unittest.TestCase):
    """Test all_esc_round_names."""

    def test_prefix_is_esc(self):
        names = all_esc_round_names()
        self.assertTrue(all(n.startswith("ESC_t") for n in names))

    def test_round_seven_included(self):
        names = all_esc_round_names()
        self.assertIn("ESC_t7 points", names)

    def test_round_one_included(self):
        names = all_esc_round_names()
        self.assertIn("ESC_t1 points", names)


class TestGetAllCompetitionsESC(unittest.TestCase):
    """Test that get_all_competitions includes ESC rounds."""

    def test_esc_2026_present(self):
        all_comps = get_all_competitions()
        esc_2026 = [c for c in all_comps if c.event_type == "ESC" and c.year == 2026]
        self.assertEqual(len(esc_2026), 7)
        self.assertEqual(sorted(c.round for c in esc_2026), list(range(1, 8)))

    def test_esc_ordering_without_gp_dates(self):
        """Without GP dates, ESC 2026 rounds should come before undated GP 2026 rounds.

        This uses the typical-month sentinel (GP≈March, ESC≈May). Since ESC has known
        dates in May, and GP has no dates (sentinel March), GP rounds cluster before ESC.
        """
        all_comps = get_all_competitions()
        year_2026 = [c for c in all_comps if c.year == 2026]

        gp_indices = [i for i, c in enumerate(year_2026) if c.event_type == "GP"]
        esc_indices = [i for i, c in enumerate(year_2026) if c.event_type == "ESC"]

        # All GP rounds (sentinel March) should come before all ESC rounds (dated May 12)
        self.assertGreater(min(esc_indices), max(gp_indices))

    def test_esc_ordering_respects_gp_dates(self):
        """When a GP round has a date after ESC, it should sort after ESC."""
        # Inject a date for GP 2026 round 5 that is after ESC's date
        fake_dates = dict(COMPETITION_DATES)
        fake_dates[(2026, 5, "GP")] = date(2026, 6, 15)

        with patch("ratings.competitions.COMPETITION_DATES", fake_dates):
            all_comps = get_all_competitions()

        year_2026 = [c for c in all_comps if c.year == 2026]
        esc_r1_idx = next(i for i, c in enumerate(year_2026) if c.event_type == "ESC" and c.round == 1)
        gp_r5_idx = next(i for i, c in enumerate(year_2026) if c.event_type == "GP" and c.round == 5)

        self.assertGreater(gp_r5_idx, esc_r1_idx)


class TestGetPriorGPRoundsESC(unittest.TestCase):
    """Test get_prior_gp_rounds for ESC competitions."""

    def test_esc_includes_same_year_gp(self):
        """ESC 2026 should use all GP 2026 rounds as baseline (like WSC)."""
        event = CompetitionIdentifier(year=2026, round=1, event_type="ESC")
        result = get_prior_gp_rounds(event)

        for rnd in range(1, 9):
            self.assertIn(CompetitionIdentifier(2026, rnd, "GP"), result)

    def test_esc_includes_prior_year_gp(self):
        """ESC 2026 baseline should also include 2025 and earlier GP rounds."""
        event = CompetitionIdentifier(year=2026, round=1, event_type="ESC")
        result = get_prior_gp_rounds(event)

        self.assertIn(CompetitionIdentifier(2025, 8, "GP"), result)
        self.assertIn(CompetitionIdentifier(2024, 1, "GP"), result)

    def test_esc_contains_only_gp(self):
        """Baseline for ESC should never include WSC or ESC rounds."""
        event = CompetitionIdentifier(year=2026, round=4, event_type="ESC")
        result = get_prior_gp_rounds(event)

        for comp in result:
            self.assertEqual(comp.event_type, "GP")


class TestFindAmbiguousOrderings(unittest.TestCase):
    """Test find_ambiguous_orderings."""

    def test_detects_esc_gp_ambiguity(self):
        """Should flag 2026 GP as ambiguous relative to dated ESC 2026."""
        warnings = find_ambiguous_orderings()
        # ESC 2026 has dates; GP 2026 does not — should be flagged
        gp_warnings = [w for w in warnings if "2026" in w and "GP" in w]
        self.assertTrue(len(gp_warnings) > 0)

    def test_clears_when_gp_dated_around_esc(self):
        """No ambiguity when all GP 2026 rounds are given dates."""
        fake_dates = dict(COMPETITION_DATES)
        # Assign one date per GP round spread across the year
        for rnd in range(1, 9):
            fake_dates[(2026, rnd, "GP")] = date(2026, rnd, 1)

        with patch("ratings.competitions.COMPETITION_DATES", fake_dates):
            warnings = find_ambiguous_orderings()

        gp_2026_warnings = [w for w in warnings if "2026" in w and "GP" in w]
        self.assertEqual(len(gp_2026_warnings), 0)

    def test_no_warnings_for_undated_gp_wsc_only_years(self):
        """Years with only undated events (e.g., 2022 GP + WSC, both undated) emit no warnings."""
        # In 2022, neither GP nor WSC have dates in COMPETITION_DATES
        warnings = find_ambiguous_orderings()
        year_2022_warnings = [w for w in warnings if w.startswith("2022")]
        self.assertEqual(len(year_2022_warnings), 0)


if __name__ == "__main__":
    unittest.main()
