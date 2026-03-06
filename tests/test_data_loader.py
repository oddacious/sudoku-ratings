"""Tests for the data_loader module."""

import unittest

import polars as pl

from ratings.data_loader import load_all_data


class TestLoadAllData(unittest.TestCase):
    """Test the load_all_data function."""

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests in this class."""
        cls.df = load_all_data()

    def test_returns_dataframe(self):
        """load_all_data should return a Polars DataFrame."""
        self.assertIsInstance(self.df, pl.DataFrame)

    def test_has_user_pseudo_id(self):
        """DataFrame should have user_pseudo_id column."""
        self.assertIn("user_pseudo_id", self.df.columns)

    def test_has_year(self):
        """DataFrame should have year column."""
        self.assertIn("year", self.df.columns)

    def test_has_gp_columns(self):
        """DataFrame should have GP point columns."""
        gp_columns = [col for col in self.df.columns if col.startswith("GP_t") and "points" in col]
        self.assertGreater(len(gp_columns), 0)
        self.assertIn("GP_t1 points", self.df.columns)

    def test_has_wsc_columns(self):
        """DataFrame should have WSC point columns."""
        wsc_columns = [col for col in self.df.columns if col.startswith("WSC_t") and "points" in col]
        self.assertGreater(len(wsc_columns), 0)
        self.assertIn("WSC_t1 points", self.df.columns)

    def test_user_pseudo_id_not_null(self):
        """All rows should have a user_pseudo_id."""
        null_count = self.df["user_pseudo_id"].null_count()
        self.assertEqual(null_count, 0)

    def test_year_range(self):
        """Years should span from 2010 to at least 2024."""
        years = sorted(self.df["year"].unique().to_list())
        self.assertEqual(years[0], 2010)
        self.assertGreaterEqual(years[-1], 2024)

    def test_row_count(self):
        """Should have a substantial number of rows."""
        # Expect at least 10000 solver-year combinations
        self.assertGreater(len(self.df), 10000)

    def test_has_name_column(self):
        """DataFrame should have Name column."""
        self.assertIn("Name", self.df.columns)

    def test_gp_and_wsc_data_present(self):
        """Both GP and WSC data should be present (not all null)."""
        gp_non_null = self.df["GP_t1 points"].drop_nulls()
        wsc_non_null = self.df["WSC_t1 points"].drop_nulls()

        self.assertGreater(len(gp_non_null), 0, "GP data should have some non-null values")
        self.assertGreater(len(wsc_non_null), 0, "WSC data should have some non-null values")


if __name__ == "__main__":
    unittest.main()
