"""Tests for ratings.methods.utils shared utilities."""

import unittest
import math

from ratings.methods.utils import sample_std, compute_exp_weighted_mean


class TestSampleStd(unittest.TestCase):
    """Test the sample_std function."""

    def test_empty_list(self):
        """Test that empty list returns 0."""
        self.assertEqual(sample_std([]), 0.0)

    def test_single_value(self):
        """Test that single value returns 0."""
        self.assertEqual(sample_std([5.0]), 0.0)

    def test_two_values(self):
        """Test standard deviation with two values."""
        # std([0, 2]) with n-1 = sqrt((1+1)/1) = sqrt(2)
        result = sample_std([0.0, 2.0])
        expected = math.sqrt(2)
        self.assertAlmostEqual(result, expected)

    def test_known_values(self):
        """Test with known values and expected result."""
        # [2, 4, 4, 4, 5, 5, 7, 9]
        # Mean = 5, variance = 32/7, std = sqrt(32/7) ≈ 2.138
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = sample_std(values)
        expected = math.sqrt(32 / 7)
        self.assertAlmostEqual(result, expected, places=5)

    def test_identical_values(self):
        """Test that identical values returns 0."""
        self.assertEqual(sample_std([5.0, 5.0, 5.0, 5.0]), 0.0)

    def test_negative_values(self):
        """Test that negative values work correctly."""
        result = sample_std([-2.0, 0.0, 2.0])
        # Mean = 0, variance = (4+0+4)/2 = 4, std = 2
        self.assertAlmostEqual(result, 2.0)


class TestComputeExpWeightedMean(unittest.TestCase):
    """Test the compute_exp_weighted_mean function."""

    def test_empty_list(self):
        """Test that empty list returns 0."""
        self.assertEqual(compute_exp_weighted_mean([]), 0.0)

    def test_single_value(self):
        """Test that single value returns that value."""
        self.assertEqual(compute_exp_weighted_mean([5.0]), 5.0)
        self.assertEqual(compute_exp_weighted_mean([100.0]), 100.0)

    def test_uniform_values(self):
        """Test that uniform values return that value regardless of decay."""
        self.assertEqual(compute_exp_weighted_mean([5.0, 5.0, 5.0]), 5.0)
        self.assertEqual(compute_exp_weighted_mean([5.0, 5.0, 5.0], decay_rate=0.5), 5.0)

    def test_decay_rate_1_is_simple_mean(self):
        """Test that decay_rate=1.0 produces simple mean (all weights equal)."""
        values = [10.0, 20.0, 30.0]
        result = compute_exp_weighted_mean(values, decay_rate=1.0)
        expected = 20.0  # Simple mean
        self.assertAlmostEqual(result, expected)

    def test_recent_values_weighted_higher(self):
        """Test that more recent values (end of list) have higher weight."""
        # With [10, 100], the 100 is more recent and should pull mean above 55
        result = compute_exp_weighted_mean([10.0, 100.0], decay_rate=0.5)
        self.assertGreater(result, 55.0)  # Higher than simple mean

        # With [100, 10], the 10 is more recent and should pull mean below 55
        result2 = compute_exp_weighted_mean([100.0, 10.0], decay_rate=0.5)
        self.assertLess(result2, 55.0)  # Lower than simple mean

    def test_known_calculation(self):
        """Test with hand-calculated expected value."""
        # [a, b, c] with decay_rate=0.5
        # weights (reversed): [0.5^2, 0.5^1, 0.5^0] = [0.25, 0.5, 1.0]
        # weighted sum = 10*0.25 + 20*0.5 + 30*1.0 = 2.5 + 10 + 30 = 42.5
        # weight sum = 0.25 + 0.5 + 1.0 = 1.75
        # result = 42.5 / 1.75 = 24.2857...
        values = [10.0, 20.0, 30.0]
        result = compute_exp_weighted_mean(values, decay_rate=0.5)
        expected = 42.5 / 1.75
        self.assertAlmostEqual(result, expected)

    def test_default_decay_rate(self):
        """Test that default decay rate is 0.90."""
        values = [100.0, 200.0]
        # With decay_rate=0.9: weights = [0.9, 1.0]
        # result = (100*0.9 + 200*1.0) / (0.9 + 1.0) = 290/1.9
        result = compute_exp_weighted_mean(values)
        expected = 290 / 1.9
        self.assertAlmostEqual(result, expected)

    def test_extreme_decay_rate(self):
        """Test with very low decay rate (almost only last value matters)."""
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        result = compute_exp_weighted_mean(values, decay_rate=0.01)
        # With decay_rate=0.01, older values have negligible weight
        # Result should be very close to 100
        self.assertGreater(result, 99.0)


if __name__ == "__main__":
    unittest.main()
