"""Shared utilities for rating methods."""


def compute_exp_weighted_mean(history: list[float], decay_rate: float = 0.90) -> float:
    """Compute exponentially weighted mean of history.

    More recent values receive higher weight. The weight for the i-th oldest
    value is decay_rate^i, with the most recent value getting weight 1.0.

    Args:
        history: List of past values (oldest first)
        decay_rate: Weight decay per position (0.90 = 10% decay per step)

    Returns:
        Exponentially weighted mean, or 0.0 if history is empty
    """
    if not history:
        return 0.0
    weights = [decay_rate ** i for i in range(len(history))]
    weights.reverse()  # Most recent gets highest weight
    return sum(h * w for h, w in zip(history, weights)) / sum(weights)


def sample_std(values: list[float]) -> float:
    """Calculate sample standard deviation.

    Uses Bessel's correction (n-1 denominator) for unbiased estimation.

    Args:
        values: List of numeric values

    Returns:
        Sample standard deviation, or 0.0 if fewer than 2 values
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
