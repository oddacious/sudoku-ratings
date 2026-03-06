# Data Errata and Known Issues

This document tracks data quality issues, anomalous rounds, and limitations of the competition data that affect rating calculations.

## GP vs WSC Scoring Incompatibility

### Problem

The difficulty adjustment system does not fully normalize scores between GP and WSC competitions. Top solvers consistently show ~30-40% higher adjusted scores in GP than WSC, despite similar percentile performance.

**Example - Tantan Dai (2024-2026):**
- GP: 999 avg adjusted points, 99.5% avg percentile
- WSC: 768 avg adjusted points, 97.8% avg percentile
- Gap: 30% higher in GP

**Example - Tiit Vunk (2024-2026):**
- GP: 886 avg adjusted points, 98.9% avg percentile
- WSC: 620 avg adjusted points, 91.8% avg percentile
- Gap: 43% higher in GP

### Root Causes

1. **Different participant pools**: GP has 600-1000 participants ranging from casual to elite. WSC has ~150-250 self-selected elite solvers. Top performers can dominate GP (4x round mean) but only moderately exceed WSC field (3x round mean).

2. **Variable WSC scoring scales**: WSC max scores vary wildly by round (195 to 1110 in 2024-2025), while GP is consistent (~900-1100). The difficulty calculation can't distinguish "more points available" from "easier puzzles."

3. **Difficulty calculated from mixed history**: When calculating difficulty for a WSC round, the reference includes both GP rounds (high scores) and WSC rounds (low scores). This conflates different scoring systems.

### Impact

Solvers who participate in both GP and WSC have their ratings dragged down by WSC participation. Solvers who only do GP appear artificially stronger.

### Solution: GP-Baseline Difficulty

The `use_gp_baseline=True` parameter in `difficulty_of_rounds()` fixes this by using only GP history as the baseline for all competitions:

- For GP rounds: uses prior GP rounds as anchors
- For WSC rounds: uses all GP rounds up to that year as anchors

This treats GP as the "canonical" scale and normalizes WSC scores to it.

---

## Population Mismatch Bug in Difficulty Calculation (Fixed)

### Problem

The `relative_difficulty_outcome_weighted()` function had a population mismatch between its numerator and denominator that caused WSC difficulty to be systematically underestimated by ~28%, inflating WSC adjusted points.

**The bug:**
```
difficulty = mean(ALL participants in round) / mean(anchor history of participants WITH history)
```

The numerator included all participants, but the denominator only reflected those with anchor (GP) history. Since participants without GP history tend to score lower in WSC, they pulled down the numerator without affecting the denominator.

**Example - 2017 WSC R1:**
- Total participants: 207
- With GP history: 121 (mean WSC score: 153.8)
- Without GP history: 86 (mean WSC score: 73.5)
- Buggy difficulty: 120.4 / 243.7 = 0.494
- Correct difficulty: 153.8 / 243.7 = 0.631
- Error: 28% underestimate

### Impact

This bug caused WSC-only players to have artificially inflated ratings. For example, Letian Ming (who only participated in WSC from 2017-2019) was incorrectly ranked #1 despite being outside the top 10 in actual WSC total points.

Calibration test (2017, players with both GP and WSC):
- Before fix: WSC adj / GP adj ratio = **1.50** (should be ~1.0)
- After fix: WSC adj / GP adj ratio = **1.20**

### Fix

The fix restricts the numerator to only include participants who have anchor history, ensuring matching populations:

```
difficulty = mean(participants WITH anchor history) / mean(anchor history)
```

This was fixed in `ratings/competition_difficulty.py` in the `relative_difficulty_outcome_weighted()` function.

**Usage:**
```python
from ratings.competition_difficulty import difficulty_of_all_rounds
difficulties = difficulty_of_all_rounds(normalized_data, use_gp_baseline=True)
```

### Other Potential Solutions (not implemented)

- Use percentile-based scoring instead of raw points
- Normalize by max score per round before difficulty adjustment

---

## Anomalous WSC Rounds

### Overview

Approximately 10% of WSC rounds (12 out of 125) have unusual scoring patterns that don't fit the standard model of "more skill = more points."

### All-or-Nothing Rounds

These rounds have binary outcomes where solvers either complete correctly (get base points + time bonus) or score 0/minimal points from a single mistake.

| Year | Round | Zero% | Max | Median | Pattern |
|------|-------|-------|-----|--------|---------|
| 2012 | 7 | 71% | 55 | 0 | Extreme all-or-nothing |
| 2016 | 12 | 56% | 340 | 0 | Extreme all-or-nothing |
| 2024 | 6 | 38% | 195 | 8 | All-or-nothing with time bonus |

**2024 WSC R6 Details:**
- Format: 150 points for correct completion, up to ~45 point time bonus
- 58 of 152 solvers (38%) scored 0
- Elite solvers with 0: Tiit Vunk (avg 386 other rounds), Thomas Snyder (avg 321), Rintaro Matsumoto (avg 283)
- Even Kota Morinishi only scored 8 points

These zeros represent "made an error" or "ran out of time," not poor performance in the normal sense.

### High-Zero Rounds (>20% zeros)

| Year | Round | Zero% | Notes |
|------|-------|-------|-------|
| 2011 | 2 | 27% | |
| 2014 | 10 | 34% | |
| 2015 | 4 | 24% | |
| 2015 | 9 | 29% | |
| 2017 | 11 | 22% | |
| 2022 | 4 | 26% | |
| 2022 | 5 | 31% | |

### Clustered-Score Rounds

Some rounds show unusual clustering at specific point values, suggesting fixed-point scoring:

| Year | Round | Mode | Mode% |
|------|-------|------|-------|
| 2018 | 7 | 80 | 39% |
| 2019 | 12 | 70 | 32% |

### Solution: Difficulty Floor

The `DIFFICULTY_FLOOR = 0.1` constant in `regression.py` caps the minimum difficulty used when calculating adjusted points. This prevents extreme inflation while still giving credit to those who succeed in difficult rounds:

- **Without floor**: 55 raw points / 0.02 difficulty = 2,750 adjusted points (50x inflation)
- **With floor**: 55 raw points / 0.10 difficulty = 550 adjusted points (10x inflation)

This affects only 5 rounds (1% of records) while preserving relative rankings within each round.

**Rounds affected:**
| Year | Round | Original Diff | Capped Diff |
|------|-------|---------------|-------------|
| 2010 | WSC R6 | 0.058 | 0.10 |
| 2012 | WSC R1 | 0.034 | 0.10 |
| 2012 | WSC R4 | 0.062 | 0.10 |
| 2012 | WSC R6 | 0.086 | 0.10 |
| 2012 | WSC R7 | 0.020 | 0.10 |

### Alternative Approaches (not implemented)

1. **Exclude extreme rounds entirely** - loses information about who succeeded
2. **Use percentile/rank** instead of points - requires more infrastructure changes

---

## WSC Score Distribution Variance

### Problem

WSC rounds have highly inconsistent score distributions, making cross-round comparisons unreliable even within WSC.

**Max/Mean ratios (2024-2025 WSC):**
- Lowest: 1.89 (2025 R7 - compressed, everyone near max)
- Highest: 4.71 (2024 R6 - most near zero)

**Max/Median ratios:**
- Lowest: 1.78 (2025 R7)
- Highest: 24.38 (2024 R6)

**Compare to GP:** Max/Mean consistently 3.5-4.8x, Max/Median consistently 4.1-5.9x

### Example: 2025 WSC R7

- Max: 750, Mean: 396, Median: 420
- Max/Mean: 1.89 (very compressed)
- Top 5 all scored 740-750
- Difficulty calculated as 1.695 ("easy") because everyone scored high
- Result: Tantan Dai's 740 points became only 437 adjusted

This round wasn't necessarily "easy" - it may have had more points available or a different format. The difficulty adjustment penalizes everyone equally for the high scores.

---

## GP Scoring Scale Change (2015 → 2016)

### Problem

GP changed its scoring system between 2015 and 2016 from a 0-100 scale to a 0-1000 scale (~8.5x increase). This caused difficulty calculations to be wildly incorrect when comparing across the boundary:

- 2014-2015 GP: mean ~30, max 100
- 2016+ GP: mean ~250, max ~950

Without normalization, 2016 GP R1 had difficulty=7.4 (instead of ~1.0) because participants appeared to score 7x their historical average.

### Impact

This caused ratings for established players to artificially quadruple as their history filled with post-2016 data. Kota Morinishi's rating went from 107 to 434 over 2016-2017—not because he improved 4x, but because the baseline was recalibrating.

### Solution

The `normalize_gp_scoring_scale()` function in `competition_results.py` multiplies 2014-2015 GP scores by 8.5 to match the 2016+ scale. This normalization factor was calculated from mean scores of participants who competed in both periods.

After normalization:
- GP difficulty is stable across all years (0.7-1.1 range)
- Kota's rating change over 2016-2017: 1.23x (reasonable) instead of 4.0x (impossible)

---

## Pre-2014 WSC Difficulty Anchor

### Problem

WSC data exists from 2010, but GP only started in 2014. Without a baseline, pre-2014 WSC rounds had difficulty=1.0, causing raw scores to be used directly. Since early WSC scoring was on a different scale, this created a discontinuity when GP-based difficulty adjustment kicked in (2014+).

### Solution

For pre-2014 WSC rounds, `get_prior_gp_rounds()` returns the 2014 GP rounds as the anchor. This allows early WSC to be calibrated against the first available GP data.

After this fix:
- Kota's rating range: 508-953 (was 75-953)
- Total rating change over career: 1.88x (was 12.69x)

---

## GP Data Quality

GP data is significantly more consistent than WSC:
- Stable participant counts (500-1000 per round)
- Consistent max scores (~900-1100)
- Consistent score distributions (Max/Mean ~4x)
- No anomalous all-or-nothing rounds detected

GP rounds can be compared directly with reasonable confidence.
