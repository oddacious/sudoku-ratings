# Sudoku Ratings

A rating system for competitive sudoku solvers, analyzing performance data from Grand Prix (GP) and World Sudoku Championship (WSC) events.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Current leaderboard
python main.py leaderboard

# Solver rating history
python main.py solver "Tantan Dai"

# Leadership progression over time
python main.py progression --from-year 2020

# Career records
python main.py records --sort wins

# Export data for external apps
python main.py export --output-dir ./export/

# See all commands
python main.py --help
```

## Rating Method

Ratings use difficulty-adjusted, exponentially-weighted points:

1. **Difficulty adjustment**: Raw points divided by round difficulty (calculated from how participants performed vs. their historical averages)
2. **Exponential weighting**: Recent rounds weighted higher (0.90 decay rate)
3. **Prior regularization**: New solvers regularized toward population mean

Achieves **84.3% pairwise prediction accuracy** on held-out data.

## Testing

```bash
python -m unittest discover tests/
```
