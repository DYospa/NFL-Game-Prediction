# NFL Game Prediction Model Using Machine Learning Techniques

**Authors:** Doron Yospa (dyospa1@student.gsu.edu), Chris Walker (cwalker204@student.gsu.edu)

A binary classification project that predicts the winner of NFL games from team performance statistics available *before* kickoff. Logistic regression serves as the baseline and gradient-boosted trees as the main model. Both are compared against two benchmarks: always picking the home team and picking the Vegas favorite.

## Project Goals

- Predict whether the home team wins each game.
- Train and tune on the 2010-2024 seasons; hold out the 2025 season as the final test set.
- Evaluate with accuracy, precision, recall, F1, AUC, and a confusion matrix.
- Explain predictions with SHAP values to show which features matter most.

## Data

- **Source:** [Kaggle: NFL Game Scores Dataset](https://www.kaggle.com/datasets/keonim/nfl-game-scores-dataset-2017-2023)
- **Expected file:** the bundled `2010-2026_scores.csv.zip`, or an extracted `2010-2026_scores.csv` next to `model.py`.
- **Columns:** `Season, Week, GameStatus, GameSlot, GameDate, AwayTeam, AwayScore, HomeTeam, HomeScore`
- **Notes:**
  - `GameDate` has no year (e.g. "August 8th"), so the year is derived from `Season`. Games in January-July belong to the following calendar year.
  - Only rows with `GameStatus == FINAL` are used. Byes, TBD games, and ties are dropped.
  - The file contains preseason games, which should be filtered out (see Known Issues).
  - It has no team stats (yards, turnovers) and no betting lines. Planned supplement: [nflverse](https://github.com/nflverse) data (`nflreadpy` / `nfl_data_py`) for team stats and `spread_line`.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activate the environment in Command Prompt with `.venv\Scripts\activate.bat`,
or in PowerShell with `.\.venv\Scripts\Activate.ps1`. Then use `python` for the
commands below. Activation is optional: you can always run
`.\.venv\Scripts\python.exe model.py` or `.\.venv\Scripts\python.exe EDA.py` directly.
In your editor, select `.venv\Scripts\python.exe` as the project interpreter.

Run from the project folder; the scores ZIP can stay compressed:

```bash
python model.py
python EDA.py
```

`EDA.py` writes five PNG charts and `analysis_summary.json` into `visualizations`.
You can also supply a CSV/ZIP path and `--output-dir` explicitly.
SHAP is optional: install it with `python -m pip install shap` to display the
model's feature-importance plot.

On Windows, `python` and `py` can select different installations. Use the same
interpreter for installation and execution. This device's Python 3.11 already
has the required packages; use `py -3.11` instead of `python` if needed.

`merge_plays.py` is a separate utility requiring yearly files named
`2010_plays.csv` through `2026_plays.csv`. These are not included in the scores
ZIP. It searches the project folder by default, or a folder you provide:

```powershell
python merge_plays.py --data-dir "D:\path\to\play-data"
```

Available years are merged into `archive/2010_2026_plays.csv`; missing years
are reported and skipped. Without any yearly files, no output is created.

## Method

1. **Load and clean:** keep completed games, parse dates, drop ties, sort chronologically.
2. **Reshape:** convert to one row per team per game.
3. **Rolling features:** for each team, the mean of its previous 5 games (points for, points against, win rate), plus rest days and games played. Every feature is `shift(1)`-ed, so the game being predicted never leaks into its own features.
4. **Game-level table:** join home and away team features and add home-minus-away difference features.
5. **Validation:** expanding-window cross-validation by season (train on seasons before *s*, validate on *s*, for 2018-2024). No shuffling.
6. **Models:** logistic regression (standardized) and `HistGradientBoostingClassifier`.
7. **Final test:** train on all seasons before 2025 and evaluate once on 2025.
8. **Interpretation:** SHAP beeswarm plot on the gradient-boosted model.

## Preliminary Results (first run, before cleaning)

| Model | CV accuracy (2018-2024) | Test accuracy | F1 | AUC |
|---|---|---|---|---|
| Always pick home | n/a | 0.527 | n/a | n/a |
| Logistic regression | 0.602 | 0.627 | 0.670 | 0.657 |
| Gradient-boosted trees | 0.586 | 0.624 | 0.680 | 0.645 |
| Vegas favorite | not yet computed | not yet computed | n/a | n/a |

These numbers are preliminary. The test set had 330 games, more than a full season, because preseason games were still included. With roughly 300 games the standard error on accuracy is about ±3 points, so the two models are statistically indistinguishable.

## Known Issues and Next Steps

- [ ] **Filter out preseason games** (they inflate game counts and add noise to rolling features), then rerun.
- [ ] Add an **Elo rating** built from game scores.
- [ ] Merge **nflverse** data for team stats (rushing, passing, turnovers) and the Vegas **spread** for the second benchmark.
- [ ] Tune the rolling window size and GBT hyperparameters using CV only. Keep 2025 untouched until the final evaluation.
- [ ] Add neutral-site and playoff flags.
- [ ] Generate SHAP plots for the final model.
- [ ] Final write-up and presentation.

## Timeline

| Date | Milestone |
|---|---|
| Sept 27 | Project proposal |
| Oct 19 | Training model complete, training begins |
| Oct 27 | Mid-semester progress update |
| Nov 9 | Training finished, testing begins |
| Nov 23 | Testing finished, presentation of results |
| Dec 3 | Final report |

## References

- [A machine learning framework for sport result prediction](https://www.researchgate.net/publication/319937079_A_machine_learning_framework_for_sport_result_prediction)
- [XGBoost: A Scalable Tree Boosting System (KDD 2016)](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
- [A Unified Approach to Interpreting Model Predictions (NeurIPS 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
