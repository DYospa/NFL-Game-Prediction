"""Generate the figures used in the NFL game-prediction progress report.

Examples
--------
Run against the repository ZIP file:

    python EDA.py \
        "D:/NFL-Game-Prediction/2010-2026_scores.csv.zip" \
        --output-dir visualizations

The script produces:
    project_map.png
    eda_overview.png
    correlation.png
    results_accuracy.png
    loss_curve.png
    analysis_summary.json
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TEST_SEASON = 2025
ROLLING_WINDOW = 5


def parse_dates(df: pd.DataFrame) -> pd.Series:
    """Convert values such as 'August 8th' into complete calendar dates."""
    month_day = (
        df["GameDate"]
        .astype(str)
        .str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
        .str.strip()
    )
    parsed = pd.to_datetime(
        month_day + " 2000", format="%B %d %Y", errors="coerce"
    )
    calendar_year = np.where(
        parsed.dt.month <= 7, df["Season"] + 1, df["Season"]
    )
    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": calendar_year,
                "month": parsed.dt.month,
                "day": parsed.dt.day,
            }
        ),
        errors="coerce",
    )


def load_and_clean(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the score data and keep completed, non-tied games."""
    if csv_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(csv_path) as archive:
            csv_members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
            ]
            if len(csv_members) != 1:
                raise ValueError(
                    "The ZIP must contain exactly one primary CSV file; found "
                    f"{len(csv_members)}."
                )
            raw = pd.read_csv(io.BytesIO(archive.read(csv_members[0])))
    else:
        raw = pd.read_csv(csv_path)
    games = raw[
        raw["GameStatus"].astype(str).str.contains("Final", case=False, na=False)
    ].copy()
    games = games.dropna(subset=["HomeScore", "AwayScore"])
    games = games[games["HomeScore"] != games["AwayScore"]]
    games["date"] = parse_dates(games)
    games = (
        games.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    games["game_id"] = games.index
    games["home_win"] = (games["HomeScore"] > games["AwayScore"]).astype(int)
    return raw, games


def build_game_table(
    games: pd.DataFrame, window: int = ROLLING_WINDOW
) -> tuple[pd.DataFrame, list[str]]:
    """Create leakage-safe rolling features and one model row per game."""
    team_rows = []
    for side, opponent in (("home", "away"), ("away", "home")):
        team_games = pd.DataFrame(
            {
                "game_id": games["game_id"],
                "date": games["date"],
                "team": games[f"{side.title()}Team"],
                "pts_for": games[f"{side.title()}Score"],
                "pts_against": games[f"{opponent.title()}Score"],
            }
        )
        team_games["win"] = (
            team_games["pts_for"] > team_games["pts_against"]
        ).astype(int)
        team_games["side"] = side
        team_rows.append(team_games)

    team_games = (
        pd.concat(team_rows)
        .sort_values(["team", "date", "game_id"])
        .reset_index(drop=True)
    )
    grouped = team_games.groupby("team")

    for column in ["pts_for", "pts_against", "win"]:
        # shift(1) ensures the game being predicted cannot enter its own features.
        team_games[f"r_{column}"] = grouped[column].transform(
            lambda series: series.shift(1).rolling(window, min_periods=2).mean()
        )

    team_games["games_played"] = grouped.cumcount()
    team_games["rest_days"] = grouped["date"].diff().dt.days.clip(upper=14)

    base_features = [
        "r_pts_for",
        "r_pts_against",
        "r_win",
        "rest_days",
        "games_played",
    ]
    home = (
        team_games[team_games["side"] == "home"]
        .set_index("game_id")[base_features]
        .add_prefix("h_")
    )
    away = (
        team_games[team_games["side"] == "away"]
        .set_index("game_id")[base_features]
        .add_prefix("a_")
    )

    model_games = games.set_index("game_id").join(home).join(away)
    for feature in base_features:
        model_games[f"d_{feature}"] = (
            model_games[f"h_{feature}"] - model_games[f"a_{feature}"]
        )

    features = [
        column
        for column in model_games.columns
        if column.startswith(("h_", "a_", "d_"))
    ]
    return model_games.dropna(subset=features), features


def classification_metrics(y_true, predictions, probabilities) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions)),
        "Recall": float(recall_score(y_true, predictions)),
        "F1": float(f1_score(y_true, predictions)),
        "AUC": float(roc_auc_score(y_true, probabilities)),
    }



def run_analysis(csv_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, clean_games = load_and_clean(csv_path)
    model_games, features = build_game_table(clean_games)

    train = model_games[model_games["Season"] < TEST_SEASON]
    test = model_games[model_games["Season"] == TEST_SEASON]
    if test.empty:
        raise ValueError(f"No games were found for test season {TEST_SEASON}.")

    model_factories = {
        "Logistic regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Gradient-boosted trees": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=300,
            l2_regularization=1.0,
        ),
    }

    results: dict[str, dict] = {}
    cross_validation: dict[str, list[float]] = {}

    for model_name, create_model in model_factories.items():
        seasonal_accuracy = []
        for season in range(2018, 2025):
            cv_train = model_games[model_games["Season"] < season]
            cv_valid = model_games[model_games["Season"] == season]
            model = create_model().fit(cv_train[features], cv_train["home_win"])
            seasonal_accuracy.append(
                float(
                    accuracy_score(
                        cv_valid["home_win"], model.predict(cv_valid[features])
                    )
                )
            )
        cross_validation[model_name] = seasonal_accuracy

        model = create_model().fit(train[features], train["home_win"])
        probabilities = model.predict_proba(test[features])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results[model_name] = classification_metrics(
            test["home_win"], predictions, probabilities
        )
        results[model_name]["Confusion matrix"] = confusion_matrix(
            test["home_win"], predictions
        ).tolist()

    results["Always home"] = {"Accuracy": float(test["home_win"].mean())}


    # Completed games and score-margin distribution.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    season_counts = clean_games.groupby("Season").size()
    axes[0].bar(
        season_counts.index.astype(str), season_counts.values, color="#2b6f8a"
    )
    axes[0].tick_params(axis="x", rotation=60, labelsize=8)
    axes[0].set(title="Completed games by season", ylabel="Games")
    axes[1].hist(
        clean_games["HomeScore"] - clean_games["AwayScore"],
        bins=31,
        color="#d97841",
        edgecolor="white",
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(
        title="Home scoring margin distribution",
        xlabel="Home score - away score",
        ylabel="Games",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=180)
    plt.close(fig)

    # Correlation matrix for matchup-difference features.
    difference_features = [column for column in features if column.startswith("d_")]
    correlation = model_games[["home_win"] + difference_features].corr()
    labels = [
        "Home win",
        "Points for",
        "Points against",
        "Win rate",
        "Rest days",
        "Games played",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(correlation.shape[0]):
        for column in range(correlation.shape[1]):
            ax.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, shrink=0.78)
    ax.set_title("Correlation of difference features")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation.png", dpi=180)
    plt.close(fig)

    # Test accuracy comparison.
    chart_names = ["Always home", "Logistic\nregression", "Gradient-boosted\ntrees"]
    chart_values = [
        results["Always home"]["Accuracy"],
        results["Logistic regression"]["Accuracy"],
        results["Gradient-boosted trees"]["Accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(
        chart_names, chart_values, color=["#8f9aa3", "#2b6f8a", "#d97841"]
    )
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("2025 test accuracy")
    ax.set_title("Initial experiment results")
    for bar, value in zip(bars, chart_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.3f}",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "results_accuracy.png", dpi=180)
    plt.close(fig)

    # Training and validation loss using 2024 as a chronological validation set.
    loss_train = model_games[model_games["Season"] < 2024]
    loss_valid = model_games[model_games["Season"] == 2024]
    iterations = [10, 25, 50, 75, 100, 150, 200, 250, 300]
    training_loss = []
    validation_loss = []

    for iteration_count in iterations:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=iteration_count,
            l2_regularization=1.0,
        ).fit(loss_train[features], loss_train["home_win"])
        training_loss.append(
            float(
                log_loss(
                    loss_train["home_win"],
                    model.predict_proba(loss_train[features])[:, 1],
                )
            )
        )
        validation_loss.append(
            float(
                log_loss(
                    loss_valid["home_win"],
                    model.predict_proba(loss_valid[features])[:, 1],
                )
            )
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        iterations,
        training_loss,
        marker="o",
        label="Training (<2024)",
        color="#2b6f8a",
    )
    ax.plot(
        iterations,
        validation_loss,
        marker="o",
        label="Validation (2024)",
        color="#d97841",
    )
    ax.set(
        xlabel="Boosting iterations",
        ylabel="Log loss",
        title="Gradient-boosted-tree learning curve",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)

    summary = {
        "raw_shape": list(raw.shape),
        "status_counts": raw["GameStatus"].value_counts(dropna=False).to_dict(),
        "missing_values": raw.isna().sum().to_dict(),
        "clean_games": len(clean_games),
        "model_games": len(model_games),
        "train_games": len(train),
        "test_games": len(test),
        "features": features,
        "home_win_rate": float(clean_games["home_win"].mean()),
        "cross_validation": cross_validation,
        "results": results,
        "loss_curve": {
            "iterations": iterations,
            "training": training_loss,
            "validation": validation_loss,
        },
        "correlations": correlation.round(4).to_dict(),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate EDA and model-result visualizations for the NFL project."
    )
    parser.add_argument(
        "csv_path", type=Path, nargs="?",
        default=Path(__file__).resolve().parent / "2010-2026_scores.csv.zip",
        help="Path to the scores CSV or ZIP (default: the bundled ZIP)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations"),
        help="Directory in which to save PNG figures and the JSON summary.",
    )
    args = parser.parse_args()

    summary = run_analysis(args.csv_path, args.output_dir)
    print(f"Generated visualization files in: {args.output_dir.resolve()}")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
