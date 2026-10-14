"""
NFL game prediction - starter pipeline.

STEP 0: run `python nfl_pipeline.py --inspect` to see your CSV's columns,
then edit the COLS mapping below to match. Everything else keys off COLS.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "2010-2026_scores.csv")  # <- your downloaded Kaggle file
TEST_SEASON = 2025
WINDOW = 5                   # rolling window (games)

# ---- EDIT THESE to match your CSV's column names ----
COLS = {
    "season": "Season",
    "date": "GameDate",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "home_score": "HomeScore",
    "away_score": "AwayScore",
    "stats": {},
    "home_spread": None,
}

def parse_dates(df):
    # "August 8th" -> "August 8", then attach a year based on the season
    s = (df[COLS["date"]].astype(str)
         .str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)
         .str.strip())
    md = pd.to_datetime(s + " 2000", format="%B %d %Y", errors="coerce")  # 2000 is a leap year placeholder
    year = np.where(md.dt.month <= 7, df[COLS["season"]] + 1, df[COLS["season"]])
    return pd.to_datetime(pd.DataFrame({"year": year, "month": md.dt.month, "day": md.dt.day}),
                          errors="coerce")


def load():
    df = pd.read_csv(CSV_PATH)
    print(df["GameStatus"].value_counts())            # check the real status values
    df = df[df["GameStatus"].astype(str).str.contains("Final", case=False, na=False)]
    df = df.dropna(subset=["HomeScore", "AwayScore"])
    df = df[df["HomeScore"] != df["AwayScore"]]       # drop ties
    df[COLS["date"]] = parse_dates(df)

    bad = df[df[COLS["date"]].isna()]
    if len(bad):
        print(f"WARNING: {len(bad)} rows with unparseable dates, e.g.:")
        print(bad[[COLS["season"], COLS["date"]]].head())
        df = df.dropna(subset=[COLS["date"]])

    df = df.sort_values(COLS["date"]).reset_index(drop=True)
    df["game_id"] = df.index
    df["home_win"] = (df[COLS["home_score"]] > df[COLS["away_score"]]).astype(int)
    return df


def to_team_games(df):
    """One row per team per game, with that team's 'for' and 'against' stats."""
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        t = pd.DataFrame({
            "game_id": df["game_id"],
            "date": df[COLS["date"]],
            "team": df[COLS[f"{side}_team"]],
            "pts_for": df[COLS[f"{side}_score"]],
            "pts_against": df[COLS[f"{other}_score"]],
        })
        t["win"] = (t["pts_for"] > t["pts_against"]).astype(int)
        for name, (h, a) in COLS["stats"].items():
            mine, theirs = (h, a) if side == "home" else (a, h)
            t[f"{name}_for"] = df[mine]
            t[f"{name}_against"] = df[theirs]
        t["side"] = side
        rows.append(t)
    return pd.concat(rows).sort_values(["team", "date", "game_id"]).reset_index(drop=True)


def add_rolling(tg):
    """Rolling means over the previous WINDOW games. shift(1) => NO leakage."""
    feat_cols = [c for c in tg.columns
                 if c in ("pts_for", "pts_against", "win") or c.endswith(("_for", "_against"))]
    g = tg.groupby("team")
    for c in feat_cols:
        tg[f"r_{c}"] = g[c].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=2).mean())
    tg["games_played"] = g.cumcount()
    tg["rest_days"] = g["date"].diff().dt.days.clip(upper=14)
    return tg


def build_game_table(df):
    tg = add_rolling(to_team_games(df))
    keep = [c for c in tg.columns if c.startswith("r_")] + ["rest_days", "games_played"]
    home = tg[tg.side == "home"].set_index("game_id")[keep].add_prefix("h_")
    away = tg[tg.side == "away"].set_index("game_id")[keep].add_prefix("a_")
    g = df.set_index("game_id").join(home).join(away)
    for c in keep:                       # difference features
        g[f"d_{c}"] = g[f"h_{c}"] - g[f"a_{c}"]
    g["home_field"] = 1                  # placeholder; add neutral-site flag if available
    feats = [c for c in g.columns if c.startswith(("d_", "h_", "a_"))]
    g = g.dropna(subset=feats)           # drops very early games with no history
    return g, feats


def metrics(y, pred, proba):
    return {
        "acc": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, proba),
    }


def season_cv(model_fn, g, feats, first_val=2018, last_val=2024):
    """Expanding-window CV by season: train on < s, validate on s."""
    scores = []
    for s in range(first_val, last_val + 1):
        tr, va = g[g[COLS["season"]] < s], g[g[COLS["season"]] == s]
        m = model_fn().fit(tr[feats], tr["home_win"])
        scores.append(accuracy_score(va["home_win"], m.predict(va[feats])))
    return np.mean(scores)


def main():
    if "--inspect" in sys.argv:
        d = pd.read_csv(CSV_PATH)
        print(d.shape); print(d.dtypes); print(d.head())
        print(d[COLS["season"]].value_counts().sort_index() if COLS["season"] in d else "")
        return

    df = load()
    g, feats = build_game_table(df)
    train = g[g[COLS["season"]] < TEST_SEASON]
    test = g[g[COLS["season"]] == TEST_SEASON]
    print(f"train games: {len(train)}, test games: {len(test)}, features: {len(feats)}")

    # Benchmark 1: always pick home
    print("Always-home accuracy:", accuracy_score(test["home_win"], np.ones(len(test))))
    # Benchmark 2: Vegas favorite (if spread available)
    if COLS["home_spread"]:
        vegas = (test[COLS["home_spread"]] > 0).astype(int)
        print("Vegas favorite accuracy:", accuracy_score(test["home_win"], vegas))

    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "gbt": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=300, l2_regularization=1.0),
        # swap in xgboost.XGBClassifier / lightgbm.LGBMClassifier later if you like
    }

    fitted = {}
    for name, fn in models.items():
        print(f"\n{name}: CV accuracy (2018-2024) = {season_cv(fn, g[g[COLS['season']] < TEST_SEASON], feats):.3f}")
        m = fn().fit(train[feats], train["home_win"])
        fitted[name] = m
        proba = m.predict_proba(test[feats])[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"{name} TEST:", {k: round(v, 3) for k, v in metrics(test["home_win"], pred, proba).items()})
        print(confusion_matrix(test["home_win"], pred))

    # SHAP (pip install shap) - run only after final model is chosen
    try:
        import shap
        expl = shap.Explainer(fitted["gbt"].predict, train[feats].sample(min(500, len(train)), random_state=0))
        sv = expl(test[feats].iloc[:200])
        shap.plots.beeswarm(sv)
    except ImportError:
        print("Install shap to produce feature-importance plots.")


if __name__ == "__main__":
    main()