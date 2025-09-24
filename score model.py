# score_model.py
# A sport-agnostic score prediction model (home_score, away_score).
# - Trains on matches.csv if present; else generates synthetic data.
# - Saves model to model.joblib and feature_names.json
# - CLI: train / predict
#
# Example CSV schema (matches.csv):
#   date,home_team,away_team,home_score,away_score,home_elo,away_elo,
#   home_form_avg,away_form_avg,rest_home,rest_away,inj_home,inj_away,
#   is_derby,weather_temp,weather_rain,home_advantage
#
# Only the numeric columns are required for training:
#   home_score, away_score (targets)
#   home_elo, away_elo, home_form_avg, away_form_avg, rest_home, rest_away,
#   inj_home, inj_away, is_derby, weather_temp, weather_rain, home_advantage

import argparse
import json
import os
import sys
import math
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES = [
    "home_elo",
    "away_elo",
    "home_form_avg",
    "away_form_avg",
    "rest_home",
    "rest_away",
    "inj_home",
    "inj_away",
    "is_derby",
    "weather_temp",
    "weather_rain",
    "home_advantage"
]

TARGETS = ["home_score", "away_score"]


def load_or_synthesize(path: str = "matches.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training data if matches.csv exists and is valid; otherwise synthesize a plausible dataset.
    Returns (X, y).
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        missing_cols = [c for c in FEATURES + TARGETS if c not in df.columns]
        if missing_cols:
            print(f"[WARN] CSV found but missing columns: {missing_cols}. Falling back to synthetic data.")
        else:
            # keep numeric only and drop NA targets
            df = df[FEATURES + TARGETS].copy()
            df = df.dropna(subset=TARGETS)
            X = df[FEATURES]
            y = df[TARGETS]
            return X, y

    # --- Synthetic data generator ---
    n = 4000
    home_elo = np.random.normal(1500, 200, size=n).clip(1000, 2200)
    away_elo = np.random.normal(1500, 200, size=n).clip(1000, 2200)

    # recent form ~ last 5 games points normalized [0..1.0] (you can reinterpret per sport)
    home_form = np.random.beta(2.5, 2.0, size=n)  # skewed toward mid-high
    away_form = np.random.beta(2.3, 2.2, size=n)

    rest_home = np.random.randint(2, 8, size=n)     # days since last game
    rest_away = np.random.randint(2, 8, size=n)
    inj_home = np.random.poisson(0.6, size=n)       # "key" injuries
    inj_away = np.random.poisson(0.6, size=n)
    is_derby = np.random.binomial(1, 0.15, size=n)  # rivalry
    weather_temp = np.random.normal(20, 7, size=n)  # °C
    weather_rain = np.random.binomial(1, 0.25, size=n)
    # sport-agnostic home advantage scalar (e.g., 0.0..1.0). Higher -> more edge.
    home_adv = np.random.beta(2, 5, size=n)

    # Latent expected scoring (generic). You can reinterpret per sport:
    #   - football: goals ~ 0..5
    #   - basketball: points scaled up (you can multiply later)
    # Use a smooth function of ELO gap, form, injuries, rest, HFA, weather.
    elo_gap = (home_elo - away_elo) / 400.0  # typical scaling
    form_gap = home_form - away_form
    rest_gap = (rest_home - rest_away) / 7.0
    inj_gap = (inj_away - inj_home) * 0.2
    derby_effect = is_derby * 0.05  # tighter/tenser -> slightly lower totals
    weather_penalty = (weather_rain * 0.15) + np.clip((10 - np.abs(weather_temp - 18)) / 50.0, 0, 0.2)

    # Base expected totals (low scoring bias). Tweak base for your sport.
    base_home = 1.2 + 0.8 * home_adv + 0.7 * elo_gap + 0.6 * form_gap + 0.25 * rest_gap + inj_gap - derby_effect + weather_penalty
    base_away = 1.0 - 0.2 * home_adv - 0.7 * elo_gap - 0.6 * form_gap - 0.20 * rest_gap - inj_gap - derby_effect + (weather_penalty * 0.6)

    # Add noise and clamp to non-negative
    home_score = np.maximum(0, np.random.normal(base_home, 0.8))
    away_score = np.maximum(0, np.random.normal(base_away, 0.8))

    # Convert to "score scale". For football this is fine; for basketball you can multiply later.
    # Keep fractional during training; we’ll round during prediction.
    data = pd.DataFrame({
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_form_avg": home_form,
        "away_form_avg": away_form,
        "rest_home": rest_home,
        "rest_away": rest_away,
        "inj_home": inj_home,
        "inj_away": inj_away,
        "is_derby": is_derby,
        "weather_temp": weather_temp,
        "weather_rain": weather_rain,
        "home_advantage": home_adv,
        "home_score": home_score,
        "away_score": away_score
    })

    X = data[FEATURES]
    y = data[TARGETS]
    return X, y


def build_pipeline() -> Pipeline:
    base = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    # RF doesn’t need scaling, but we include imputation
    model = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("rf", MultiOutputRegressor(base))
    ])
    return model


def train(model_out: str = "model.joblib", features_out: str = "feature_names.json"):
    X, y = load_or_synthesize("matches.csv")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_valid), 0, None)
    mae = mean_absolute_error(y_valid, preds, multioutput="raw_values")
    r2 = r2_score(y_valid, preds, multioutput="variance_weighted")

    # Light CV just to sanity check
    cv_scores = cross_val_score(model, X, y, cv=3, scoring="r2")
    print(f"[TRAIN] Validation MAE (home, away): {mae[0]:.3f}, {mae[1]:.3f}")
    print(f"[TRAIN] Validation R2 (weighted): {r2:.3f}")
    print(f"[TRAIN] 3-fold CV R2: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

    joblib.dump(model, model_out)
    with open(features_out, "w") as f:
        json.dump(FEATURES, f)
    print(f"[TRAIN] Saved model to '{model_out}' and feature list to '{features_out}'.")


def _prompt_float(name: str, default=None) -> float:
    while True:
        s = input(f"{name}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not s and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("  Please enter a number.")


def collect_inputs() -> np.ndarray:
    print("\nEnter features (any sport):")
    print("(Tip: If unsure, just press Enter to accept the suggested default.)\n")
    vals = {
        "home_elo": _prompt_float("home_elo (1000-2200)", 1600),
        "away_elo": _prompt_float("away_elo (1000-2200)", 1500),
        "home_form_avg": _prompt_float("home_form_avg (0.0-1.0, last-5 performance)", 0.6),
        "away_form_avg": _prompt_float("away_form_avg (0.0-1.0)", 0.5),
        "rest_home": _prompt_float("rest_home (days)", 4),
        "rest_away": _prompt_float("rest_away (days)", 3),
        "inj_home": _prompt_float("inj_home (key injuries count)", 0),
        "inj_away": _prompt_float("inj_away (key injuries count)", 1),
        "is_derby": _prompt_float("is_derby (0/1)", 0),
        "weather_temp": _prompt_float("weather_temp (°C)", 22),
        "weather_rain": _prompt_float("weather_rain (0/1)", 0),
        "home_advantage": _prompt_float("home_advantage (0.0-1.0)", 0.4),
    }
    return np.array([[vals[k] for k in FEATURES]]), vals


def _int_score_pair(arr: np.ndarray, sport_scale: float = 1.0) -> Tuple[int, int]:
    """
    Convert continuous outputs to non-negative integer scores.
    Optionally scale up for high-scoring sports (e.g., basketball).
    """
    a = np.maximum(0.0, arr) * float(sport_scale)
    # For high scoring sports you may want to round normally.
    return int(round(a[0])), int(round(a[1]))


def predict_cli(model_path: str = "model.joblib", scale: float = 1.0):
    if not os.path.exists(model_path):
        print("[INFO] No trained model found. Training one now with synthetic or CSV data...")
        train(model_out=model_path)

    model: Pipeline = joblib.load(model_path)
    X, raw = collect_inputs()

    y_pred = model.predict(X)[0]
    y_pred = np.clip(y_pred, 0, None)

    home_score, away_score = _int_score_pair(y_pred, sport_scale=scale)

    print("\n=== Prediction ===")
    print("Inputs used:")
    for k, v in raw.items():
        print(f"  {k}: {v}")
    print(f"\nRaw model output (home, away): {y_pred[0]:.3f}, {y_pred[1]:.3f}")
    if scale != 1.0:
        print(f"(Applied sport scale ×{scale:.1f} for high-scoring sport.)")
    print(f"\nPredicted scoreline: Home {home_score} — Away {away_score}\n")


def main():
    parser = argparse.ArgumentParser(description="Sport-agnostic score prediction model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train on matches.csv (if present) or synthetic data")
    pt.add_argument("--model", default="model.joblib")
    pt.add_argument("--features", default="feature_names.json")

    pp = sub.add_parser("predict", help="Predict a scoreline by entering feature values")
    pp.add_argument("--model", default="model.joblib")
    pp.add_argument("--scale", type=float, default=1.0,
                    help="Scale scores for high-scoring sports (e.g., basketball ~ 40–50).")

    args = parser.parse_args()

    if args.cmd == "train":
        train(model_out=args.model, features_out=args.features)
    elif args.cmd == "predict":
        predict_cli(model_path=args.model, scale=args.scale)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
