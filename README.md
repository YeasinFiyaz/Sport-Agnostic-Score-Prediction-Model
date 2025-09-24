# Sport-Agnostic Score Prediction Model

A Python-based machine learning model that predicts scores for any sport (soccer, basketball, cricket, etc.) using general features like team strength, form, rest days, injuries, home advantage, and weather.

# The model:

Works out of the box with synthetic data (no dataset required).

Can optionally train on your own matches.csv dataset.

Supports custom scaling for high-scoring sports (e.g., basketball, cricket).

Runs via a simple command-line interface.

# Features

Train on your own data (matches.csv) or let it auto-generate synthetic match data.

Predict match outcomes interactively by entering values like team ratings, form, and injuries.

Custom sport scaling — adapt for football (low scoring) or basketball/cricket (high scoring).

Built with scikit-learn and Random Forest regressors.

Portable — no dependencies on heavy frameworks (XGBoost, TensorFlow, etc.).


# Project Flow
├── score_model.py      # Main script (train + predict)
├── matches.csv         # (Optional) Your dataset
├── model.joblib        # Trained model (auto-saved after training)
├── feature_names.json  # Saved feature list
└── README.md           # This file
