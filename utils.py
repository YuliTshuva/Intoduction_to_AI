"""
Yuli Tshuva
Utils functions for the project
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from sklearn.metrics import accuracy_score
from constants import N_JOBS


def one_hot_encode(data, col):
    """
    One-hot encodes a column in a DataFrame.
    """
    data = pd.concat([data, pd.get_dummies(data[col], prefix=col, dtype=np.int8, dummy_na=True)], axis=1)
    data.drop(col, axis=1, inplace=True)
    return data


def get_xgb_model(X, y, X_train, X_val, y_train, y_val, trials=1000):
    """
    Returns an XGBoost trained model with optimal HP.
    """

    def objective(trial):
        param = {
            "silent": 1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            'n_jobs': N_JOBS
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    best_params = study.best_params
    model = XGBClassifier(**best_params)
    model.fit(X, y)

    return model


def get_lgbm_model(X, y, X_train, X_val, y_train, y_val, trials=1000):
    """
    Returns an XGBoost trained model with optimal HP.
    """

    def objective(trial):
        # Specify a search space using distributions across plausible values of hyperparameters.
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            'n_jobs': N_JOBS
        }

        model = LGBMClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    best_params = study.best_params
    model = LGBMClassifier(**best_params)
    model.fit(X, y)

    return model


def get_catboost_model(X, y, X_train, X_val, y_train, y_val, trials=1000):
    """
    Returns an XGBoost trained model with optimal HP.
    """

    def objective(trial):
        # Specify a search space using distributions across plausible values of hyperparameters.
        param = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 100),
            'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.1, 20.0),
            'random_strength': trial.suggest_float('random_strength', 1.0, 2.0),
            'n_jobs': N_JOBS
        }

        model = CatBoostClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    best_params = study.best_params
    model = CatBoostClassifier(**best_params)
    model.fit(X, y)

    return model
