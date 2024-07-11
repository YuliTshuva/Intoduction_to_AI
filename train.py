"""
Yuli Tshuva
Train the model
"""

from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_xgb_model, get_lgbm_model, get_catboost_model
import pickle
from constants import *
import warnings
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

# Check validity of sizes
if TRAIN_SIZE + VAL_SIZE + TEST_SIZE != 1:
    raise ValueError("TRAIN_SIZE + VAL_SIZE + TEST_SIZE must equal 1")

# Load data
X, y = pd.read_csv(join(DATA_DIR, "X.csv")), pd.read_csv(join(DATA_DIR, "y.csv"))

# Split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                  test_size=VAL_SIZE / (1 - TEST_SIZE),
                                                  random_state=SEED)

# Save the sets
X_train_full.to_csv(join(DATA_DIR, "X_train.csv"), index=False)
X_test.to_csv(join(DATA_DIR, "X_test.csv"), index=False)
y_train_full.to_csv(join(DATA_DIR, "y_train.csv"), index=False)
y_test.to_csv(join(DATA_DIR, "y_test.csv"), index=False)

# Train models
xgb_model = get_xgb_model(X_train_full, y_train_full, X_train, X_val, y_train, y_val, trials=N_TRIALS)
lgbm_model = get_lgbm_model(X_train_full, y_train_full, X_train, X_val, y_train, y_val, trials=N_TRIALS)
catboost_model = get_catboost_model(X_train_full, y_train_full, X_train, X_val, y_train, y_val, trials=N_TRIALS)

# Save models
with open(join(MODELS_DIR, "xgb_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
with open(join(MODELS_DIR, "lgbm_model.pkl"), "wb") as f:
    pickle.dump(lgbm_model, f)
with open(join(MODELS_DIR, "catboost_model.pkl"), "wb") as f:
    pickle.dump(catboost_model, f)
