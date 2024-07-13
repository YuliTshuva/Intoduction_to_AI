"""
Yuli Tshuva
Find the minimal columns needed for the model
"""

from os.path import join
import pandas as pd
import pickle
from constants import *
import warnings
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import random
from tqdm.auto import tqdm
from matplotlib import rcParams

# Set fonts
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 14

warnings.filterwarnings("ignore")

# Check validity of sizes
if TRAIN_SIZE + VAL_SIZE + TEST_SIZE != 1:
    raise ValueError("TRAIN_SIZE + VAL_SIZE + TEST_SIZE must equal 1")

# Load the sets
X_train_full = pd.read_csv(join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(join(DATA_DIR, "X_test.csv"))
y_train_full = pd.read_csv(join(DATA_DIR, "y_train.csv"))
y_test = pd.read_csv(join(DATA_DIR, "y_test.csv"))

# Load model
with open(join(MODELS_DIR, "catboost_model.pkl"), "rb") as f:
    catboost_model = pickle.load(f)

# Get the best hyperparameters found
model_params = catboost_model.get_all_params()
lst = ["iterations", "learning_rate", "depth", "subsample", "colsample_bylevel", "min_data_in_leaf", 'l2_leaf_reg',
       'bagging_temperature', 'random_strength', "silent"]
best_params = {el: model_params[el] for el in lst if el in model_params}

# Find feature importance
feature_importance = catboost_model.get_feature_importance()
feature_importance = pd.DataFrame(feature_importance, index=X_train_full.columns, columns=["importance"])
feature_importance = feature_importance.sort_values(by="importance", ascending=False)
sorted_features = list(feature_importance.index)

# Plot the model performance as a function of the number of features
scores = []
count = 0
for i in tqdm(range(1, len(sorted_features) + 1), total=len(sorted_features)):
    count += 1

    # Get the i most important features
    X_train_i = X_train_full[sorted_features[:i]]
    X_test_i = X_test[sorted_features[:i]]

    # Set initial model
    model = CatBoostClassifier(**best_params, silent=True)

    # Fit the model
    model.fit(X_train_i, y_train_full)

    # Predict using the model
    preds = model.predict(X_test_i)

    # Get the score
    score = accuracy_score(y_test, preds)

    # Append the score
    scores.append(score)

    if score == 1:
        break

# Plot the scores
plt.plot(range(1, count + 1), scores, color=random.sample(COLORS, 1)[0])
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.title("Model performance as a function\nof the number of features", fontsize=17)
plt.xticks(range(1, count + 1))
plt.savefig(join(PLOTS_DIR, "model_performance_vs_features.png"))
plt.show()
