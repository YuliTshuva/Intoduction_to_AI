"""
Yuli Tshuva
Plot the model and get results
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd
import random
from constants import *
from os.path import join
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import warnings
import shap

warnings.filterwarnings("ignore")

# Set font
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 15

# Pick colors
colors = ["deepskyblue", "turquoise", "hotpink"]

# Read data
X_test, y_test = pd.read_csv(join(DATA_DIR, "X_test.csv")), pd.read_csv(join(DATA_DIR, "y_test.csv"))

# Load models
with open(join(MODELS_DIR, "xgb_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)
with open(join(MODELS_DIR, "lgbm_model.pkl"), "rb") as f:
    lgbm_model = pickle.load(f)
with open(join(MODELS_DIR, "catboost_model.pkl"), "rb") as f:
    catboost_model = pickle.load(f)
models = {"XGBoost": xgb_model, "LightGBM": lgbm_model, "CatBoost": catboost_model}

### Plot results
fig, ax = plt.subplots(1, 2, figsize=(13, 6))

# Plot a bar plot of the accuracy of the models
accuracies = {model_name: accuracy_score(y_test, model.predict(X_test)) for model_name, model in models.items()}
ax[0].bar(list(accuracies.keys()), list(accuracies.values()),
          color=colors,
          edgecolor="black")
ax[0].set_title("Accuracy of the models", fontsize=22)
ax[0].set_ylabel("Accuracy", fontsize=18)
ax[0].set_xlabel("Model", fontsize=18)
ax[0].set_yticks(np.arange(0, 1.1, 0.1))

# Plot a ROC curve of the models
i = 0
for model_name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict(X_test))
    ax[1].plot(fpr, tpr, label=f"{model_name}: {auc:.3f}", color=colors[i])
    i += 1
ax[1].set_title("ROC curve of the models", fontsize=22)
ax[1].set_ylabel("True Positive Rate", fontsize=18)
ax[1].set_xlabel("False Positive Rate", fontsize=18)
ax[1].legend()

# Final adjustments
plt.tight_layout()
plt.savefig(join(PLOTS_DIR, "results.png"))
plt.show()

# Plot the feature importance of the models
fig, ax = plt.subplots(1, 3, figsize=(16, 10))
fig.suptitle("Feature Importance of the models", fontsize=35, y=0.96)
i = 0
for model_name, model in models.items():
    feature_importances = model.feature_importances_
    feature_importances = 100 * feature_importances / feature_importances.sum()
    sorted_idx = np.argsort(feature_importances)[-25:]
    pos = np.arange(sorted_idx.shape[0]) + .5
    ax[i].barh(pos, feature_importances[sorted_idx], align='center',
               color=colors[i], edgecolor="black")
    ax[i].set_yticks(pos)
    ax[i].set_yticklabels(np.array(X_test.columns)[sorted_idx], fontsize=15)
    ax[i].set_xlabel('Relative Importance', fontsize=19)
    ax[i].set_title(f'{model_name}', fontsize=24)
    i += 1

# Final adjustments
plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
plt.savefig(join(PLOTS_DIR, "feature_importance.png"))
plt.show()

# Create the explainer and compute Shapley values
explainer = shap.Explainer(xgb_model, X_test)
shap_values = explainer(X_test)

# Plot the Shapley values with a title
shap.plots.beeswarm(shap_values, max_display=25, show=False)
plt.title('Shapley Values for XGBoost Model')
plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
plt.savefig(join(PLOTS_DIR, "xgb_shapley_values.png"))
plt.show()
