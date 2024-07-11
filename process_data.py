"""
Yuli Tshuva
Processing data for the project
"""

# Imports
from utils import one_hot_encode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from os.path import join

# Set fonts
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 15

# Set constants
DATA_DIR = "data"
DATA_PATH = join(DATA_DIR, "mushrooms.csv")
COLORS = ["dodgerblue", "hotpink", "salmon", "royalblue", "aquamarine", "turquoise", "mediumturquoise",
          "deepskyblue", "violet", "mediumslateblue"]
PLOTS_DIR = "plots"

# load the data
df = pd.read_csv(DATA_PATH)

# Plot data distribution
unique_values = [np.unique(df[col]).shape[0] for col in df.columns]
bars = [unique_values.count(i) for i in range(1, max(unique_values) + 1)]
plt.bar(list(range(1, len(bars) + 1)), bars, color='aquamarine', edgecolor="black")
plt.xticks(list(range(1, len(bars) + 1)))
plt.title("Unique Values Distribution", fontsize=18)
plt.xlabel("Amount of Unique Values")
plt.ylabel("Amount of Columns")
plt.savefig(join(PLOTS_DIR, "unique_values_distribution.png"), dpi=400)
plt.show()

# Split data
X, y = df.drop("class", axis=1), df["class"].apply(lambda x: 1 if x == "e" else 0)

# Subtract the column with one unique value and one-hot encode the rest
for col in X.columns:
    if len(X[col].unique()) == 1:
        X.drop(col, axis=1, inplace=True)
    else:
        X = one_hot_encode(X, col)

# Save the processed data
X.to_csv(join(DATA_DIR, "X.csv"), index=False)
y.to_csv(join(DATA_DIR, "y.csv"), index=False)



