
import pandas as pd
from pathlib import Path

scores = pd.read_csv(Path("data/scores/ae/scores.csv"))


# group by type, replace value, noise and radius

scores = scores.groupby(["FE", "Replace Value", "Noise", "Experiment", "Biopsy", "Marker", "Iteration"]).mean().reset_index()
print(scores.nunique())
input()

#print(scores[scores["Marker"] == "pRB"])

# select no noise and mean replace value
scores = scores[(scores["Noise"] == "no_noise") & (scores["Replace Value"] == "mean")]

print(scores[scores["Marker"] == "pRB"])

# select best MAE score for each experiment from all iterations
scores = scores.loc[scores.groupby(["FE", "Experiment", "Biopsy", "Marker"])["MAE"].idxmin()]

print(scores[scores["Marker"] == "pRB"])
print(scores.value_counts())