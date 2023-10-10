import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

en_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "en", "scores.csv"))
en_scores = en_scores[en_scores["FE"] == 0]
# rename EXP to AP
en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})
# sort by marker
en_scores.sort_values(by=["Marker"], inplace=True)


std = en_scores.groupby(["Marker", "Mode", "Biopsy"]).std().reset_index()
mean = en_scores.groupby(["Marker", "Mode", "Biopsy"]).mean().reset_index()

en_std = std.groupby(["Mode", "Biopsy"]).std().reset_index()
en_std["Model"] = "EN"

ae_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae", "scores.csv"))
# Select ae scores where fe  == 0, replace value == mean and noise  == 0
ae_scores = ae_scores[(ae_scores["FE"] == 0) & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0)]
# select only non hp scores
ae_scores = ae_scores[ae_scores["HP"] == 0]
ae_scores.sort_values(by=["Marker"], inplace=True)
ae_scores = ae_scores[np.abs(ae_scores["MAE"] - ae_scores["MAE"].mean()) <= (3 * ae_scores["MAE"].std())]

ae_scores["Mode"] = ae_scores["Mode"].replace({"EXP": "AP"})

std = ae_scores.groupby(["Marker", "Mode", "Biopsy"]).std().reset_index()
ae_std = std.groupby(["Mode", "Biopsy"]).std().reset_index()
ae_std["Model"] = "AE"

lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
# select only non hp scores
lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
# rename lgbm scores EXP TO AP
lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})
# sort by marker
lgbm_scores.sort_values(by=["Marker"], inplace=True)

std = lgbm_scores.groupby(["Marker", "Mode", "Biopsy"]).std().reset_index()
lgbm_std = std.groupby(["Mode", "Biopsy"]).std().reset_index()

lgbm_std["Model"] = "LGBM"

ae_m_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae_m", "scores.csv"))
# Select ae scores where fe  == 0, replace value == mean and noise  == 0
ae_m_scores = ae_m_scores[
    (ae_m_scores["FE"] == 0) & (ae_m_scores["Replace Value"] == "mean") & (ae_m_scores["Noise"] == 0)]
# select only non hp scores
ae_m_scores = ae_m_scores[ae_m_scores["HP"] == 0]
ae_m_scores.sort_values(by=["Marker"], inplace=True)
# replace EXP WITH AP
ae_m_scores["Mode"] = ae_m_scores["Mode"].replace({"EXP": "AP"})

std = ae_m_scores.groupby(["Marker", "Mode", "Biopsy"]).std().reset_index()
ae_m_std = std.groupby(["Mode", "Biopsy"]).std().reset_index()
ae_m_std["Model"] = "AE M"

std = pd.concat([lgbm_std, ae_std, en_std, ae_m_std], axis=0)
std = std[["Model", "Mode", "MAE", "Biopsy"]]
print(std)

# create boxen plot
ax = sns.boxenplot(data=std, x="Mode", y="MAE", hue="Mode", hue_order=["IP", "AP"])
ax.set_title("Standard Deviation of MAE Scores")
plt.show()
