import pandas as pd
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    lgbm_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "lgbm", "scores.csv"))
    lgbm_scores = lgbm_scores[lgbm_scores["FE"] == 0]
    # select only non hp scores
    lgbm_scores = lgbm_scores[lgbm_scores["HP"] == 0]
    # replace EXP WITH AP
    lgbm_scores["Mode"] = lgbm_scores["Mode"].replace({"EXP": "AP"})

    en_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "en", "scores.csv"))
    en_scores = en_scores[en_scores["FE"] == 0]
    # replace EXP WITH AP
    en_scores["Mode"] = en_scores["Mode"].replace({"EXP": "AP"})

    ae_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae", "scores.csv"))
    ae_m_scores = pd.read_csv(Path("data", "cleaned_data", "scores", "ae_m", "scores.csv"))
    # replace EXP WITH AP
    ae_scores["Mode"] = ae_scores["Mode"].replace({"EXP": "AP"})

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_scores = ae_scores[
        (ae_scores["FE"] == 0) & (ae_scores["Replace Value"] == "mean") & (ae_scores["Noise"] == 0)]
    # select only non hp scores
    ae_scores = ae_scores[ae_scores["HP"] == 0]
    ae_scores.sort_values(by=["Marker"], inplace=True)

    # Select ae scores where fe  == 0, replace value == mean and noise  == 0
    ae_m_scores = ae_m_scores[
        (ae_m_scores["FE"] == 0) & (ae_m_scores["Replace Value"] == "mean") & (ae_m_scores["Noise"] == 0)]
    # select only non hp scores
    ae_m_scores = ae_m_scores[ae_m_scores["HP"] == 0]
    ae_m_scores.sort_values(by=["Marker"], inplace=True)
    # replace EXP WITH AP
    ae_m_scores["Mode"] = ae_m_scores["Mode"].replace({"EXP": "AP"})

    # merge all scores together
    all_scores = pd.concat([lgbm_scores, en_scores, ae_scores, ae_m_scores], axis=0)

    print("Mean and std of MAE scores per network")
    exp_scores = all_scores[all_scores["Mode"] == "AP"]
    ip_scores = all_scores[all_scores["Mode"] == "IP"]
    print("EXP scores")
    print(exp_scores.groupby(["Network"])["MAE"].agg(["mean", "std"]))

    print("IP scores")
    print(ip_scores.groupby(["Network"])["MAE"].agg(["mean", "std"]))
