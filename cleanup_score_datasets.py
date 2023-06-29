import pandas as pd
from pathlib import Path
import os, shutil
from tqdm import tqdm


def load_lgbm_scores(load_path: str, mode: str, network: str) -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if Path(name).suffix == ".csv":
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, f"Not all biopsies could be loaded for load path {load_path}"
    scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])
    scores["Mode"] = mode
    scores["Network"] = network
    return scores


def load_en_scores(load_path: str, mode: str, network: str) -> pd.DataFrame:
    scores = []
    for root, dirs, files in os.walk(load_path):
        for name in files:
            if Path(name).suffix == ".csv":
                scores.append(pd.read_csv(os.path.join(root, name), sep=",", header=0))

    assert len(scores) == 8, f"Not all biopsies could be loaded for load path {load_path}"
    scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])
    scores["Mode"] = mode
    scores["Network"] = network
    return scores


def prepare_en_scores(save_path: Path):
    print("Preparing elastic net scores...")
    en_path = Path(save_path, "en")

    if en_path.exists():
        shutil.rmtree(en_path)
    en_path.mkdir(parents=True, exist_ok=True)

    scores = []
    scores.append(load_lgbm_scores("data/scores/Mesmer/ip/EN", "IP", "EN"))
    scores.append(load_lgbm_scores("data/scores/Mesmer/exp/EN", "EXP", "EN"))

    scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])
    # remove segmentation, snr, panel columns
    scores = scores.drop(columns=["Segmentation", "SNR", "Panel"])
    scores["FE"] = 0

    # replace _ with '' for biopsy column
    scores["Biopsy"] = scores["Biopsy"].apply(lambda x: x.replace("_", " "))

    scores.to_csv(Path(en_path, "scores.csv"), index=False)


def prepare_lbgm_scores(save_path: Path):
    try:
        print("Preparing light gbm scores...")
        lgbm_path = Path(save_path, "lgbm")

        if lgbm_path.exists():
            shutil.rmtree(lgbm_path)
        lgbm_path.mkdir(parents=True, exist_ok=True)

        microns = [0, 23, 46, 92, 138, 184]
        scores = []
        for micron in tqdm(microns):
            ip_path = f"data/scores/Mesmer/ip/Ludwig{f'_sp_{micron}' if micron != 0 else ''}"
            exp_path = f"data/scores/Mesmer/exp/Ludwig{f'_sp_{micron}' if micron != 0 else ''}"
            scores.append(load_lgbm_scores(ip_path, "IP", "LGBM"))
            scores.append(load_lgbm_scores(exp_path, "EXP", "LGBM"))

        # load hyper scores
        # scores.append(load_lgbm_scores("data/scores/Mesmer/ip/Ludwig_hyper", "IP", "LGBM"))
        # scores.append(load_lgbm_scores("data/scores/Mesmer/exp/Ludwig_hyper", "EXP", "LGBM"))

        scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])
        # replace _ with '' for biopsy column
        scores["Biopsy"] = scores["Biopsy"].apply(lambda x: x.replace("_", " "))

        # convert Hyper Flase to 0
        scores["Hyper"] = scores["Hyper"].apply(lambda x: 0 if x == "False" else 1)
        # convert Hyper column to int
        scores["Hyper"] = scores["Hyper"].apply(lambda x: int(x))

        # Remove Load Path Random Seed,
        scores = scores.drop(columns=["Load Path", "Random Seed"])
        scores.to_csv(Path(lgbm_path, "scores.csv"), index=False)
    except BaseException as ex:
        print(ex)
        print("LGBM scores could not be cleaned up")


def prepare_ae_scores(save_path: Path):
    print("Preparing ae scores")
    ae_path = Path(save_path, "ae")

    if ae_path.exists():
        shutil.rmtree(ae_path)
    ae_path.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
    scores["Mode"] = scores["Mode"].apply(lambda x: x.upper())
    # convert FE column to int
    scores["FE"] = scores["FE"].apply(lambda x: int(x))
    # replace _ with '' for biopsy column
    scores["Biopsy"] = scores["Biopsy"].apply(lambda x: x.replace("_", " "))

    # group by marker, biopsy and experiment, only keep iteration 5-9
    scores = scores.groupby(["Marker", "Biopsy", "Experiment", "Mode", "HP", "FE", "Noise", "Replace Value"]).nth(
        [5, 6, 7, 8, 9]).reset_index()

    # calculate mean of MAE scores
    scores = scores.groupby(["Marker", "Biopsy", "Experiment", "Mode", "HP", "FE", "Noise", "Replace Value"]).mean(
        numeric_only=True).reset_index()

    # remove load path and random seed
    if "Load Path" in scores.columns:
        scores = scores.drop(columns=["Load Path"])
    scores.to_csv(Path(ae_path, "scores.csv"), index=False)


def prepare_gnn_scores(save_path: Path):
    print("Preparing gnn scores")
    gnn_path = Path(save_path, "gnn")

    if gnn_path.exists():
        shutil.rmtree(gnn_path)
    gnn_path.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(Path("data", "scores", "gnn", "scores.csv"))
    # make Mode upper case
    scores["Mode"] = scores["Mode"].apply(lambda x: x.upper())
    # convert FE column to int
    scores["FE"] = scores["FE"].apply(lambda x: int(x))

    # replace _ with '' for biopsy column
    scores["Biopsy"] = scores["Biopsy"].apply(lambda x: x.replace("_", " "))

    # group by marker, biopsy and experiment, only keep iteration 5-9
    scores = scores.groupby(["Marker", "Biopsy", "Experiment", "Mode", "FE", "Noise", "Replace Value"]).nth(
        [5, 6, 7, 8, 9]).reset_index()

    # calculate mean of MAE scores
    scores = scores.groupby(["Marker", "Biopsy", "Experiment", "Mode", "FE", "Noise", "Replace Value"]).mean(
        numeric_only=True).reset_index()

    # remove load path and random seed
    if "Load Path" in scores.columns:
        scores = scores.drop(columns=["Load Path"])

    # remove iteration column and imputation column
    scores = scores.drop(columns=["Iteration", "Imputation"])
    scores.to_csv(Path(gnn_path, "scores.csv"), index=False)


if __name__ == '__main__':

    # create new scores folder
    save_path = Path("data/cleaned_data/scores")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # prepare_en_scores(save_path=save_path)
    prepare_lbgm_scores(save_path=save_path)
    prepare_ae_scores(save_path=save_path)
    prepare_gnn_scores(save_path=save_path)
