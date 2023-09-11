import pandas as pd
from pathlib import Path
import os, shutil
from tqdm import tqdm
from argparse import ArgumentParser

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def load_lgbm_scores(load_path: str, mode: str, network: str) -> pd.DataFrame:
    try:
        scores = []
        for root, dirs, files in os.walk(load_path):
            for name in files:
                if Path(name).suffix == ".csv":
                    file_name = os.path.join(root, name)
                    score = pd.read_csv(file_name, sep=",", header=0)
                    if 'Unnamed: 0' in score.columns:
                        score = score.drop(columns=['Unnamed: 0'])

                    scores.append(score)

        assert len(scores) == 8, f"Not all biopsies could be loaded for load path {load_path}"
        scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])
        scores["Mode"] = mode
        scores["Network"] = network
        return scores

    except Exception as e:
        print(e)


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
    scores["Network"] = "EN"
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
            ip_path = f"data/scores/lgbm/ip/{micron}"
            exp_path = f"data/scores/lgbm/exp/{micron}"
            scores.append(load_lgbm_scores(ip_path, "IP", "LGBM"))
            scores.append(load_lgbm_scores(exp_path, "EXP", "LGBM"))

        scores = pd.concat(scores, axis=0).sort_values(by=["Marker"])

        # replace _ with '' for biopsy column
        scores["Biopsy"] = scores["Biopsy"].apply(lambda x: x.replace("_", " "))

        # convert Hyper False to 0
        # check if hyper column is type int
        if scores["Hyper"].dtype != "int64":
            scores["Hyper"] = scores["Hyper"].apply(lambda x: 0 if x == "False" else 1)
        # convert Hyper column to int
        scores["Hyper"] = scores["Hyper"].apply(lambda x: int(x))

        # Remove Load Path Random Seed,
        scores = scores.drop(columns=["Load Path", "Random Seed"])
        scores["Network"] = "LGBM"

        if "Hyper" in scores.columns:
            # rename hyper column to hp
            scores = scores.rename(columns={"Hyper": "HP"})

        scores.to_csv(Path(lgbm_path, "scores.csv"), index=False)

    except BaseException as ex:
        print(ex)
        print("LGBM scores could not be cleaned up")
        print(scores)


def prepare_ae_scores(save_path: Path, imputation: str = None):
    print(f"Preparing ae scores with imputation {imputation}")

    if imputation is None:
        scores = pd.read_csv(Path("data", "scores", "ae", "scores.csv"))
        network = "AE"
        ae_path = Path(save_path, "ae")
    elif imputation == "multi":
        scores = pd.read_csv(Path("data", "scores", "ae_m", "scores.csv"))
        network = "AE M"
        ae_path = Path(save_path, "ae_m")
    elif imputation == "tma":
        scores = pd.read_csv(Path("data", "scores", "ae_tma", "scores.csv"))
        network = "AE TMA"
        ae_path = Path(save_path, "ae_tma")
    else:
        scores = pd.read_csv(Path("data", "scores", "ae_all", "scores.csv"))
        network = "AE ALL"
        ae_path = Path(save_path, "ae_all")

    if ae_path.exists():
        shutil.rmtree(ae_path)
    ae_path.mkdir(parents=True, exist_ok=True)

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

    # select only scores of shared markers
    scores = scores[scores["Marker"].isin(SHARED_MARKERS)]

    # remove load path and random seed
    if "Load Path" in scores.columns:
        scores = scores.drop(columns=["Load Path"])

    if imputation == "multi":
        # remove round column:
        scores = scores.drop(columns=["Round"])

    # drop imputation & iteration columns
    scores = scores.drop(columns=["Imputation", "Iteration"])
    if "Network" not in scores.columns:
        scores["Network"] = network
    scores.to_csv(Path(ae_path, "scores.csv"), index=False)


def prepare_vae_scores(save_path: Path, imputation: str = None):
    print("Preparing vae scores")
    vae_path = Path(save_path, "vae")

    if imputation is None:
        scores = pd.read_csv(Path("data", "scores", "vae", "scores.csv"))
        network = "VAE"
        vae_path = Path(save_path, "vae")
    elif imputation == "multi":
        scores = pd.read_csv(Path("data", "scores", "vae_m", "scores.csv"))
        network = "VAE M"
        vae_path = Path(save_path, "vae_m")
    else:
        scores = pd.read_csv(Path("data", "scores", "vae_all", "scores.csv"))
        network = "VAE ALL"
        vae_path = Path(save_path, "vae_all")

    if vae_path.exists():
        shutil.rmtree(vae_path)
    vae_path.mkdir(parents=True, exist_ok=True)

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

    # drop imputation & iteration columns
    scores = scores.drop(columns=["Imputation", "Iteration"])
    scores["Network"] = network
    scores.to_csv(Path(vae_path, "scores.csv"), index=False)


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
    scores["Network"] = "GNN"
    scores.to_csv(Path(gnn_path, "scores.csv"), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, required=True,
                        choices=["lgbm", "ae", "gnn", "ae_all", "ae_m", "ae_tma"])
    args = parser.parse_args()

    # create new scores folder
    save_path = Path("data/cleaned_data/scores")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    model = args.model

    # prepare_en_scores(save_path=save_path)

    if model == "lgbm":
        try:
            prepare_lbgm_scores(save_path=save_path)
        except:
            print("Could not prepare lgbm scores")

    elif model == "gnn":
        try:
            prepare_gnn_scores(save_path=save_path)
        except:
            print("Could not prepare gnn scores")

    elif model == "ae":
        try:
            prepare_ae_scores(save_path=save_path)
        except:
            print("Could not prepare ae scores")

    elif model == "ae_all":
        try:
            prepare_ae_scores(save_path=save_path, imputation="all")
        except:
            print("Could not prepare ae scores")

    elif model == "ae_m":
        try:
            prepare_ae_scores(save_path=save_path, imputation="multi")
        except:
            print("Could not prepare ae multi scores")

    elif model == "ae_tma":
        try:
            prepare_ae_scores(save_path=save_path, imputation="tma")
        except:
            print("Could not prepare ae tma scores")
