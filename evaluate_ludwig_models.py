import os, argparse
from pathlib import Path
from ludwig.api import LudwigModel
import pandas as pd

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def create_scores_dir(combination: str, radius) -> Path:
    scores_directory = Path("data/scores/Mesmer")
    scores_directory = Path(scores_directory, combination)

    if radius is not None:
        scores_directory = Path(scores_directory, f"Ludwig_sp_{radius}")
    else:
        scores_directory = Path(scores_directory, f"Ludwig")

    scores_directory = Path(scores_directory)

    if not scores_directory.exists():
        scores_directory.mkdir(parents=True, exist_ok=True)

    return scores_directory


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--biopsy", type=str, required=True, help="path to biopsy")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="path to dataset")
    parser.add_argument('-r', '--radius', type=str, required=False, default=None, help="The radius")
    args = parser.parse_args()

    radius: str = args.radius

    biopsy_path: str = args.biopsy
    dataset: str = args.dataset
    test_biopsy_name = "_".join(Path(dataset).stem.split('_')[:3])

    scores = []

    if 'in_patient' in str(biopsy_path):
        combination = 'IP'
    elif 'out_patient' in str(biopsy_path):
        combination = 'OP'
    elif '_6_2' in str(biopsy_path) or '_exp' in str(biopsy_path):
        combination = 'EXP'
    else:
        raise ValueError("Unknown mode")

    if combination == "IP" and test_biopsy_name == Path(biopsy_path).stem:
        raise ValueError("Train and test biopsy are the same")

    save_path = create_scores_dir(combination=combination, radius=radius)

    for marker in SHARED_MARKERS:
        results_path = Path(biopsy_path, marker, "results")
        for root, marker_sub_directories, files in os.walk(str(results_path)):
            if "experiment_run" in marker_sub_directories:
                for experiment in marker_sub_directories:
                    models = None
                    try:
                        model = LudwigModel.load(str(Path(results_path, experiment, 'model')))
                    except:
                        continue
                    eval_stats, _, _ = model.evaluate(dataset=dataset)

                    # Marker,MAE,MSE,RMSE,Biopsy,Panel,Type,Segmentation,SNR,FE,Mode,Hyper
                    print(eval_stats)

                    scores.append(
                        {
                            "Marker": marker,
                            "MAE": eval_stats[marker]['mean_absolute_error'],
                            "MSE": eval_stats[marker]['mean_squared_error'],
                            "RMSE": eval_stats[marker]['root_mean_squared_error'],
                            "Biopsy": test_biopsy_name,
                            "Combination": combination,
                            "FE": radius,
                            "Mode": "Ludwig",
                            "Hyper": 0
                        }
                    )

    scores = pd.DataFrame(scores)
    print(scores)
    scores.to_csv(Path(save_path, f"{test_biopsy_name}_scores.csv"), index=False)
