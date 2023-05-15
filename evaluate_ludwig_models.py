import os, argparse
from pathlib import Path
from ludwig.api import LudwigModel
import pandas as pd

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def create_scores_dir(combination: str, radius: int) -> Path:
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
    parser.add_argument('-b', "--biopsy", type=str, required=True,
                        help="The biopsy. For an ip mode that is the train biopsy, for exp mode that is the test biopsy due to ludwigs setup")
    parser.add_argument('-sp', '--spatial', type=int, required=False, default=None, help="The radius",
                        choices=[23, 46, 92, 138, 184])
    parser.add_argument('--mode', type=str, choices=['ip', 'exp'], help="The mode", default='ip')
    args = parser.parse_args()

    spatial_radius: int = args.spatial
    mode = args.mode
    biopsy: str = args.biopsy

    if mode == "ip":
        # change last number of biopsy to 1 if it is 2
        if biopsy[-1] == "2":
            test_biopsy_name = biopsy[:-1] + "1"
        else:
            test_biopsy_name = biopsy[:-1] + "2"

        print(biopsy)
        print(test_biopsy_name)
        assert test_biopsy_name[-1] != biopsy[-1], "The bx should not be the same"
        if spatial_radius is None:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", "tumor_mesmer", "preprocessed", f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')
            base_path = Path("mesmer", "tumor_in_patient", biopsy)
        else:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", f"tumor_mesmer_sp_{spatial_radius}", "preprocessed",
                     f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')
            base_path = Path("mesmer", f"tumor_in_patient_sp_{spatial_radius}", biopsy)

    else:
        test_biopsy_name = biopsy
        assert test_biopsy_name == biopsy, "The bx should be the same"
        print(test_biopsy_name)

        if spatial_radius is None:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", "tumor_mesmer", "preprocessed", f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')
            base_path = Path("mesmer", "tumor_exp_patient", biopsy)
        else:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", f"tumor_mesmer_sp_{spatial_radius}", "preprocessed",
                     f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')
            base_path = Path("mesmer", f"tumor_exp_patient_sp_{spatial_radius}", biopsy)

    scores = []

    save_path = create_scores_dir(combination=mode, radius=spatial_radius)

    for marker in SHARED_MARKERS:
        results_path = Path(base_path, marker, "results")
        for root, marker_sub_directories, files in os.walk(str(results_path)):
            if "experiment_run" in marker_sub_directories:
                for experiment in marker_sub_directories:
                    models = None
                    try:
                        model = LudwigModel.load(str(Path(results_path, experiment, 'model')))
                    except:
                        continue
                    eval_stats, _, _ = model.evaluate(dataset=test_dataset)

                    # Marker,MAE,MSE,RMSE,Biopsy,Panel,Type,Segmentation,SNR,FE,Mode,Hyper
                    print(eval_stats)

                    scores.append(
                        {
                            "Marker": marker,
                            "MAE": eval_stats[marker]['mean_absolute_error'],
                            "MSE": eval_stats[marker]['mean_squared_error'],
                            "RMSE": eval_stats[marker]['root_mean_squared_error'],
                            "Biopsy": test_biopsy_name,
                            "Combination": mode,
                            "FE": spatial_radius,
                            "Mode": "Ludwig",
                            "Hyper": 0
                        }
                    )

    scores = pd.DataFrame(scores)
    print(scores)
    scores.to_csv(Path(save_path, f"{test_biopsy_name}_scores.csv"), index=False)
