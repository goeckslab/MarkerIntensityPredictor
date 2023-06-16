import os, argparse
from pathlib import Path
from ludwig.api import LudwigModel
import pandas as pd
import random
from tqdm import tqdm
import sys

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def create_scores_dir(combination: str, radius: int, hyper: bool) -> Path:
    scores_directory = Path("data/scores/Mesmer")
    scores_directory = Path(scores_directory, combination)

    if radius is not None:
        scores_directory = Path(scores_directory, f"Ludwig_sp_{radius}")
    elif hyper:
        scores_directory = Path(scores_directory, f"Ludwig_hyper")
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
    parser.add_argument('--hyper', action="store_true", help="Use hyperopt", default=False)
    parser.add_argument("--iterations", "-i", type=int, default=2, help="The number of iterations")
    args = parser.parse_args()

    spatial_radius: int = args.spatial
    mode = args.mode
    biopsy: str = args.biopsy
    hyper: bool = args.hyper
    iterations: int = args.iterations

    print(f"Mode: {mode}")
    print(f"Biopsy: {biopsy}")
    print(f"Radius: {spatial_radius}")
    print(f"Hyper: {hyper}")
    print(f"Iterations: {iterations}")

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
            if not hyper:
                base_path = Path("mesmer", "tumor_in_patient", biopsy)
            else:
                base_path = Path("mesmer", "tumor_in_patient_hyper", biopsy)


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
            if not hyper:
                base_path = Path("mesmer", "tumor_exp_patient", biopsy)
            else:
                base_path = Path("mesmer", "tumor_exp_patient_hyper", biopsy)
        else:
            test_dataset: pd.DataFrame = pd.read_csv(
                Path("data", f"tumor_mesmer_sp_{spatial_radius}", "preprocessed",
                     f"{test_biopsy_name}_preprocessed_dataset.tsv"), sep='\t')
            base_path = Path("mesmer", f"tumor_exp_patient_sp_{spatial_radius}", biopsy)

    print(f"Base path: {base_path}")
    scores = []

    save_path = create_scores_dir(combination=mode, radius=spatial_radius, hyper=hyper)
    file_name = f"{test_biopsy_name}_scores.csv"
    print("Save path: ", save_path)
    print("File name: ", file_name)

    for marker in SHARED_MARKERS:
        results_path = Path(base_path, marker, "results")
        for root, marker_sub_directories, files in os.walk(str(results_path)):
            if "experiment_run" in marker_sub_directories:
                for experiment in marker_sub_directories:
                    models = None
                    try:
                        model = LudwigModel.load(str(Path(results_path, experiment, 'model')))
                    except KeyboardInterrupt as ex:
                        print("Keyboard interrupt")
                        sys.exit(0)
                    except BaseException as ex:
                        print(ex)
                        continue

                    for i in tqdm(range(1, iterations)):

                        random_seed = random.randint(0, 100000)
                        # sample new dataset from test_data
                        test_data_sample = test_dataset.sample(frac=0.7, random_state=random_seed,
                                                               replace=True)
                        test_data_sample.reset_index(drop=True, inplace=True)
                        try:
                            eval_stats, _, _ = model.evaluate(dataset=test_data_sample)
                        except KeyboardInterrupt as ex:
                            print("Keyboard interrupt")
                            sys.exit(0)
                        except BaseException as ex:
                            continue
                        scores.append(
                            {
                                "Marker": marker,
                                "MAE": eval_stats[marker]['mean_absolute_error'],
                                "MSE": eval_stats[marker]['mean_squared_error'],
                                "RMSE": eval_stats[marker]['root_mean_squared_error'],
                                "Biopsy": test_biopsy_name,
                                "Mode": mode,
                                "FE": spatial_radius if spatial_radius is not None else 0,
                                "Network": "Ludwig",
                                "Hyper": hyper,
                                "Load Path": str(Path(results_path, experiment, 'model')),
                                "Random Seed": random_seed,
                            }
                        )

                        if i % 2 == 0:
                            scores = pd.DataFrame(scores)
                            if Path(save_path, file_name).exists():
                                print(f"Temp saving scores for marker {marker}...")
                                print("Found existing scores...")
                                print("Merging...")
                                temp_scores = pd.read_csv(Path(save_path, file_name))
                                scores = pd.concat([temp_scores, scores], ignore_index=True)

                            scores.to_csv(Path(save_path, file_name), index=False)
                            scores = []

    scores = pd.DataFrame(scores)

    # Merge existing scores
    if Path(save_path, file_name).exists():
        print("Found existing scores...")
        print("Merging...")
        temp_scores = pd.read_csv(Path(save_path, file_name))
        scores = pd.concat([temp_scores, scores], ignore_index=True)

    print("Final save...")
    scores.to_csv(Path(save_path, file_name), index=False)
