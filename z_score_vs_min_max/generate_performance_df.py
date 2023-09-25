from pathlib import Path
from ludwig.api import LudwigModel
import sys
import pandas as pd

# Only valid for ludwig models


DIRS = {
    "Standard": Path("mesmer", "tumor_exp_patient_z_scores"),
    "Min Max": Path("mesmer", "tumor_exp_patient"),
}

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':

    evaluation_results = []

    for scaler, directory in DIRS.items():
        for biopsy in BIOPSIES:
            for marker in SHARED_MARKERS:
                load_path = Path(directory, biopsy, marker, "results", "experiment_run", "model")
                try:
                    model = LudwigModel.load(str(load_path))
                except KeyboardInterrupt as ex:
                    sys.exit(0)
                except BaseException as ex:
                    print(ex)
                    continue

                test_dataset: pd.DataFrame = pd.read_csv(
                    Path("data", "tumor_mesmer_z_scores", "preprocessed", f"{biopsy}_preprocessed_dataset.tsv"),
                    sep='\t')

                eval_stats, _, _ = model.evaluate(dataset=test_dataset)

                evaluation_results.append(
                    {
                        "Marker": marker,
                        "MAE": eval_stats[marker]['mean_absolute_error'],
                        "MSE": eval_stats[marker]['mean_squared_error'],
                        "RMSE": eval_stats[marker]['root_mean_squared_error'],
                        "RMSEPE": eval_stats[marker]['root_mean_squared_percentage_error'],
                        "Biopsy": biopsy,
                        "Mode": "EXP",
                        "FE": 0,
                        "Network": "LGBM",
                        "Hyper": 0,
                        "Load Path": str(load_path),
                        "Scaler": scaler
                    }
                )

    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.to_csv(Path("z_score_vs_min_max", "scores.csv"), index=False)
