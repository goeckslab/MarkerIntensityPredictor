import pandas as pd
from ludwig.api import LudwigModel
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run null model')
    parser.add_argument('--experiments', "-ex", type=int, default=1, help='The amount of experiments to run')
    args = parser.parse_args()
    experiments: int = args.experiments

    biopsy_evaluation = []

    for i in range(experiments):
        for biopsy in BIOPSIES:

            patient: str = '_'.join(biopsy.split('_')[:2])
            print(f"Processing biopsy: {biopsy} and patient {patient}")
            # load ground truth
            # test_data: pd.DataFrame = pd.read_csv(
            #    Path("data", "tumor_mesmer", "preprocessed", f"{biopsy}_preprocessed_dataset.tsv"), sep="\t")

            train_data: pd.DataFrame = pd.read_csv(
                Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"), sep="\t")

            # assert ground truth is not empty
            assert train_data.shape[0] > 0, "Ground truth is empty"

            for marker in tqdm(train_data.columns):
                X = train_data.drop(marker, axis=1)
                y = train_data[marker]

                # split data into train and test using train test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

                regression_model = LinearRegression().fit(X_train, y_train)

                # draw one random samples from test data
                x_truth = X_test.sample(n=1, random_state=42)
                # draw same sample from y_test
                y_truth = y_test.loc[x_truth.index]

                # draw sample for ludwig with the evaluates sample index
                evaluation_sample = train_data.loc[x_truth.index]

                # assert that test sample shape is one
                assert evaluation_sample.shape[0] == 1, "Test sample is not one"

                y_hat = regression_model.predict(x_truth)

                biopsy_evaluation.append({
                    "Biopsy": biopsy,
                    "Marker": marker,
                    "MAE": mean_absolute_error(y_truth, y_hat),
                    "MSE": mean_squared_error(y_truth, y_hat),
                    "RMSE": mean_squared_error(y_truth, y_hat, squared=False),
                    "Sample Index": evaluation_sample.index[0],
                    "Null Model": 1,
                    "Model Path": ""
                })

                model_path: Path = Path("mesmer", "tumor_exp_patient", biopsy, marker, "results", "experiment_run",
                                        "model")
                print(f"Loading model from: {model_path}")
                # load model for marker and biopsy
                model = LudwigModel.load(str(model_path))
                own_eval_stats, _, _ = model.evaluate(evaluation_sample)

                biopsy_evaluation.append({
                    "Biopsy": biopsy,
                    "Marker": marker,
                    "MAE": own_eval_stats[marker]['mean_absolute_error'],
                    "MSE": own_eval_stats[marker]['mean_squared_error'],
                    "RMSE": own_eval_stats[marker]['root_mean_squared_error'],
                    "Sample Index": evaluation_sample.index[0],
                    "Null Model": 0,
                    "Model Path": str(model_path)
                })

    try:
        biopsy_evaluation = pd.DataFrame(biopsy_evaluation)
        biopsy_evaluation.to_csv(Path("null_model", "sample_performance.csv"), index=False)
    except BaseException as ex:
        print(ex)
        biopsy_evaluation = pd.DataFrame(biopsy_evaluation)
        biopsy_evaluation.to_csv(Path("null_model_sample_performance.csv"), index=False)
