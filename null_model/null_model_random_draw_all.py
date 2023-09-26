import pandas as pd
from ludwig.api import LudwigModel
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse, logging

BIOPSIES = ["9_2_1", "9_2_2", "9_3_1", "9_3_2", "9_14_1", "9_14_2", "9_15_1", "9_15_2"]
SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("ae_imputation_m/debug.log"),
                        logging.StreamHandler()
                    ])


def setup_log_file(save_path: Path):
    save_file = Path(save_path, "random_draw_all_debug.log")

    if save_file.exists():
        save_file.unlink()

    file_logger = logging.FileHandler(save_file, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_logger.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for handler in log.handlers[:]:  # remove all old handlers
        log.removeHandler(handler)
    log.addHandler(file_logger)
    log.addHandler(logging.StreamHandler())


def evaluate_samples(marker: str, train_set: pd.DataFrame, test_set: pd.DataFrame, biopsy: str, train_set_used: bool):
    try:

        y_truth = test_set[marker]

        # draw random samples from train set for each test set sample
        y_hat = train_set.sample(n=len(test_set), replace=True)[marker]

        biopsy_evaluation.append({
            "Biopsy": biopsy,
            "Marker": marker,
            "MAE": mean_absolute_error(y_truth, y_hat),
            "MSE": mean_squared_error(y_truth, y_hat),
            "RMSE": mean_squared_error(y_truth, y_hat, squared=False),
            "Null Model": 1,
            "Model Path": "",
            "Train": int(train_set_used),
        })

        model_path: Path = Path("mesmer", "tumor_exp_patient", biopsy, marker, "results", "experiment_run",
                                "model")

        if f"{biopsy}_{marker}" not in lgbm_loaded_models:
            lgbm_loaded_models[f"{biopsy}_{marker}"] = LudwigModel.load(str(model_path))

        # load model for marker and biopsy
        model = lgbm_loaded_models[f"{biopsy}_{marker}"]
        own_eval_stats, _, _ = model.evaluate(train_set)

        biopsy_evaluation.append({
            "Biopsy": biopsy,
            "Marker": marker,
            "MAE": own_eval_stats[marker]['mean_absolute_error'],
            "MSE": own_eval_stats[marker]['mean_squared_error'],
            "RMSE": own_eval_stats[marker]['root_mean_squared_error'],
            "Null Model": 0,
            "Model Path": str(model_path),
            "Train": int(train_set_used),
        })

    except BaseException as ex:
        logging.error(f"Error evaluating {marker} for {biopsy}.")
        logging.error(ex)
        logging.error(ex.__traceback__)
        raise


if __name__ == '__main__':
    setup_log_file(save_path=Path("null_model"))

    parser = argparse.ArgumentParser(description='Run null model')
    parser.add_argument('--experiments', "-ex", type=int, default=1, help='The amount of experiments to run')
    args = parser.parse_args()
    experiments: int = args.experiments

    biopsy_evaluation = []
    lgbm_loaded_models = {}
    train_data_sets = {}
    test_data_sets = {}

    try:
        for i in range(experiments):
            logging.info(f"Started {i} experiment...")
            for biopsy in BIOPSIES:

                patient: str = '_'.join(biopsy.split('_')[:2])
                logging.debug(f"Processing biopsy: {biopsy} and patient {patient}")
                # load ground truth

                if biopsy not in test_data_sets:
                    test_data_sets[biopsy] = pd.read_csv(
                        Path("data", "tumor_mesmer", "preprocessed", f"{biopsy}_preprocessed_dataset.tsv"),
                        sep="\t")

                # load train data
                if patient not in train_data_sets:
                    train_data_sets[patient] = pd.read_csv(
                        Path("data", "tumor_mesmer", "combined", "preprocessed", f"{patient}_excluded_dataset.tsv"),
                        sep="\t")

                # load train data from dict
                train_data: pd.DataFrame = train_data_sets[patient]
                test_data: pd.DataFrame = test_data_sets[biopsy]

                # assert ground truth is not empty
                assert train_data.shape[0] > 0, "Ground truth is empty"

                for marker in tqdm(train_data.columns):
                    evaluate_samples(marker=marker, train_set=train_data, test_set=train_data, biopsy=biopsy,
                                     train_set_used=True)
                    evaluate_samples(marker=marker, train_set=train_data, test_set=test_data, biopsy=biopsy,
                                     train_set_used=False)


    except BaseException as ex:
        logging.error(ex)
        print("Saving current results...")

    try:
        biopsy_evaluation = pd.DataFrame(biopsy_evaluation)

        save_path: Path = Path("data", "cleaned_data", "null_model", "random_draw_all_performance.csv")
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        biopsy_evaluation.to_csv(save_path, index=False)
    except BaseException as ex:
        logging.error(ex)
        biopsy_evaluation = pd.DataFrame(biopsy_evaluation)
        biopsy_evaluation.to_csv(Path("null_model_random_draw_all_performance.csv"), index=False)
