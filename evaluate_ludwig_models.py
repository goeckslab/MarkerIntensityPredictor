from pathlib import Path
from ludwig.api import LudwigModel
import pandas as pd
import random
from tqdm import tqdm
import logging, sys, os, argparse
from typing import List

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']


def get_logger(
        LOG_FORMAT='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        LOG_NAME='',
        LOG_FILE_INFO='ludwig_evaluation.log',
        LOG_FILE_ERROR='ludwig_evaluation.err'):
    log = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(LOG_FORMAT)

    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)

    file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='w')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_error = logging.FileHandler(LOG_FILE_ERROR, mode='w')
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    log.setLevel(logging.DEBUG)

    return log


def create_scores_dir(mode: str, radius: int, hyper: bool) -> Path:
    scores_directory = Path("data/scores/lgbm")
    scores_directory = Path(scores_directory, mode)

    if not hyper:
        scores_directory = Path(scores_directory, f"{radius}")
    else:
        scores_directory = Path(scores_directory, f"Ludwig_hyper")

    scores_directory = Path(scores_directory)

    if not scores_directory.exists():
        scores_directory.mkdir(parents=True, exist_ok=True)

    return scores_directory


def save_scores(save_folder: Path, file_name: str, scores: List):
    logging.debug(f"Temp saving scores")
    scores = pd.DataFrame(scores)

    # find empty rows
    empty_rows = scores.isnull().all(axis=1)
    if empty_rows.any():
        logging.debug(f"Found empty rows: {empty_rows.sum()}")
    # drop empty rows
    scores = scores[~empty_rows]

    # find nan seed rows
    nan_seed_rows = scores['Random Seed'].isnull()
    # replace with 0
    scores.loc[nan_seed_rows, 'Random Seed'] = 0

    if 'Unnamed: 0' in scores.columns or len(scores.columns) > 11:
        logger.debug("Found unnamed column or more than 11 columns. Skipping...")
        return

    if Path(save_path, file_name).exists():
        logging.debug("Found existing scores...")
        logging.debug("Merging...")
        try:
            temp_scores = pd.read_csv(Path(save_path, file_name))
            scores = pd.concat([temp_scores, scores], ignore_index=True)
        except BaseException as ex:
            # continue without doing anything
            logger.error("Error occurred saving scores")
            logger.error(ex)

            pass

    scores.to_csv(Path(save_folder, file_name), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--biopsy", type=str, required=True,
                        help="The biopsy. For an ip mode that is the train biopsy, for exp mode that is the test biopsy due to ludwigs setup")
    parser.add_argument('-sp', '--spatial', type=int, required=False, default=0, help="The radius",
                        choices=[23, 46, 92, 138, 184])
    parser.add_argument('--mode', type=str, choices=['ip', 'exp'], help="The mode", default='ip')
    parser.add_argument('--hyper', action="store_true", help="Use hyperopt", default=False)
    parser.add_argument("--subsets", "-s", type=int, default=101, help="The number of subsets")
    args = parser.parse_args()

    logger = get_logger()
    spatial_radius: int = args.spatial
    mode = args.mode
    biopsy: str = args.biopsy
    hyper: bool = args.hyper
    subsets: int = args.subsets

    logger.debug(f"Mode: {mode}")
    logger.debug(f"Biopsy: {biopsy}")
    logger.debug(f"Radius: {spatial_radius}")
    logger.debug(f"Hyper: {hyper}")
    logger.debug(f"Subsets: {subsets}")

    if mode == "ip":
        # change last number of biopsy to 1 if it is 2
        if biopsy[-1] == "2":
            test_biopsy_name = biopsy[:-1] + "1"
        else:
            test_biopsy_name = biopsy[:-1] + "2"

        logger.debug(biopsy)
        logger.debug(test_biopsy_name)
        assert test_biopsy_name[-1] != biopsy[-1], "The bx should not be the same"
        if spatial_radius == 0:
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
        logger.debug(test_biopsy_name)

        if spatial_radius == 0:
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

    logger.debug(f"Base path: {base_path}")
    scores = []
    save_path = create_scores_dir(mode=mode, radius=spatial_radius, hyper=hyper)
    score_file_name = f"{test_biopsy_name}_scores.csv"
    logger.debug(f"Save path:  {str(save_path)}")
    logger.debug(f"Score file name: {score_file_name}")
    try:
        for marker in SHARED_MARKERS:
            results_path = Path(base_path, marker, "results")
            for root, marker_sub_directories, files in os.walk(str(results_path)):
                if "experiment_run" in marker_sub_directories:
                    for experiment in marker_sub_directories:
                        models = None
                        try:
                            model = LudwigModel.load(str(Path(results_path, experiment, 'model')))
                        except KeyboardInterrupt as ex:
                            logger.debug("Keyboard interrupt")
                            sys.exit(0)
                        except BaseException as ex:
                            logger.error(ex)
                            continue

                        for i in tqdm(range(1, subsets)):
                            random_seed = random.randint(0, 100000)
                            # sample new dataset from test_data
                            test_data_sample = test_dataset.sample(frac=0.7, random_state=random_seed,
                                                                   replace=True)

                            test_data_sample.reset_index(drop=True, inplace=True)
                            try:
                                eval_stats, _, _ = model.evaluate(dataset=test_data_sample)
                            except KeyboardInterrupt as ex:
                                logger.debug("Keyboard interrupt")
                                sys.exit(0)
                            except BaseException as ex:
                                logger.error(f"Error occurred for experiment: {experiment}")
                                logger.error(
                                    f"Model loaded using path: {str(Path(results_path, experiment, 'model'))}")
                                logger.error(ex)
                                logger.error("Continuing to next experiment")
                                continue

                            scores.append(
                                {
                                    "Marker": marker,
                                    "MAE": eval_stats[marker]['mean_absolute_error'],
                                    "MSE": eval_stats[marker]['mean_squared_error'],
                                    "RMSE": eval_stats[marker]['root_mean_squared_error'],
                                    "Biopsy": test_biopsy_name,
                                    "Mode": mode,
                                    "FE": spatial_radius,
                                    "Network": "Ludwig",
                                    "Hyper": int(hyper),
                                    "Load Path": str(Path(results_path, experiment, 'model')),
                                    "Random Seed": int(random_seed),
                                }
                            )

                            if i % 20 == 0:
                                logger.debug("Temp saving scores...")
                                save_scores(scores=scores, save_folder=save_path, file_name=score_file_name)
                                scores = []

                        if len(scores) > 0:
                            # Save scores after each experiment
                            logger.debug("Temp saving scores...")
                            save_scores(scores=scores, save_folder=save_path, file_name=score_file_name)
                            scores = []

        if len(scores) > 0:
            logger.debug("Saving scores...")
            save_scores(scores=scores, save_folder=save_path, file_name=score_file_name)
    except KeyboardInterrupt as ex:
        logger.debug("Detected Keyboard interrupt...")
        logger.debug("Saving scores....")
        if len(scores) > 0:
            save_scores(scores=scores, save_folder=save_path, file_name=score_file_name)
