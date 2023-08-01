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
        LOG_FILE_INFO='ludwig_prediction.log',
        LOG_FILE_ERROR='ludwig_prediction.err'):
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
    scores_directory = Path("data/cleaned_data/predictions/lgbm")
    scores_directory = Path(scores_directory, mode)

    if radius is not None:
        scores_directory = Path(scores_directory, f"{radius}")
    elif hyper:
        scores_directory = Path(scores_directory, f"Ludwig_hyper")
    else:
        scores_directory = Path(scores_directory, "0")

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
        logger.debug(test_biopsy_name)

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

    logger.debug(f"Base path: {base_path}")
    predictions = {}

    save_path = create_scores_dir(mode=mode, radius=spatial_radius, hyper=hyper)
    file_name = f"{test_biopsy_name}_scores.csv"
    logger.debug(f"Save path:  {str(save_path)}")
    logger.debug(f"File name: {file_name}")
    try:
        for marker in SHARED_MARKERS:
            results_path = Path(base_path, marker, "results")
            for root, marker_sub_directories, files in os.walk(str(results_path)):
                if  marker_sub_directories == "experiment_run":
                    for experiment in marker_sub_directories:

                        current_path: Path = Path(root, experiment)
                        path_splits: [] = current_path.parts

                        experiment_id: int = 0 if path_splits[-1] == "experiment_run" else int(
                            path_splits[-1].split("_")[-1])

                        protein = path_splits[3]

                        logging.debug(f"Biopsy: {biopsy}")
                        logging.debug(f"Mode: {mode}")
                        logging.debug(f"Experiment ID: {experiment_id}")
                        logging.debug(f"Radius: {spatial_radius}")
                        logging.debug(f"Hyper: {hyper}")
                        logging.debug(f"Protein: {protein}")

                        assert mode == "ip" or mode == "exp", f"Mode {mode} not in ['ip', 'exp']"

                        assert hyper in [0, 1], f"Hyper {hyper} not in [0,1]"
                        assert protein in SHARED_MARKERS, f"Protein {protein} not in SHARED_MARKERS"

                        unique_key = f"{biopsy}||{mode}||{spatial_radius}||{hyper}"
                        input()

                        models = None
                        try:
                            model = LudwigModel.load(str(Path(results_path, experiment, 'model')))
                        except KeyboardInterrupt as ex:
                            logger.debug("Keyboard interrupt")
                            sys.exit(0)
                        except BaseException as ex:
                            logger.error(ex)
                            continue

                        try:
                            # predict on test_data
                            predictions, _ = model.predict(dataset=test_dataset)

                        except KeyboardInterrupt as ex:
                            logger.debug("Keyboard interrupt")
                            sys.exit(0)

                        except BaseException as ex:
                            logger.error(f"Error occurred for experiment: {experiment}")
                            logger.error(f"Model loaded using path: {str(Path(results_path, experiment, 'model'))}")
                            logger.error(ex)
                            logger.error("Continuing to next experiment")
                            continue

                        try:
                            # predict values with model
                            protein_predictions, _ = model.predict(dataset=test_dataset)

                        except KeyboardInterrupt as ex:
                            logger.debug("Keyboard interrupt")
                            sys.exit(0)

                        except BaseException as ex:
                            logger.error(f"Error occurred for experiment: {experiment}")
                            logger.error(f"Model loaded using path: {str(Path(results_path, experiment, 'model'))}")
                            logger.error(ex)
                            logger.error("Continuing to next experiment")
                            continue

                        predictions[protein] = protein_predictions[protein].values




    except KeyboardInterrupt as ex:
        logger.debug("Keyboard interrupt")
        sys.exit(0)
