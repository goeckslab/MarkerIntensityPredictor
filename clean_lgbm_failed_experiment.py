import logging, sys, os, shutil
from pathlib import Path
from ludwig.api import LudwigModel

SHARED_MARKERS = ['pRB', 'CD45', 'CK19', 'Ki67', 'aSMA', 'Ecad', 'PR', 'CK14', 'HER2', 'AR', 'CK17', 'p21', 'Vimentin',
                  'pERK', 'EGFR', 'ER']

SEARCH_PATHS = [
    Path("mesmer", "tumor_in_patient"),
    Path("mesmer", "tumor_in_patient_sp_23"),
    Path("mesmer", "tumor_in_patient_sp_46"),
    Path("mesmer", "tumor_in_patient_sp_92"),
    Path("mesmer", "tumor_in_patient_sp_138"),
    Path("mesmer", "tumor_in_patient_sp_184"),
    Path("mesmer", "tumor_exp_patient"),
    Path("mesmer", "tumor_exp_patient_sp_23"),
    Path("mesmer", "tumor_exp_patient_sp_46"),
    Path("mesmer", "tumor_exp_patient_sp_92"),
    Path("mesmer", "tumor_exp_patient_sp_138"),
    Path("mesmer", "tumor_exp_patient_sp_184"),

]

logging.root.handlers = []
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ])


def setup_log_file(save_path: Path):
    save_file = Path(save_path, "lgbm_check_for_complete_experiment.log")

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


def delete_all_markers_for_failed_experiment(experiment_path: Path):
    logging.debug(f"Path: {experiment_path}")
    experiment: str = experiment_path.parts[-1]
    biopsy: str = experiment_path.parts[-4]

    for marker in SHARED_MARKERS:
        marker_path = Path(experiment_path.parts[0], experiment_path.parts[1], biopsy, marker, 'results', experiment)
        logging.debug(f"Marker path: {marker_path}")
        # delete marker_path if exists
        if marker_path.exists():
            shutil.rmtree(marker_path)
            logging.debug("Removed: " + str(marker_path))


if __name__ == '__main__':

    setup_log_file(Path.cwd())
    logging.debug("Started searching...")
    unfinished_experiment_paths = []
    for load_path in SEARCH_PATHS:
        for root, sub_directories, files in os.walk(load_path):
            for sub_directory in sub_directories:
                current_path = Path(root, sub_directory)
                if 'experiment_run' not in str(current_path) or 'experiment_run' not in current_path.parts[-1]:
                    continue

                logging.debug("Current path: " + str(current_path))

                try:
                    model_hyper_json: Path = Path(current_path, 'model', 'model_hyperparameters.json')
                    training_set_metadata_json: Path = Path(current_path, 'model', 'training_set_metadata.json')

                    if not model_hyper_json.exists() or not training_set_metadata_json.exists():
                        logging.debug(f"Found missing files! Deleting experiment: {current_path}")
                        delete_all_markers_for_failed_experiment(current_path)
                        continue


                except KeyboardInterrupt as ex:
                    logging.debug("Keyboard interrupt")
                    sys.exit(0)

                except BaseException as ex:
                    logging.debug(f"Found missing experiment {current_path}. Adding to list. ")
                    unfinished_experiment_paths.append(current_path)
                    delete_all_markers_for_failed_experiment(current_path)
                    logging.debug(f"List contains {len(unfinished_experiment_paths)} elements")
