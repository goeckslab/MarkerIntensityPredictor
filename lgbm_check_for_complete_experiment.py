import logging

SHARED_MARKER = []

SEARCH_PATHS  = [
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

if __name__ == '__main__':

    unfinished_experiment_paths = []
    for load_path in SEARCH_PATHS:
        for root, sub_directories, files in os.walk(load_path):
            for sub_directory in sub_directories:
                current_path = Path(root, sub_directory)
                if 'experiment_run' not in str(current_path) or 'experiment_run' != current_path.parts[-1]:
                    continue

                logging.debug("Current path: " + str(current_path))

                try:
                    model = LudwigModel.load(str(Path(current_path, 'model')))
                except KeyboardInterrupt as ex:
                    logging.debug("Keyboard interrupt")
                    sys.exit(0)

                except BaseException as ex:
                    logging.debug(f"Found missing experiment {current_path}. Adding to list. ")
                    logging.debug(f"List contains {len(unfinished_experiment_paths)} elements")
                    unfinished_experiment_paths.append(current_path)


    for path in unfinished_experiment_paths:
        logging.debug(f"Path: {path}")
        experiment: str = path.parts[-1]
        biopsy: str = path.parts[-4]
        for marker in SHARED_MARKER:
            marker_path = Path(path[0], biopsy, marker, 'results', experiment)
            logging.debug(f"Marker path: {marker_path}")
            input()
            # delete marker_path if exists
            if marker_path.exists():
                shutil.rmtree(marker_path)
                logging.debug("Removed: " + str(marker_path))




