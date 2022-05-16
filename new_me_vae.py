import pandas as pd
from library.data import DataLoader, FolderManagement
from library.preprocessing import SplitHandler, Preprocessing
from library.new_me_vae.new_me_vae_model import NewMeVAE
import argparse
from pathlib import Path
from library.predictions.predictions import Predictions
from library.evalation.evaluation import Evaluation

base_path = Path("new_me_vae")


def get_args():
    """
    Load all provided cli args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", action="store", required=True,
                        help="The name of the run being run",
                        type=str)
    parser.add_argument("--tracking_url", "-t", action="store", required=False,
                        help="The tracking url for the mlflow tracking server", type=str,
                        default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", "-e", action="store", required=False,
                        help="Assigns the run to a particular experiment. "
                             "If the experiment does not exists it will create a new one.",
                        default="Default", type=str)
    parser.add_argument("--description", "-d", action="store", required=False,
                        help="A description for the experiment to give a broad overview. "
                             "This is only used when a new experiment is being created. Ignored if experiment exists",
                        type=str)
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder used to load the data")
    parser.add_argument("--exclude", action="store", required=False,
                        help="A file which can be excluded from training.", default=None)
    parser.add_argument("--morph", action="store_true", help="Include morphological data", default=True)
    parser.add_argument("--seed", "-s", action="store", help="Include morphological data", type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    FolderManagement.create_directory(path=base_path)

    # Load data
    me_vae_train_cells, features, files_used = DataLoader.load_files_in_folder(folder=args.folder,
                                                                               file_to_exclude=args.exclude)

    me_vae_train_data, me_vae_validation_data = SplitHandler.create_splits(cells=me_vae_train_cells,
                                                                           create_val=False, seed=args.seed,
                                                                           features=features)

    me_vae_marker_train_data, me_vae_morph_train_data = SplitHandler.split_dataset_into_markers_and_morph_features(
        data_set=me_vae_train_data)

    me_vae_marker_validation_data, me_vae_morph_validation_data = SplitHandler.split_dataset_into_markers_and_morph_features(
        data_set=me_vae_validation_data)

    # Normalize
    me_vae_marker_train_data: pd.DataFrame = pd.DataFrame(
        data=Preprocessing.normalize(me_vae_marker_train_data.copy()),
        columns=me_vae_marker_train_data.columns)

    me_vae_morph_train_data: pd.DataFrame = pd.DataFrame(
        data=Preprocessing.normalize(me_vae_morph_train_data.copy()),
        columns=me_vae_morph_train_data.columns)

    me_vae_marker_validation_data: pd.DataFrame = Preprocessing.normalize(data=me_vae_marker_validation_data.copy(),
                                                                          create_dataframe=True,
                                                                          columns=me_vae_marker_validation_data.columns.tolist())

    me_vae_morph_validation_data: pd.DataFrame = Preprocessing.normalize(data=me_vae_morph_validation_data.copy(),
                                                                         create_dataframe=True,
                                                                         columns=me_vae_morph_validation_data.columns.tolist())

    me_vae_train_target_data: pd.DataFrame = Preprocessing.normalize(data=me_vae_train_data.copy(),
                                                                     create_dataframe=True,
                                                                     columns=features)

    me_vae_val_target_data: pd.DataFrame = Preprocessing.normalize(data=me_vae_validation_data.copy(),
                                                                   create_dataframe=True,
                                                                   columns=features)

    # Load test cell, which is the excluded data file
    me_vae_test_cells, _ = DataLoader.load_single_cell_data(file_name=args.exclude)

    me_vae_test_cells = pd.DataFrame(data=me_vae_test_cells, columns=features)

    # Split hold out set into marker and morph data for testing model performance
    marker_test_data, morph_test_data = SplitHandler.split_dataset_into_markers_and_morph_features(
        data_set=me_vae_test_cells.copy())

    # Normalize test data
    marker_test_data: pd.DataFrame = Preprocessing.normalize(data=marker_test_data, create_dataframe=True,
                                                             columns=marker_test_data.columns.tolist())

    morph_test_data: pd.DataFrame = Preprocessing.normalize(data=morph_test_data, create_dataframe=True,
                                                            columns=morph_test_data.columns.tolist())

    me_vae_test_data = Preprocessing.normalize(data=me_vae_test_cells.copy(), create_dataframe=True, columns=features)

    me_vae: NewMeVAE = NewMeVAE(embedding_dimensions=5, marker_input_dimensions=me_vae_marker_train_data.shape[1],
                                morph_input_dimensions=me_vae_morph_validation_data.shape[1], learning_rate=0.0001,
                                results_path=base_path)
    me_vae.build_model()


    print(me_vae_marker_train_data)
    input()
    print(me_vae_morph_train_data)
    input()

    me_vae.train(marker_train_data=me_vae_marker_train_data, marker_val_data=me_vae_marker_validation_data,
                 morph_train_data=me_vae_morph_train_data, morph_val_data=me_vae_morph_validation_data,
                 train_target_data=me_vae_train_target_data, val_target_data=me_vae_val_target_data)

    # Predictions
    encoded_data, reconstructed_data = Predictions.encode_decode_new_me_vae_data(marker_encoder=me_vae.marker_encoder,
                                                                                 morph_encoder=me_vae.morph_encoder,
                                                                                 decoder=me_vae.decoder,
                                                                                 marker_data=marker_test_data,
                                                                                 morph_data=morph_test_data,
                                                                                 features=features,
                                                                                 save_path=base_path,
                                                                                 use_mlflow=False)

    print(reconstructed_data)
    input()

    # Evaluate
    me_vae_r2_scores = Evaluation.calculate_r2_scores(ground_truth_data=me_vae_test_data,
                                                      compare_data=reconstructed_data,
                                                      features=features)
    print(me_vae_r2_scores)
