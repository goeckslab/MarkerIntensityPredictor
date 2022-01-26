import os
import shutil
import pandas as pd
from typing import Tuple
from Shared.data_loader import DataLoader
from args_parser import ArgumentParser
import Plotting.main as plt
from AE.ae import AutoEncoder
from DAE.dae import DenoisingAutoEncoder
from src.model import VAutoEncoder
from pathlib import Path
import sys
import logging
from LinearRegression.lr import LinearMarkerIntensity
from Shared.prepare import Prepare
from ClusterAnalysis.main import ClusterAnalysis
from PCA.main import PCAMode
from sklearn.model_selection import KFold


def execute_linear_regression():
    train_file: Path
    test_file = None
    try:
        train_file = Path(args.file.name)
        if args.validation is not None:
            test_file = Path(args.validation)
        else:
            test_file = None
    except FileNotFoundError:
        logging.error(f"Could not locate file {args.file} or {args.validation}")
        sys.exit()

    print("Started linear marker intensity prediction")
    marker = LinearMarkerIntensity(train_file, test_file, args)
    marker.load()
    marker.train_predict()
    marker.write_csv()
    print("Finished linear marker intensity prediction")


def execute_auto_encoder():
    ae = AutoEncoder(args)
    ae.load_data()
    ae.build_auto_encoder()
    ae.predict()
    ae.calculate_r2_score()
    ae.create_h5ad_object()
    ae.create_test_predictions()
    ae.create_correlation_data()
    ae.write_created_data_to_disk()
    ae.get_activations()
    ae.plot_model()


def execute_denoising_auto_encoder():
    dae = DenoisingAutoEncoder(args)
    dae.load_data()
    dae.add_noise()
    dae.build_auto_encoder()
    dae.predict()
    dae.calculate_r2_score()
    dae.create_h5ad_object()
    dae.k_means()
    dae.create_test_predictions()
    dae.write_created_data_to_disk()


def execute_vae():
    data_set, markers = load_data_set()
    if args.folds > 0:
        kf = KFold(n_splits=5)
        run = 0
        # Correct this: https://towardsdatascience.com/5-reasons-why-you-should-use-cross-validation-in-your-data-science-project-8163311a1e79
        for train, test in kf.split(data_set):
            # Cleanup old folders
            path = Path("results", "vae", f"Run_{str(run)}")
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.mkdir(path)

            train_set = data_set.iloc[train]
            test_set = data_set.iloc[test]
            # Create new vae object
            vae = VAutoEncoder(args=args, markers=markers, dataset=data_set, train=train_set, test=test_set,
                               results_folder=path)
            vae.build_auto_encoder()
            vae.predict()
            vae.calculate_r2_score()
            vae.create_h5ad_object()
            vae.create_test_predictions()
            vae.create_correlation_data()
            vae.write_created_data_to_disk()
            vae.get_activations()
            run += 1

    else:
        print("Running vae without folds")
        vae = VAutoEncoder(args=args, dataset=data_set, markers=markers)
        vae.check_mean_and_std()
        vae.build_auto_encoder()
        vae.predict()
        vae.calculate_r2_score()
        vae.create_h5ad_object()
        vae.create_test_predictions()
        vae.create_correlation_data()
        vae.write_created_data_to_disk()
        vae.get_activations()


def pca_clustering():
    pca = PCAMode(args)
    pca.load_data()
    pca.reduce_dimensions()


def cluster_analysis():
    cluster = ClusterAnalysis(args)
    # Load all file
    if args.mean:
        print("Creating mean plots")
        cluster.create_mean_score_plots()
    else:
        cluster.create_cluster()


def load_data_set() -> Tuple[pd.DataFrame, list]:
    """
    Loads the data set given by the cli args
    """
    print("Loading data...")

    inputs: pd.DataFrame
    markers: list

    if args.file:
        inputs, markers = DataLoader.get_data(
            input_file=args.file, keep_morph=args.morph)

    elif args.dir:
        inputs, markers = DataLoader.load_folder_data(
            args.dir, args.morph)

    else:
        print("Please specify a directory or a file")
        sys.exit()

    return inputs, markers


if __name__ == "__main__":
    args = ArgumentParser.get_args()

    Prepare.create_folders(args)
    invoked_parser = args.command

    if invoked_parser == "lr":
        execute_linear_regression()

    elif invoked_parser == "ae":
        execute_auto_encoder()

    elif invoked_parser == "dae":
        execute_denoising_auto_encoder()

    elif invoked_parser == "vae":
        execute_vae()

    elif invoked_parser == "plt":
        plt.start(args)

    elif invoked_parser == 'cl':
        print("Starting cluster analysis")
        cluster_analysis()

    elif invoked_parser == 'pca':
        pca_clustering()
