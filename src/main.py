import os

import pandas as pd

from args_parser import ArgumentParser
import Plotting.main as plt
from AE.ae import AutoEncoder
from DAE.dae import DenoisingAutoEncoder
from VAE.main import VAutoEncoder
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
        train_file = Path(args.file)
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
    vae = VAutoEncoder(args)
    vae.load_data_set()

    kf = KFold(n_splits=6)
    run = 0
    for train, test in kf.split(vae.data_set):
        # Cleanup old folders
        path = Path("results", "vae", str(run))
        if os.path.isfile(path):
            os.remove(path)
        os.mkdir(path)

        vae.results_folder = Path("results", "vae", str(run))
        train_set = vae.data_set.iloc[train]
        test_set = vae.data_set.iloc[test]
        vae.load_data(train_set, test_set)
        vae.build_auto_encoder()
        vae.predict()
        vae.calculate_r2_score()
        vae.create_h5ad_object()
        vae.create_test_predictions()
        vae.create_correlation_data()
        vae.write_created_data_to_disk()
        vae.reset()
        run += 1


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
