from args_parser import ArgumentParser
import Plotting.main as plt
from AE.ae import AutoEncoder
from DAE.dae import DenoisingAutoEncoder
from pathlib import Path
import sys
import logging
from LinearRegression.lr import LinearMarkerIntensity
from Shared.prepare import Prepare


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

    elif invoked_parser == "plt":
        plt.start(args)
