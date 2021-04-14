from VAE.MarkerIntensity import MarkerIntensity
from LinearRegression.LinearMarkerIntensity import LinearMarkerIntensity
from shared.services.args_parser import ArgumentParser
from LudwigAi.ludwig import LudwigAi
from pathlib import Path
import logging
import sys

if __name__ == "__main__":

    args = ArgumentParser.load_args()
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

    if args.mode == "Linear":
        marker = LinearMarkerIntensity(train_file, args, test_file)
        marker.load()
        marker.train_predict()
        marker.write_csv()
        marker.create_plots()

    elif args.mode == "DL":
        marker = MarkerIntensity(train_file)
        marker.load()
        marker.train()
        marker.prediction()
    else:
        ludwig = LudwigAi(train_file)