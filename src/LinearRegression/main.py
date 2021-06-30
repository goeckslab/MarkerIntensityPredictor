from LinearRegression.lr import LinearMarkerIntensity
from pathlib import Path
import logging
import sys


def start(args):
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
    marker = LinearMarkerIntensity(train_file, args, test_file)
    marker.load()
    marker.train_predict()
    marker.write_csv()
    print("Finished linear marker intensity prediction")
