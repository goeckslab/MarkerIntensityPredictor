from MarkerIntensity import MarkerIntensity
from LinearMarkerIntensity import LinearMarkerIntensity
from services.args_parser import ArgumentParser
from pathlib import Path
import logging
import sys

if __name__ == "__main__":

    args = ArgumentParser.load_args()
    path: Path
    try:
        path = Path(args.file)
    except FileNotFoundError:
        logging.error(f"Could not locate file {args.file}")
        sys.exit()

    if not args.dl:
        marker = LinearMarkerIntensity(path, args.model)
        marker.load()
        marker.train_predict()
        marker.write_csv()
        marker.create_plots()

    else:
        marker = MarkerIntensity(path)
        marker.load()
        marker.train()
        marker.prediction()
