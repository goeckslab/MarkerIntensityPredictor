from src.data.folder_management import FolderManagement
from src.arguments.args_parser import ArgumentParser
from src.model.main import VAutoEncoder
from src.data.data_loader import DataLoader
import logging
import mlflow
from src.plots.plots import Plotting

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def handle_model_training(args):
    # Load cells and markers from the given file
    with mlflow.start_run(run_name='Variational Auto Encoder') as run:
        mlflow.log_param("file", args.file)
        mlflow.log_param("morphological_data", args.morph)
        cells, markers = DataLoader.load_data(file_name=args.file)
        print(cells, markers)

        vae = VAutoEncoder(args, cells, markers)
        vae.build_auto_encoder()
        Plotting.plot_model_performance(vae.history, "Model performance")


if __name__ == "__main__":
    args = ArgumentParser.get_args()
    FolderManagement.create_folders(args)

    # Get invoked parser
    invoked_parser = args.command

    if invoked_parser == "model":
        handle_model_training(args=args)
