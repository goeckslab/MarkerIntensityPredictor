import logging
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class Plotting:
    @staticmethod
    def plot_model_performance(history, file_name: str):
        logger.info("Plotting model performance")
        plt.figure(num=None, figsize=(6, 4), dpi=90)
        for key in history.history:
            plt.plot(history.history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        save_path = Path("VAE", "results", f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()
