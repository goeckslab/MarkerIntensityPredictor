import logging
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import pandas as pd
import seaborn as sns

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

    @staticmethod
    def plot_reconstructed_markers(X, X_pred, markers, file_name: str):
        logging.info("Plotting reconstructed intensities")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(X, ax=ax1, xticklabels=markers)
        sns.heatmap(X_pred, ax=ax2, xticklabels=markers)

        ax1.set_title("X Test")
        ax2.set_title("Reconstructed X Test")
        fig.tight_layout()

        save_path = Path("VAE", "results", f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    @staticmethod
    def plot_r2_scores(r2_scores: pd.DataFrame, file_name: str):
        ax = sns.barplot(x="Marker", y="Score", data=r2_scores)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()
        fig.tight_layout()
        save_path = Path("VAE", "results", f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()
