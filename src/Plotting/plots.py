import logging
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class Plotting:
    # The base path such as AE or VAE. This is the path where the files will be stored
    __base_path: Path

    def __init__(self, base_path: Path):
        self.__base_path = base_path

    def plot_model_performance(self, history, sub_directory: str, file_name: str):
        logger.info("Plotting model performance")
        plt.figure(num=None, figsize=(6, 4), dpi=90)
        for key in history.history:
            plt.plot(history.history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), sub_directory)
        plt.close()

    def plot_reconstructed_markers(self, X, X_pred, markers, sub_directory: str, file_name: str):
        logging.info("Plotting reconstructed intensities")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(X, ax=ax1, xticklabels=markers)
        sns.heatmap(X_pred, ax=ax2, xticklabels=markers)

        ax1.set_title("X Test")
        ax2.set_title("Reconstructed X Test")
        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), sub_directory)
        plt.close()

    def plot_r2_scores(self, r2_scores: pd.DataFrame, sub_directory: str, file_name: str):
        ax = sns.barplot(x="Marker", y="Score", data=r2_scores)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()
        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), sub_directory)
        plt.close()

    def plot_markers(self, X_train, X_test, X_val, markers, sub_directory: str, file_name: str):
        logging.info("Plotting markers")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(X_train, ax=ax1, xticklabels=markers)
        sns.heatmap(X_test, ax=ax2, xticklabels=markers)
        sns.heatmap(X_val, ax=ax3, xticklabels=markers)

        ax1.set_title("X Train")
        ax2.set_title("X Test")
        ax3.set_title("X Validation")
        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), sub_directory)
        plt.close()

    def plot_r2_scores_comparison(self, ae_r2_scores: pd.DataFrame, vae_r2_scores: pd.DataFrame):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(19, 20), dpi=300, sharex=False)
        sns.barplot(x='Marker', y='Score', data=ae_r2_scores, ax=ax1)
        sns.barplot(x='Marker', y='Score', data=vae_r2_scores, ax=ax2)

        differences = pd.DataFrame()
        differences['Vae'] = vae_r2_scores['Score']
        differences['Ae'] = ae_r2_scores['Score']
        differences['Marker'] = vae_r2_scores['Marker']
        differences['Difference'] = differences["Vae"] - differences["Ae"]
        sns.barplot(x="Marker", y="Difference", data=differences, ax=ax3)

        ax1.set_title("AE R2 Scores")
        ax2.set_title("VAE R2 Scores")
        ax3.set_title("Difference VAE to AE")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)

        fig.tight_layout()
        save_path = Path(self.__base_path, "r2_comparison.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()
