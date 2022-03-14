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

    def __init__(self, base_path: Path, args):
        self.__base_path = base_path
        if args.tracking_url is not None:
            mlflow.set_tracking_uri = args.tracking_url

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

    def plot_reconstructed_markers(self, test_data, reconstructed_data, markers, mlflow_directory: str,
                                   file_name: str):
        logging.info("Plotting reconstructed intensities")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)
        sns.heatmap(test_data, ax=ax1, xticklabels=markers)
        sns.heatmap(reconstructed_data, ax=ax2, xticklabels=markers)

        ax1.set_title("X Test")
        ax2.set_title("Reconstructed X Test")
        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), mlflow_directory)
        plt.close()

    def plot_r2_scores(self, r2_scores: pd.DataFrame, mlflow_directory: str, file_name: str):
        ax = sns.barplot(x="Marker", y="Score", data=r2_scores)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()
        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), mlflow_directory)
        plt.close()

    def plot_markers(self, train_data, test_data, markers, mlflow_directory: str, file_name: str, val_data=None):
        logging.info("Plotting markers")

        if val_data is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), dpi=300, sharex=True)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), dpi=300, sharex=True)

        sns.heatmap(train_data, ax=ax1, xticklabels=markers)
        sns.heatmap(test_data, ax=ax2, xticklabels=markers)

        if val_data is not None:
            sns.heatmap(val_data, ax=ax3, xticklabels=markers)

        ax1.set_title("X Train")
        ax2.set_title("X Test")

        if val_data is not None:
            ax3.set_title("X Validation")

        fig.tight_layout()

        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), mlflow_directory)
        plt.close()

    def plot_weights(self, weights, markers: list, mlflow_directory: str, fig_name: str):
        df = pd.DataFrame(weights, columns=markers)
        ax = sns.heatmap(df)

        # Markers count is 26. As all are present we can change the labels
        if len(df) == 26:
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(markers)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(markers)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        fig = ax.get_figure()
        fig.tight_layout()
        save_path = Path(self.__base_path, f"{fig_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path), mlflow_directory)
        plt.close()

    def compare_vae_to_ae_scores(self, ae_scores: pd.DataFrame, vae_scores: pd.DataFrame):
        """
        Plots a bar plot comparing ae scores vs vae scores
        @param ae_scores:
        @param vae_scores:
        @return:
        """
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(19, 20), dpi=300, sharex=False)
        sns.barplot(x='Marker', y='Score', data=ae_scores, ax=ax1)
        sns.barplot(x='Marker', y='Score', data=vae_scores, ax=ax2)

        # Create difference dataframe
        differences = pd.DataFrame(columns=["Marker", "Difference"], data={"Marker": ae_scores["Marker"].values,
                                                                           "Difference": vae_scores["Score"] -
                                                                                         ae_scores[
                                                                                             "Score"]})
        sns.barplot(x="Marker", y="Difference", data=differences, ax=ax3)

        ax1.set_title("AE R2 Scores")
        ax2.set_title("VAE R2 Scores")
        ax3.set_title("Difference VAE to AE")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)

        fig.tight_layout()
        save_path = Path(self.__base_path, "mean_r2_comparison.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_score_bar_plot(self, r2_scores: pd.DataFrame, compare_score: pd.DataFrame, r2_score_title: str,
                          compare_title: str, file_name: str):
        """
        Plots the given r2 scores and calculate the difference between the r2 scores and compare_score
        @param r2_scores: A pandas dataframe with r2 scores
        @param compare_score: The pandas dataframe for comparison
        @param r2_score_title: The title for the first given dataframe
        @param compare_title: The title for the comparison dataframe
        @param file_name: The file name s
        @return:
        """
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(19, 20), dpi=300, sharex=False)
        sns.barplot(x='Marker', y='Score', data=r2_scores, ax=ax1)
        sns.barplot(x='Marker', y='Score', data=compare_score, ax=ax2)

        # Create difference dataframe
        differences = pd.DataFrame(columns=["Marker", "Difference"], data={"Marker": r2_scores["Marker"].values,
                                                                           "Difference": r2_scores["Score"] -
                                                                                         compare_score[
                                                                                             "Score"]})
        sns.barplot(x="Marker", y="Difference", data=differences, ax=ax3)

        ax1.set_title(r2_score_title)
        ax2.set_title(compare_title)
        ax3.set_title(f"Difference {r2_score_title} vs. {compare_title} ")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)

        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_scores_distribution(self, r2_scores: dict, file_name: str):
        """
        Plots a graph using a dictionary
        @param r2_scores:
        @param file_name:
        @return:
        """
        num_rows = 1
        if len(r2_scores.items()) > 3:
            num_rows = int(len(r2_scores.items()) / 3)

        fig, axs = plt.subplots(ncols=3, nrows=num_rows, figsize=(25, 20), dpi=300, sharex=False)

        col: int = 0
        row: int = 0
        for experiment_name, r2_score in r2_scores.items():
            sns.boxplot(data=r2_score, ax=axs[row, col])
            axs[row, col].set_title(experiment_name)
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90)
            col += 1

            if col == 3:
                row += 1
                col = 0

        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def plot_weights_distribution(self, weights: pd.DataFrame, layer: str, prefix: str = None):

        fig, ax1 = plt.subplots(ncols=1, figsize=(28, 20), dpi=300, sharex=False)
        df = pd.DataFrame(columns=["Weights", "Markers"])
        for column, weights in weights.iteritems():
            for weight in weights.values:
                df = df.append({
                    "Markers": column,
                    "Weights": weight
                }, ignore_index=True)

        sns.displot(df, x="Weights", hue="Markers", legend=True)
        fig.tight_layout()
        plt.legend(loc='lower center')

        if prefix is not None:
            save_path = Path(self.__base_path, f"{prefix}_{layer}_weights_distribution.png")
        else:
            save_path = Path(self.__base_path, f"{layer}_weights_distribution.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def plot_r2_score_differences(self, r2_score_difference: pd.DataFrame, prefix: str):
        ax = sns.barplot(x='Marker', y='Score', data=r2_score_difference)
        ax.set_title("Mean R2 Score Difference")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig = ax.get_figure()
        fig.tight_layout()
        save_path = Path(self.__base_path, f"{prefix}_mean_r2_difference.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()

    def r2_score_distribution(self, combined_r2_scores: pd.DataFrame, comparing_r2_scores: pd.DataFrame, title: str,
                              comparing_title: str, file_name: str):
        """
        Plots the distribution of r2 scores
        @param combined_r2_scores:
        @param comparing_r2_scores:
        @param title:
        @param comparing_title:
        @param file_name:
        @return:
        """
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(19, 20), dpi=300, sharex=False)
        sns.boxplot(data=combined_r2_scores, ax=ax1)
        sns.boxplot(data=comparing_r2_scores, ax=ax2)

        ax1.set_title(title)
        ax2.set_title(comparing_title)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

        fig.tight_layout()
        save_path = Path(self.__base_path, f"{file_name}.png")
        plt.savefig(save_path)
        mlflow.log_artifact(str(save_path))
        plt.close()
